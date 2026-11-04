"""Read-only CPU evidence for motion-sharing figures on an existing stream.

The no-sharing comparator re-encodes the SAME integer translation field with
zero regional symbols, identical quantization, births, deletions and compressor.
Each packet uses its original preceding reference; it is an encoding diagnostic,
not a second training run or independently streamed alternative sequence.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.model import rigid_tree_codec as codec, sparse_integer_codec as integer
from pipeline.model.stream_packet import read_stream_step
from pipeline.model.spatial_gate import spatial_accumulate
from scripts.benchmark_rigid_residual_tree import _load_reference


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_shared_packet(packet, total, region_count, old_count):
    """Keep all non-motion payloads unchanged; use the same production codec."""
    if np.abs(total).max()>32767:
        raise ValueError('No-sharing total overflows existing signed16 grid; never clip')
    raw=codec.uncompressed_motion_packet(packet)
    reader=integer._Reader(raw[:-codec.CHECKSUM_BYTES])
    integer._spatial_prefix(reader)
    prefix=bytearray(raw[:reader.cursor])
    source=codec._Reader(raw);source.cursor=reader.cursor
    codec._sparse_read(source,region_count-1,delta_rows=True)
    codec._sparse_read(source,old_count,delta_rows=True)
    tail=raw[source.cursor:-codec.CHECKSUM_BYTES]
    # No regional motion needs refined gates. Keep the same reference format
    # but give this comparator the minimal valid frontier (root open only).
    gate_bytes=(region_count+7)//8
    prefix[-gate_bytes:]=bytes([1])+bytes(gate_bytes-1)
    body=prefix
    zero_region=torch.zeros(region_count-1,3,dtype=torch.int16)
    codec._sparse_write(body,zero_region,zero_region)
    codec._sparse_write(body,torch.from_numpy(total.astype(np.int16)),
                        torch.zeros(old_count,3,dtype=torch.int16))
    body.extend(tail)
    body.extend(hashlib.sha256(body).digest()[:codec.CHECKSUM_BYTES])
    return codec.compress_motion_packet(body,mode='auto')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--example-frame',type=int,default=149)
    p.add_argument('--anchors-per-frame',type=int,default=1200)
    args=p.parse_args();run=args.run.resolve();out=args.output.resolve()
    if out.exists() or out.is_relative_to(run):raise ValueError('Use a new evidence directory outside the source run')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    stream=run/'frame_storage/rigid_tree'
    state,raw=_load_reference(stream/'reference_frame0.npz',None)
    region_count=len(state.parents);depth=state.spatial_paths.shape[1];rest=raw['sh_rest'].shape[1]
    del raw
    frames=sorted(int(f.stem.split('_')[-1]) for f in stream.glob('frame_*.json'))
    assert frames==list(range(1,300)), 'Use the complete300 baseline'
    source={str(f.relative_to(run)):sha(f) for f in stream.glob('*') if f.is_file()}
    rows=[];sample_rows=[];started=time.perf_counter()
    for frame in frames:
        packet=(stream/f'motion_{frame:03d}.bin').read_bytes()
        step=read_stream_step(stream,frame,region_count=region_count,depth=depth,rest=rest)
        if step.region_step!=step.gs_step or not step.relative:
            raise ValueError('This matched-symbol diagnostic requires equal region/GS grids and relative regions')
        previous=state
        state=codec.decode_motion(packet,previous)
        manifest=json.loads((stream/f'frame_{frame:03d}.json').read_text())
        assert state.digest().hex()==manifest['decoded_state_sha256'],frame
        keep=np.delete(np.arange(len(previous.xyz)),step.deleted)
        assert np.array_equal(keep,state.survivor_rows.numpy())
        xyz=previous.xyz.numpy()[keep]
        owner=state.owner[:step.old_count].numpy()
        regional=spatial_accumulate(torch.from_numpy(step.regional_symbols),previous.parents).numpy()[owner].astype(np.int64)
        residual=np.zeros((step.old_count,3),dtype=np.int64)
        residual[step.gs_rows]=step.gs_symbols
        total=regional+residual
        # Check the original float32 arithmetic exactly; do not equate the
        # one-add alternative to the original two-add arithmetic without audit.
        expected=(torch.from_numpy(xyz)+torch.from_numpy(regional).float()*step.region_step+
                  torch.from_numpy(residual).float()*step.gs_step)
        assert torch.equal(expected,state.xyz[:step.old_count]),frame
        counter=no_shared_packet(packet,total,region_count,step.old_count)
        alternative=codec.decode_motion(counter,previous)
        assert torch.equal(alternative.rotation,state.rotation),frame
        assert np.array_equal(alternative.survivor_rows.numpy(),keep)
        # Read back the new raw sparse block and check ALL integer vectors.
        alt_raw=codec.uncompressed_motion_packet(counter)
        rr=integer._Reader(alt_raw[:-16]);integer._spatial_prefix(rr)
        reader=codec._Reader(alt_raw);reader.cursor=rr.cursor
        rt,rot=codec._sparse_read(reader,region_count-1,delta_rows=True)
        gt,grot=codec._sparse_read(reader,step.old_count,delta_rows=True)
        assert not rt.any() and not rot.any() and not grot.any()
        assert np.array_equal(gt.numpy().astype(np.int64),total),frame
        difference=float((alternative.xyz-state.xyz).abs().max())
        if difference/step.gs_step>.02:
            raise ValueError('Unexpected alternative float32 reordering error')
        norm=np.linalg.norm(total,axis=1)
        top_count=max(1,len(norm)//4)
        moving=np.argpartition(norm,-top_count)[-top_count:]
        rng=np.random.default_rng(9051+frame)
        anchors=rng.choice(moving,min(args.anchors_per_frame,len(moving)),replace=False)
        distances,neighbours=cKDTree(xyz).query(xyz[anchors],k=2,workers=1)
        nearest=np.where(neighbours[:,0]!=anchors,neighbours[:,0],neighbours[:,1])
        random=rng.integers(0,len(xyz),len(anchors))
        random[random==anchors]=(random[random==anchors]+1)%len(xyz)
        near_motion=np.linalg.norm(total[anchors]-total[nearest],axis=1)
        random_motion=np.linalg.norm(total[anchors]-total[random],axis=1)
        extent=step.gs_step/6e-5
        # Store only sampled scalar pairs, not raw scenes or images.
        sample_rows.extend(dict(frame=frame,pair=kind,distance_extent=float(distance),
                                motion_difference_bins=float(motion))
            for kind,ids,motions in [('near',nearest,near_motion),('random',random,random_motion)]
            for distance,motion in zip(np.linalg.norm(xyz[anchors]-xyz[ids],axis=1)/extent,motions))
        row=dict(frame=frame,old_count=step.old_count,birth_count=step.birth_count,
            total_nonzero=int(np.any(total!=0,axis=1).sum()),residual_nonzero=len(step.gs_rows),
            total_L1=int(np.abs(total).sum()),residual_L1=int(np.abs(residual).sum()),
            total_L2_sum=float(norm.sum()),residual_L2_sum=float(np.linalg.norm(residual,axis=1).sum()),
            actual_motion_bytes=len(packet),no_shared_motion_bytes=len(counter),
            complete_dynamic_bytes=step.stored_bytes,region_count=len(np.unique(owner)),
            motion_grid_world=step.gs_step,nearest_difference_bins=float(near_motion.mean()),
            random_difference_bins=float(random_motion.mean()),anchors=len(anchors),
            float_reordering_max_world=difference,float_reordering_max_bins=difference/step.gs_step,
            quantized_motion_exact=True,state_sha=state.digest().hex(),
            alternative_packet_sha=hashlib.sha256(counter).hexdigest())
        rows.append(row)
        if frame==args.example_frame:
            np.savez_compressed(out/'example.npz',previous_xyz=xyz,current_xyz=state.xyz[:step.old_count].numpy(),
                total=total.astype(np.int32),shared=regional.astype(np.int32),residual=residual.astype(np.int32),
                owner=owner,region_centres=previous.spatial_centers.numpy(),grid_world=step.gs_step)
        if frame%20==0 or frame==299:
            print(json.dumps(dict(frame=frame,seconds=round(time.perf_counter()-started,1),
                current_motion_KB=np.mean([r['actual_motion_bytes'] for r in rows])/1000,
                matched_no_share_KB=np.mean([r['no_shared_motion_bytes'] for r in rows])/1000)),flush=True)
            (out/'progress.json').write_text(json.dumps(dict(last_frame=frame,rows=rows[-3:]),indent=2)+'\n')
    for name,data in [('per_frame.csv',rows),('locality_pairs.csv',sample_rows)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    total_points=sum(r['old_count'] for r in rows)
    total_nonzero=sum(r['total_nonzero'] for r in rows);residual_nonzero=sum(r['residual_nonzero'] for r in rows)
    a=sum(r['actual_motion_bytes'] for r in rows);b=sum(r['no_shared_motion_bytes'] for r in rows)
    summary=dict(schema='motion_sharing_report_v1',run=str(run),frames=299,example_frame=args.example_frame,
        actual_motion_mean_KB=a/299/1000,matched_no_share_motion_mean_KB=b/299/1000,
        motion_byte_reduction=1-a/b,actual_motion_total_MB=a/1e6,matched_no_share_motion_total_MB=b/1e6,
        complete_dynamic_mean_KB=sum(r['complete_dynamic_bytes'] for r in rows)/299/1000,
        total_point_events=total_points,total_nonzero_fraction=total_nonzero/total_points,
        residual_nonzero_fraction=residual_nonzero/total_points,
        residual_row_reduction=1-residual_nonzero/total_nonzero,
        residual_L1_reduction=1-sum(r['residual_L1'] for r in rows)/sum(r['total_L1'] for r in rows),
        residual_L2_reduction=1-sum(r['residual_L2_sum'] for r in rows)/sum(r['total_L2_sum'] for r in rows),
        nearest_difference_bins=float(np.mean([r['nearest_difference_bins'] for r in rows])),
        random_difference_bins=float(np.mean([r['random_difference_bins'] for r in rows])),
        nearest_difference_reduction=1-np.mean([r['nearest_difference_bins'] for r in rows])/np.mean([r['random_difference_bins'] for r in rows]),
        all299_state_hashes_verified=True,all_integer_motion_fields_exact=True,
        maximum_alternative_float32_difference_bins=max(r['float_reordering_max_bins'] for r in rows),
        locality_scope='Same anchors sampled from top25% displacement magnitude per frame, nearest nonself versus random nonself partner. Fitted model motion, not independent motion ground truth.',
        coding_scope='Each no-sharing packet uses its original previous reference; regional symbols zero, minimal valid gates, total point symbols on identical grid. Original births/deletions/pose tail preserved, same production auto compressor. Complete motion packet bytes include structure and newborn poses. Not a retraining ablation or a new sequential stream. Float32 addition reorder differences separately audited.',
        source_script_sha=sha(Path(__file__)),input_sha256=source)
    assert source=={str(f.relative_to(run)):sha(f) for f in stream.glob('*') if f.is_file()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2),flush=True)


if __name__=='__main__':main()

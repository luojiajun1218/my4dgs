"""Verify all300 GPU-decoded frames, then time checked incremental playback.

Verification can share a GPU and makes no speed claim. Timed playback requires
exclusive GPU access, a matching complete verification certificate, and exactly
299 dynamic updates. Future sparse packets are bounded to two; future dense
decoded frames are never cached.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import benchmark_rigid_residual_tree as base
from pipeline.model.gpu_stream import GPUStream
from pipeline.model.stream_packet import read_stream_step
from pipeline.model.birth_quantization import reference_digest
from libgs.metric import psnr


def source_hashes():
    files=[*ROOT.glob('pipeline/**/*.py'),ROOT/'scripts/benchmark_gpu_stream.py',
           ROOT/'scripts/benchmark_rigid_residual_tree.py',ROOT/'scripts/replay_rigid_tree.py',
           *ROOT.glob('extensions/stream_update/*')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def input_hashes(stream):
    files=[stream/'reference_frame0.npz']
    for frame in range(1,300):
        files += [stream/f'frame_{frame:03d}.json',stream/f'motion_{frame:03d}.bin']
        manifest=json.loads((stream/f'frame_{frame:03d}.json').read_text())
        for key in ('birth','opacity_update'):
            if manifest.get(key):
                expected=('birth' if key=='birth' else 'opacity')+f'_{frame:03d}.bin'
                if manifest[key]!=expected:
                    raise ValueError('Unexpected input packet path')
                files.append(stream/expected)
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def camera_signature(view):
    return dict(shape=list(view.image.shape),fov=[float(view.fovx),float(view.fovy)],
        matrices={key:hashlib.sha256(getattr(view,key).cpu().contiguous().numpy().tobytes()).hexdigest()
            for key in ('world_view_transform','full_proj_transform','camera_center')})


def make(run):
    config=base._load_config(run)
    if config.data.extra_dataset_kwargs['num_frames']!=300:
        raise ValueError('This protocol requires the complete300-frame run')
    module,dm=base._make_module(config,torch.device('cuda'))
    if dm.num_frames!=300:
        raise ValueError('The dataset must contain300 frames')
    stream=run/'frame_storage/rigid_tree'
    state,raw=base._load_reference(stream/'reference_frame0.npz',None)
    capacity=len(state.xyz)+299*int(config.module.birth.max_new_gaussians)
    return config,module,dm,stream,state,raw,capacity


@torch.no_grad()
def verify(run):
    before_sources=source_hashes()
    config,module,dm,stream,state,raw,capacity=make(run)
    before_inputs=input_hashes(stream)
    player=GPUStream(state,raw,capacity=capacity)
    rows=[];signature=None;static=True
    for frame in range(300):
        complete=0
        if frame:
            step=read_stream_step(stream,frame,region_count=player.regions,depth=player.depth,
                                  rest=player.rest,max_points=capacity)
            player.advance(step)
            state,raw=base._decode_frame(stream,frame,state,raw)
            complete=step.stored_bytes
        actual=player.capture_raw();decoded=player.capture_state(actual)
        for name in base.RAW_FIELDS:
            if actual[name].numpy().tobytes()!=raw[name].numpy().tobytes():
                raise ValueError(f'GPU raw {name} differs from independent CPU at frame{frame}')
        if decoded.digest()!=state.digest():
            raise ValueError(f'GPU state differs from independent CPU at frame{frame}')
        raw_sha=reference_digest(actual)
        if frame and (decoded.digest()!=step.digest or raw_sha!=step.raw_digest):
            raise ValueError(f'GPU decode differs from original manifest at frame{frame}')
        player.verify_render_inputs(raw)
        view,count=base._cam00(dm,frame,torch.device('cuda'))
        if count!=1 or tuple(view.image.shape)!=(3,1008,1345):
            raise ValueError('Full300 verification needs the original1345x1008 cam00 test split')
        current=camera_signature(view)
        if signature is None: signature=current
        static=static and signature==current
        base._install_raw(module,raw,torch.device('cuda'))
        module.renderer.gaussians=module.gaussians
        original=module.forward(view)['render'].clamp(0.,1.)
        module.renderer.gaussians=player.view
        image=module.forward(view)['render'].clamp(0.,1.)
        if not torch.equal(image,original):
            raise ValueError(f'GPU player image differs at frame{frame}: {float((image-original).abs().max())}')
        score=float(psnr(image,view.image))
        difference=0.
        if frame:
            expected=json.loads((run/f'frame_summary/frame_{frame:03d}.json').read_text())['evaluation']['test']['psnr']
            difference=score-expected
            if abs(difference)>1e-4:
                raise ValueError(f'Saved PSNR differs at frame{frame}: {difference}')
        rows.append(dict(frame=frame,gaussians=len(actual['xyz']),psnr=score,saved_psnr_difference=difference,
            pixel_max_abs_difference=0.,state_sha=decoded.digest().hex(),raw_sha=raw_sha,
            complete_bytes=complete,render_sha=hashlib.sha256(image.cpu().contiguous().numpy().tobytes()).hexdigest()))
        if frame%30==29 or frame==299:
            print('VERIFIED_RENDER',frame+1,'of300',flush=True)
        del view,image,original
    if source_hashes()!=before_sources or input_hashes(stream)!=before_inputs:
        raise ValueError('Source or packets changed during verification')
    return dict(schema='gpu_stream_complete_verification_v1',run=str(run),frames=300,dynamic_frames=299,
        all_raw_states_renderer_inputs_and_pixels_exact=True,static_camera_all300=static,
        camera=signature,source_hashes=before_sources,input_hashes=before_inputs,
        all300_psnr=float(np.mean([row['psnr'] for row in rows])),
        complete_dynamic_KB=sum(row['complete_bytes'] for row in rows)/299/1000,
        reference_bytes=(stream/'reference_frame0.npz').stat().st_size,
        scope='All300 frames independently CPU/GPU decoded and re-rendered at cam00. Raw/state bits, activated renderer inputs, actual RGB pixels, saved PSNR and original manifests checked. No FPS claim; verification may share a GPU.',rows=rows)


def ordered_steps(stream,player,capacity,workers=2):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit(frame):
            return pool.submit(read_stream_step,stream,frame,region_count=player.regions,
                depth=player.depth,rest=player.rest,max_points=capacity)
        futures=[submit(frame) for frame in (1,2)]
        for frame in range(1,300):
            current=futures.pop(0).result()
            if frame+2<300: futures.append(submit(frame+2))
            yield current


@torch.no_grad()
def playback(run,certificate,rounds):
    if (certificate['run']!=str(run) or certificate['frames']!=300 or
            not certificate['all_raw_states_renderer_inputs_and_pixels_exact'] or
            not certificate['static_camera_all300'] or certificate['source_hashes']!=source_hashes()):
        raise ValueError('Complete verification certificate does not match this player/run')
    config,module,dm,stream,state,raw,capacity=make(run)
    if input_hashes(stream)!=certificate['input_hashes']:
        raise ValueError('The verified packets changed')
    view,count=base._cam00(dm,299,torch.device('cuda'))
    if count!=1 or camera_signature(view)!=certificate['camera']:
        raise ValueError('Playback camera differs from the verified sequence')
    passes=[]
    for repeat in range(rounds):
        gpu_check=base.check_gpu_exclusivity()
        player=GPUStream(state,raw,capacity=capacity)
        module.renderer.gaussians=player.view
        for _ in range(10): module.forward(view)['render'].clamp(0.,1.)
        torch.cuda.synchronize()
        started=time.perf_counter();host=[]
        iterator=iter(ordered_steps(stream,player,capacity))
        for frame in range(1,300):
            wait=time.perf_counter();step=next(iterator);ready=time.perf_counter()
            if step.frame!=frame: raise ValueError('Playback order changed')
            module.renderer.gaussians=player.advance(step)
            updated=time.perf_counter()
            image=module.forward(view)['render'].clamp(0.,1.)
            submitted=time.perf_counter()
            host.append(dict(frame=frame,wait_for_packet_seconds=ready-wait,
                host_update_seconds=updated-ready,host_render_call_seconds=submitted-updated))
        # Exhaust/close the bounded producer inside the timed interval.
        try: next(iterator)
        except StopIteration: pass
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        base.check_gpu_exclusivity()
        actual=player.capture_raw();decoded=player.capture_state(actual)
        if (decoded.digest().hex()!=certificate['rows'][-1]['state_sha'] or
                reference_digest(actual)!=certificate['rows'][-1]['raw_sha'] or
                hashlib.sha256(image.cpu().contiguous().numpy().tobytes()).hexdigest()!=certificate['rows'][-1]['render_sha']):
            raise ValueError('Timed playback final state or actual render differs from verification')
        passes.append(dict(round=repeat,dynamic_frames=299,seconds=elapsed,fps=299/elapsed,
                           gpu_check=gpu_check,host=host))
        print('PLAYBACK',repeat,299/elapsed,'FPS',flush=True)
        del player
    return dict(schema='gpu_stream_complete_playback_v1',run=str(run),dynamic_frames=299,
        rounds=rounds,fps=299*rounds/sum(row['seconds'] for row in passes),passes=passes,
        all300_psnr=certificate['all300_psnr'],complete_dynamic_KB=certificate['complete_dynamic_KB'],
        source_hashes=source_hashes(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(),
        scope='Wall time includes file reads and packet checksums, two CPU sparse parsers with at most two future packets, ordered state-link and exact opacity predictor checks, host attribute prediction, asynchronous H2D uploads, GPU geometry/routing and actual Module.forward+clamp for all299 updates. CUDA is synchronized at pass boundaries. Initial reference loading, JIT/allocations, camera/GT loading and full-state SHA/preflight validation are startup costs excluded from timing. No dense future frames are cached. Final state/raw/pixels checked after each pass; GPU exclusivity required.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--certificate',type=Path)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.rounds<1: parser.error('rounds must be positive')
    torch.set_num_threads(4)
    run=args.run.resolve()
    if args.verify_only:
        result=verify(run)
    elif args.certificate:
        result=playback(run,json.loads(args.certificate.read_text()),args.rounds)
    else:
        parser.error('Run --verify-only before supplying a complete --certificate')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()

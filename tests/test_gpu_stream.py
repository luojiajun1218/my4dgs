import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch

from pipeline.model import rigid_tree_codec as codec
from pipeline.model import rigid_tree_birth_codec as birth
from pipeline.model import sparse_integer_codec as layout
from pipeline.model.birth_quantization import RAW_FIELDS,reference_digest
from pipeline.model.stream_packet import _birth_attributes,_opacity_symbols,read_stream_step
from tests.test_codec import (reference_raw,motion_model,appearance_fixture,
                              opacity_fixture,wrapped)


def make_stream(directory,births=2,rotate=False):
    raw=reference_raw();model=motion_model(raw);prior=model.frame_reference_state
    rows=torch.tensor([0,2,3,4,5,6,7]);n=len(rows)
    rt=torch.zeros(len(prior.parents),3,dtype=torch.int16);rr=torch.zeros_like(rt)
    gt=torch.zeros(n,3,dtype=torch.int16);gr=torch.zeros_like(gt)
    gt[2]=torch.tensor([1,-2,0],dtype=torch.int16)
    if rotate:gr[0,1]=1
    sources=torch.tensor([0,4][:births],dtype=torch.long)
    xyz,q=codec.compose_decoded_survivors(prior,2.,rows,prior.owner[rows],prior.parents,
        rt,rr,gt,gr,.001,.001,spatial_open=prior.spatial_open,spatial_relative=True)
    newborn_xyz=xyz[sources]+torch.tensor([.002,0.,-.002])
    payload=codec.encode_motion(prior,1,2.,rows,prior.owner[rows],prior.parents,rt,rr,gt,gr,
        sources,newborn_xyz,q[sources],.001,.001,spatial_open=prior.spatial_open,
        spatial_relative=True,birth_pose_steps=(.002,.001),
        gs_translation_step_extent=.001,gs_rotation_step_rad=.001)
    payload=codec.compress_motion_packet(payload)
    decoded=codec.decode_motion(payload,prior)
    current={name:value[rows].clone() for name,value in raw.items()}
    current['xyz']=decoded.xyz[:n];current['rotation']=decoded.rotation[:n]
    (directory/'motion_001.bin').write_bytes(payload)
    if births:
        predictor={name:current[name][sources] for name in birth.ATTRIBUTES}
        symbols={name:torch.ones_like(value) for name,value in predictor.items()}
        steps={name:.125 for name in predictor}
        newborn={name:predictor[name]+symbols[name]*steps[name] for name in predictor}
        packet=birth.encode_birth_symbols(newborn,n,symbols=symbols,steps=steps,
            predictor=predictor,source_rows=sources)
        assert packet is not None
        (directory/'birth_001.bin').write_bytes(packet)
        for name in birth.ATTRIBUTES:current[name]=torch.cat((current[name],newborn[name]))
    current['xyz'],current['rotation']=decoded.xyz,decoded.rotation
    manifest=dict(frame=1,motion='motion_001.bin',birth='birth_001.bin' if births else None,
        opacity_update=None,old_count=n,birth_count=births,total_count=n+births,
        previous_count=len(prior.xyz),decoded_state_sha256=decoded.digest().hex(),
        raw_state_sha256=reference_digest(current))
    (directory/'frame_001.json').write_text(json.dumps(manifest))
    return prior,raw,decoded,current,manifest


class StreamPacketTests(unittest.TestCase):
    def test_birth_symbols_and_absolute_fallback_match_original_attribute_decoder(self):
        for rest in (0,3):
            base,raw,symbols,steps,predictor,sources,xyz,q=appearance_fixture(rest=rest)
            packets=[birth.encode_birth_attributes(raw,3),birth.encode_birth_symbols(raw,3,
                symbols=symbols,steps=steps,predictor=predictor,source_rows=sources)]
            for packet in packets:
                parsed=_birth_attributes(packet,3,2,rest)
                original=birth.decode_birth_attributes(packet,base,xyz,q,sources)
                for index,name in enumerate(birth.ATTRIBUTES):
                    actual=parsed.values[name].astype(np.float32)
                    if parsed.steps is not None:
                        actual=base[name][sources].numpy()+actual*np.float32(parsed.steps[index])
                    self.assertEqual(actual.tobytes(),original[name].numpy().tobytes())
                damaged=bytearray(packet);damaged[-1]^=1
                with self.assertRaises(ValueError):_birth_attributes(damaged,3,2,rest)

    def test_opacity_parsing_keeps_original_rows_symbols_and_all_envelope_checks(self):
        from pipeline.model.legacy_opacity_codec import decode_opacity_delta,HEADER
        import zlib
        reference,symbols,raw=opacity_fixture()
        planar=layout.encode_opacity_layout(raw)
        for packet in [raw[:HEADER.size]+zlib.compress(raw[HEADER.size:]),
                       *wrapped(planar,len(raw),layout.OPACITY_MAGICS)]:
            parsed=_opacity_symbols(packet,1,len(reference))
            original=decode_opacity_delta(packet,reference,frame_index=1)
            np.testing.assert_array_equal(parsed.rows,original.rows.numpy())
            np.testing.assert_array_equal(parsed.values[:,0],original.symbols.numpy())
            with self.assertRaises(ValueError):_opacity_symbols(packet,2,len(reference))
            with self.assertRaises(ValueError):_opacity_symbols(packet+b'x',1,len(reference))

    def test_bulk_pose_tail_preserves_zero_birth_and_rejects_noncanonical_sources(self):
        valid=layout._varint(0)+struct.pack('<2d',.001,.002)+b'\0'*7+b'\0'*16
        layout._check_motion_tail(valid,1,3,7,offset=True)
        layout._check_motion_tail(struct.pack('<2d',.001,.002)+b'\0'*16,0,3,7,offset=True)
        for malformed in (b'\x80\0'+valid[1:],layout._varint(3)+valid[1:],valid[:-1]):
            with self.assertRaises(ValueError):layout._check_motion_tail(malformed,1,3,7,offset=True)

    def test_whole_packet_population_lineage_and_required_file_validation(self):
        for births in (0,2):
            with self.subTest(births=births),tempfile.TemporaryDirectory() as folder:
                directory=Path(folder);prior,_,decoded,_,manifest=make_stream(directory,births)
                arguments=dict(region_count=len(prior.parents),depth=prior.spatial_paths.shape[1],rest=3)
                step=read_stream_step(directory,1,**arguments)
                self.assertEqual(step.digest,decoded.digest())
                np.testing.assert_array_equal(step.sources,decoded.birth_sources.numpy())
                with self.assertRaises(ValueError):read_stream_step(directory,1,max_points=6,**arguments)
                changed={**manifest,'previous_count':100}
                (directory/'frame_001.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'population'):read_stream_step(directory,1,**arguments)
                changed={**manifest,'motion':'../unexpected.bin'}
                (directory/'frame_001.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'filename'):read_stream_step(directory,1,**arguments)

    def test_unsupported_old_rotation_updates_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);prior,*_=make_stream(directory,rotate=True)
            with self.assertRaisesRegex(ValueError,'zero old rotation'):
                read_stream_step(directory,1,region_count=len(prior.parents),depth=prior.spatial_paths.shape[1],rest=3)


@unittest.skipUnless(torch.cuda.is_available(),'GPU incremental stream checks')
class GPUStreamTests(unittest.TestCase):
    def test_real_packet_births_deletions_and_empty_birth_match_cpu_bits(self):
        from pipeline.model.gpu_stream import GPUStream
        for births in (0,2):
            with self.subTest(births=births),tempfile.TemporaryDirectory() as folder:
                directory=Path(folder);prior,raw,expected,attributes,_=make_stream(directory,births)
                player=GPUStream(prior,raw,capacity=32)
                step=read_stream_step(directory,1,region_count=len(prior.parents),depth=player.depth,rest=player.rest)
                player.advance(step)
                actual=player.capture_raw();state=player.capture_state(actual)
                self.assertEqual(state.digest(),expected.digest())
                self.assertEqual(reference_digest(actual),reference_digest(attributes))
                player.verify_render_inputs(attributes)


if __name__=='__main__':unittest.main()

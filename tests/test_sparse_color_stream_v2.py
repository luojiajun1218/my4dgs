import json
import tempfile
import struct
import unittest
from pathlib import Path

import torch
from pipeline.model import full_attribute_update as fua
from scripts.replay_rigid_tree import decode_frame
from scripts.benchmark_rigid_residual_tree import _actual_frame_storage
from tests.test_full_attribute_update import make_stream, legacy_packets, reference


class SparseColorV2Tests(unittest.TestCase):
    def test_encoder_selection_and_sparse_formats(self):
        ref = reference(8)
        sym = {k: torch.zeros_like(v) for k, v in ref.items()}
        sel = torch.zeros(8, dtype=torch.bool); sel[2] = True
        sym['sh_dc'][2, 0, 0] = 2
        fua.validate_sparse_color_selection(sym, sel, 1)
        packet = fua.encode_full_attribute_delta(ref, sym, frame_index=10, steps=(.005,.001,.001), sparse_only=True)
        self.assertIn(packet[:4], (fua.SPARSE_MAGIC, fua.PLANAR_MAGIC))
        for packet in legacy_packets(ref, sym)[1:]:
            fua.decode_full_attribute_delta(packet, ref, frame_index=10, color_only_budget=1)
        with self.assertRaises(ValueError): fua.validate_sparse_color_selection(sym, sel, 0)
        with self.assertRaises(ValueError): fua.validate_sparse_color_selection(sym, ~sel, 1)
        sym['sh_dc'][3, 0, 0] = 1
        with self.assertRaises(ValueError): fua.validate_sparse_color_selection(sym, sel, 1)
        sym['sh_dc'][3, 0, 0] = 0; sym['logscale'][2, 0] = 1
        with self.assertRaises(ValueError): fua.validate_sparse_color_selection(sym, sel, 1)

    def test_codec_rejects_dense_budget_fields_and_corruption(self):
        ref = reference(8); sym = {k: torch.zeros_like(v) for k,v in ref.items()}
        sym['sh_dc'][2,0,0] = 1; sym['sh_dc'][3,0,0] = 2
        dense, sparse, planar = legacy_packets(ref, sym)
        for packet in (dense, sparse, planar):
            with self.assertRaises(ValueError): fua.decode_full_attribute_delta(packet, ref, frame_index=10, color_only_budget=1)
        with self.assertRaises(ValueError): fua.decode_full_attribute_delta(planar, ref, frame_index=11, color_only_budget=2)
        broken = bytearray(planar); broken[-1] ^= 1
        with self.assertRaises(ValueError): fua.decode_full_attribute_delta(broken, ref, frame_index=10, color_only_budget=2)
        changed = {k:v.clone() for k,v in ref.items()}; changed['sh_dc'][0,0,0] = 1
        with self.assertRaises(ValueError): fua.decode_full_attribute_delta(planar, changed, frame_index=10, color_only_budget=2)
        sym['logscale'][0,0] = 1
        with self.assertRaises(ValueError): fua.decode_full_attribute_delta(legacy_packets(ref, sym)[2], ref, frame_index=10, color_only_budget=2)
        zero = {k:torch.zeros_like(v) for k,v in ref.items()}
        self.assertIsNone(fua.encode_full_attribute_delta(ref, zero, frame_index=10, steps=(.005,.001,.001), sparse_only=True))
        with self.assertRaises(ValueError): fua.decode_full_attribute_delta(legacy_packets(ref, zero)[2], ref, frame_index=10, color_only_budget=2)
        # Valid envelope/checksum but duplicate row (zero gap), out-of-range
        # row, noncanonical varint and zero symbol row must all be rejected.
        empty = struct.pack('<I', 0)
        for gaps, count, values in ((b'\x01\x00', 2, b'\x01\0\0\0\0\0'*2),
                                    (b'\x09', 1, b'\x01\0\0\0\0\0'),
                                    (b'\x81\x00', 1, b'\x01\0\0\0\0\0'),
                                    (b'\x01', 1, b'\0'*6)):
            body=empty+struct.pack('<I',count)+gaps+values+empty
            packet=fua._encode_body(fua.SPARSE_MAGIC,body,10,8,3,(.005,.001,.001),fua._reference_hash(ref))
            with self.assertRaises(ValueError): fua.decode_full_attribute_delta(packet,ref,frame_index=10,color_only_budget=2)

    def test_manifest_empty_missing_extra_and_versions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); state, raw, manifest, summary, _ = make_stream(p, magic=fua.SPARSE_MAGIC)
            ref={k:raw[k] for k in fua.FIELDS}; sym={k:torch.zeros_like(v) for k,v in ref.items()}
            sym['sh_dc'][2,0,0]=1
            packet=fua.encode_full_attribute_delta(ref,sym,frame_index=1,steps=(.005,.001,.001),sparse_only=True)
            (p/'fullattr_001.bin').write_bytes(packet)
            manifest.update(color_stream_version=2,color_point_budget=1)
            def save(): (p/'frame_001.json').write_text(json.dumps(manifest))
            save(); _, result=decode_frame(p,1,state,raw); self.assertNotEqual(float(result['sh_dc'][2,0,0]),0)
            manifest['birth']=None
            summary['full_attribute_update']['archive_bytes']=len(packet)
            summary['packet_bytes']=(p/'motion_001.bin').stat().st_size+len(packet)
            summary['manifest_bytes']=(p/'frame_001.json').stat().st_size
            summary['encoded_bytes']=summary['complete_dynamic_update_bytes']=summary['packet_bytes']+summary['manifest_bytes']
            actual=_actual_frame_storage(p,1,manifest,summary,11,'reference_frame0.npz')
            self.assertEqual(actual['complete_update_bytes'],summary['encoded_bytes'])
            manifest['color_point_mask_count']=0; save()
            with self.assertRaises(ValueError): decode_frame(p,1,state,raw)
            del manifest['color_point_mask_count']; manifest['color_stream_version']=3; save()
            with self.assertRaises(ValueError): decode_frame(p,1,state,raw)
            manifest['color_stream_version']=2; manifest['full_attribute_update']=None; save()
            with self.assertRaises(ValueError): decode_frame(p,1,state,raw)
            (p/'fullattr_001.bin').unlink(); decode_frame(p,1,state,raw)
            manifest['full_attribute_update']='fullattr_001.bin'; save()
            with self.assertRaises(FileNotFoundError): decode_frame(p,1,state,raw)
            manifest['full_attribute_update']=None; (p/'pointmask_001.bin').write_bytes(b'extra'); save()
            with self.assertRaises(ValueError): decode_frame(p,1,state,raw)

if __name__ == '__main__': unittest.main()

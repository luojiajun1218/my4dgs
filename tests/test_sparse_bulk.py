import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline.model import sparse_integer_codec as layout
from pipeline.model import rigid_tree_codec as codec
import test_spatial_motion as fixtures


class BulkSparseDecodeTests(unittest.TestCase):
    def test_bulk_varints_match_scalar_canonical_reader_and_leave_trailers(self):
        rng=np.random.default_rng(728)
        values=np.r_[np.array([0,1,127,128,16383,16384,65535,65536,2**28,2**32-1],dtype=np.uint64),
                     rng.integers(0,2**32,size=10001,dtype=np.uint64)]
        encoded=layout.encode_unsigned_values(values)
        for count in (0,1,9,10,len(values)):
            with self.subTest(count=count):
                scalar=layout._Reader(encoded+b'trailer');bulk=layout._Reader(encoded+b'trailer')
                expected=[scalar.integer() for _ in range(count)]
                actual=layout._read_unsigned_values(bulk,count)
                self.assertEqual(actual.tolist(),expected)
                self.assertEqual(bulk.cursor,scalar.cursor)
                self.assertEqual(bytes(bulk.take(bulk.remaining)),bytes(scalar.take(scalar.remaining)))

    def test_bulk_rejects_the_scalar_readers_malformed_and_noncanonical_words(self):
        examples=[(b'\x80\x00',1),(b'\x81\x00',1),(b'\x80'*5+b'\x01',1),
                  (b'\xff'*4+b'\x10',1),(b'\x80',1),(b'\x00',2),(b'\x00\x81\x00',2)]
        for data,count in examples:
            with self.subTest(data=data,count=count):
                scalar=layout._Reader(data)
                with self.assertRaises(ValueError):
                    for _ in range(count):scalar.integer()
                with self.assertRaises(ValueError):
                    layout._read_unsigned_values(layout._Reader(data),count)

    def test_large_mixed_sparse_blocks_reconstruct_exact_bytes_and_integer_arrays(self):
        rng=np.random.default_rng(7421)
        for count in (0,1,10003):
            t=rng.integers(-32768,32768,size=(count,3),dtype=np.int16)
            r=rng.integers(-32768,32768,size=(count,3),dtype=np.int16)
            t[rng.random(count)<.7]=0;r[rng.random(count)<.85]=0
            raw=bytearray();codec._sparse_write(raw,t,r)
            reader=layout._Reader(raw)
            planar=layout._encode_sparse(reader,count)
            reconstructed=layout._decode_sparse(layout._Reader(planar),count)
            self.assertEqual(bytes(raw),reconstructed)
            decoded_t,decoded_r=codec._sparse_read(codec._Reader(reconstructed),count,delta_rows=True)
            torch.testing.assert_close(decoded_t,torch.from_numpy(t),rtol=0,atol=0)
            torch.testing.assert_close(decoded_r,torch.from_numpy(r),rtol=0,atol=0)

    def test_gap_validation_preserves_survivor_and_deletion_boundaries(self):
        for values,limit,deletion,valid in [([1,1,1],3,False,True),([0,0,0],3,True,True),
                ([1,2,1],3,False,False),([0],3,False,False),([0,0,1],3,True,False)]:
            data=layout.encode_unsigned_values(values);reader=layout._Reader(data)
            if valid:self.assertEqual(reader.gaps(len(values),limit,deletion=deletion),data)
            else:
                with self.assertRaises(ValueError):reader.gaps(len(values),limit,deletion=deletion)

    def test_spatial_decode_does_not_compute_unused_rigid_pivots(self):
        model=fixtures.JointImageRateTests().model()
        payload=fixtures.JointImageRateTests.packet(model)
        with patch.object(codec,'region_pivots',side_effect=AssertionError('unused spatial pivots')):
            decoded=codec.decode_motion(payload,model.frame_reference_state)
        xyz,q=model.materialized_pose()
        torch.testing.assert_close(decoded.xyz,xyz,rtol=0,atol=0)
        torch.testing.assert_close(decoded.rotation,q,rtol=0,atol=0)

    def test_retained_planar_arrays_match_raw_parser_and_preserve_section_cursor(self):
        rng=np.random.default_rng(8102)
        for count in (0,1,12003):
            t=rng.integers(-32768,32768,size=(count,3),dtype=np.int16)
            r=rng.integers(-32768,32768,size=(count,3),dtype=np.int16)
            t[rng.random(count)<.6]=0;r[rng.random(count)<.9]=0
            raw=bytearray();codec._sparse_write(raw,t,r)
            planar=layout._encode_sparse(layout._Reader(raw),count)
            retained=[]
            self.assertEqual(bytes(raw),layout._decode_sparse(layout._Reader(planar),count,decoded=retained))
            sections={0:(len(raw),count,*retained[0])}
            reader=codec._Reader(raw+b'trailer')
            actual=codec._sparse_read(reader,count,delta_rows=True,sparse_sections=sections)
            self.assertFalse(sections)
            self.assertEqual(bytes(reader.take(7)),b'trailer')
            for value,expected in zip(actual,(t,r)):
                torch.testing.assert_close(value,torch.from_numpy(expected),rtol=0,atol=0)

    def test_retained_sections_cannot_bypass_zero_block_validation(self):
        for flag in (1,2,3):
            raw=b'\x01\x01'+bytes([flag])+b'\0'*(6*(bool(flag&1)+bool(flag&2)))
            planar=layout._encode_sparse(layout._Reader(raw),1)
            with self.assertRaisesRegex(ValueError,'Redundant zero'):
                codec._sparse_read(codec._Reader(raw),1,delta_rows=True)
            with self.assertRaisesRegex(ValueError,'Redundant zero'):
                layout._decode_sparse(layout._Reader(planar),1,decoded=[])

    def test_retaining_an_empty_section_does_not_allocate_untrusted_population_size(self):
        retained=[]
        with patch.object(layout.np,'zeros',side_effect=AssertionError('dense allocation before state check')):
            self.assertEqual(layout._decode_sparse(layout._Reader(b'\0'),2**32-1,decoded=retained),b'\0')
        self.assertTrue(all(value.size==0 for value in retained[0]))

    def test_whole_packet_uses_retained_sections_and_checks_original_checksum(self):
        model=fixtures.JointImageRateTests().model()
        raw=codec.uncompressed_motion_packet(fixtures.JointImageRateTests.packet(model))
        planar=layout.pack_layout(layout.encode_motion_layout(raw),len(raw),layout.MOTION_MAGICS)
        expected=codec.decode_motion(raw,model.frame_reference_state)
        original=codec._sparse_read
        def checked(reader,count,**kwargs):
            self.assertIn(reader.cursor,kwargs['sparse_sections'])
            return original(reader,count,**kwargs)
        with patch.object(codec,'_sparse_read',side_effect=checked) as read:
            actual=codec.decode_motion(planar,model.frame_reference_state)
        self.assertEqual(read.call_count,2)
        self.assertEqual(actual.digest(),expected.digest())
        damaged=raw[:-1]+bytes([raw[-1]^1])
        packet=layout.pack_layout(layout.encode_motion_layout(damaged),len(damaged),layout.MOTION_MAGICS)
        with patch.object(codec,'_compose_decoded_pose',side_effect=AssertionError('must validate first')), \
                self.assertRaisesRegex(ValueError,'checksum'):
            codec.decode_motion(packet,model.frame_reference_state)


if __name__=='__main__':unittest.main()

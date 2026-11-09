import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from pipeline.model import full_attribute_update as fua
from pipeline.module import Module
from scripts.replay_rigid_tree import decode_frame
from tests.test_full_attribute_update import make_stream, legacy_packets, reference


class SparseAttrV3Tests(unittest.TestCase):
    def setUp(self):
        self.ref=reference(8)
        self.sym={k:torch.zeros_like(v) for k,v in self.ref.items()}

    def packet(self):
        return fua.encode_full_attribute_delta(self.ref,self.sym,frame_index=10,steps=(.005,.001,.001),sparse_only=True)

    def decode(self, packet, budget, fields=('sh_dc','logscale')):
        return fua.decode_full_attribute_delta(packet,self.ref,frame_index=10,color_only_budget=budget,allowed_fields=fields)

    def test_union_overlap_disjoint_and_empty_fields(self):
        self.sym['sh_dc'][1,0,0]=1; self.sym['logscale'][1,0]=2
        for p in legacy_packets(self.ref,self.sym)[1:]: self.decode(p,1)
        self.sym['logscale'].zero_(); self.sym['logscale'][2,0]=3
        for p in legacy_packets(self.ref,self.sym)[1:]:
            with self.assertRaisesRegex(ValueError,'union'): self.decode(p,1)
            self.decode(p,2)
        self.sym['sh_dc'].zero_(); self.decode(self.packet(),2)
        self.sym['logscale'].zero_(); self.sym['sh_dc'][4,0,0]=1; self.decode(self.packet(),1)
        self.sym['sh_dc'].zero_(); self.assertIsNone(self.packet())
        with self.assertRaises(ValueError): self.decode(legacy_packets(self.ref,self.sym)[1],1)

    def test_authorization_selection_and_bad_packets(self):
        selected=torch.zeros(8,dtype=torch.bool); selected[1:3]=True
        self.sym['sh_dc'][1,0,0]=1; self.sym['logscale'][2,0]=2
        fua.validate_sparse_color_selection(self.sym,selected,2,('sh_dc','logscale'))
        with self.assertRaises(ValueError): fua.validate_sparse_color_selection(self.sym,selected,1,('sh_dc','logscale'))
        with self.assertRaises(ValueError): fua.validate_sparse_color_selection(self.sym,selected,2)
        with self.assertRaises(ValueError): self.decode(legacy_packets(self.ref,self.sym)[0],2)
        with self.assertRaises(ValueError): self.decode(self.packet(),2,('sh_dc',))
        self.sym['sh_rest'][1,0,0]=1
        with self.assertRaises(ValueError): fua.validate_sparse_color_selection(self.sym,selected,2,('sh_dc','logscale'))
        with self.assertRaises(ValueError): self.decode(self.packet(),2)
        self.sym['sh_rest'].zero_(); self.sym['logscale'][4,0]=1
        with self.assertRaises(ValueError): fua.validate_sparse_color_selection(self.sym,selected,2,('sh_dc','logscale'))
        p=self.packet(); broken=bytearray(p); broken[-1]^=1
        with self.assertRaises(ValueError): self.decode(broken,3)
        with self.assertRaises(ValueError): fua.decode_full_attribute_delta(p,self.ref,frame_index=9,color_only_budget=3,allowed_fields=('sh_dc','logscale'))
        wrong={k:v.clone() for k,v in self.ref.items()}; wrong['sh_dc'][0,0,0]=1
        with self.assertRaises(ValueError): fua.decode_full_attribute_delta(p,wrong,frame_index=10,color_only_budget=3,allowed_fields=('sh_dc','logscale'))

    def test_manifest_v3_and_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); state,raw,m,_,_=make_stream(p,magic=fua.SPARSE_MAGIC)
            ref={k:raw[k] for k in fua.FIELDS}; sym={k:torch.zeros_like(v) for k,v in ref.items()}
            sym['sh_dc'][2,0,0]=1; sym['logscale'][3,0]=2
            packet=fua.encode_full_attribute_delta(ref,sym,frame_index=1,steps=(.005,.001,.001),sparse_only=True)
            (p/'fullattr_001.bin').write_bytes(packet)
            m.update(color_stream_version=3,color_point_budget=2,color_attribute_fields=['sh_dc','logscale'])
            def save(): (p/'frame_001.json').write_text(json.dumps(m))
            save(); _,result=decode_frame(p,1,state,raw)
            self.assertNotEqual(float(result['logscale'][3,0]),0.)
            for bad in (None, ['sh_dc'], ['logscale','sh_dc'], ['sh_dc','sh_rest'], ['sh_dc','logscale','logscale']):
                m['color_attribute_fields']=bad;save()
                with self.assertRaises(ValueError):decode_frame(p,1,state,raw)
            m['color_attribute_fields']=['sh_dc','logscale'];m['color_point_budget']=1;save()
            with self.assertRaises(ValueError):decode_frame(p,1,state,raw)
            m['color_point_budget']=2;m['logscale_update']='logscale_001.bin';save()
            with self.assertRaises(ValueError):decode_frame(p,1,state,raw)
            del m['logscale_update'];m['full_attribute_update']=None;save()
            with self.assertRaises(ValueError):decode_frame(p,1,state,raw)
            (p/'fullattr_001.bin').unlink();decode_frame(p,1,state,raw)
            m['full_attribute_update']='fullattr_001.bin';save()
            with self.assertRaises(FileNotFoundError):decode_frame(p,1,state,raw)


if __name__=='__main__':unittest.main()

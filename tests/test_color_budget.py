"""Point-budget selection, compressed masks, and decoder boundary checks."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from pipeline.model.color_point_mask import encode_color_point_mask,decode_color_point_mask
from pipeline.model.full_attribute_update import SPARSE_MAGIC
from scripts.replay_rigid_tree import decode_frame
from tests.test_full_attribute_update import make_stream

class ColorBudgetTests(unittest.TestCase):
 def test_mask_codec(self):
  mask=np.zeros(5000,dtype=np.bool_);mask[[1,20,73,1200,4999]]=True
  packet=encode_color_point_mask(mask,7);self.assertIsNotNone(packet)
  self.assertTrue(np.array_equal(mask,decode_color_point_mask(packet,7,len(mask))))
  self.assertIsNone(encode_color_point_mask(np.zeros(3,dtype=np.bool_),7))
  for data,frame,n in ((packet[:-1],7,len(mask)),(packet,8,len(mask)),(packet,7,len(mask)-1)):
   with self.assertRaises(ValueError):decode_color_point_mask(data,frame,n)
 def test_replay_rejects_unselected_point_residual(self):
  with tempfile.TemporaryDirectory() as d:
   stream=Path(d);prior,raw,manifest,_,_=make_stream(stream,magic=SPARSE_MAGIC)
   from pipeline.model.full_attribute_update import encode_full_attribute_delta
   ref={k:raw[k] for k in ('logscale','sh_dc','sh_rest')}
   symbols={k:torch.zeros_like(v) for k,v in ref.items()};symbols['sh_dc'][-1,0,0]=7
   (stream/'fullattr_001.bin').write_bytes(encode_full_attribute_delta(ref,symbols,frame_index=1,steps=(.005,.001,.001)))
   mask=np.zeros(512,dtype=np.bool_);mask[-1]=True
   packet=encode_color_point_mask(mask,1);(stream/'pointmask_001.bin').write_bytes(packet)
   manifest.update(color_point_mask='pointmask_001.bin',color_point_mask_count=1,color_point_budget=1)
   (stream/'frame_001.json').write_text(json.dumps(manifest))
   _,decoded=decode_frame(stream,1,prior,raw);self.assertNotEqual(float(decoded['sh_dc'][-1,0,0]),0.)
   mask[-1]=False;mask[0]=True;(stream/'pointmask_001.bin').write_bytes(encode_color_point_mask(mask,1))
   with self.assertRaisesRegex(ValueError,'outside'):decode_frame(stream,1,prior,raw)
 def test_replay_rejects_non_dc_attributes(self):
  with tempfile.TemporaryDirectory() as d:
   stream=Path(d);prior,raw,manifest,_,_=make_stream(stream,magic=SPARSE_MAGIC)
   mask=np.zeros(512,dtype=np.bool_);mask[-1]=True
   (stream/'pointmask_001.bin').write_bytes(encode_color_point_mask(mask,1))
   manifest.update(color_point_mask='pointmask_001.bin',color_point_mask_count=1,color_point_budget=1)
   (stream/'frame_001.json').write_text(json.dumps(manifest))
   with self.assertRaisesRegex(ValueError,'non-DC'):decode_frame(stream,1,prior,raw)
if __name__=='__main__':unittest.main()

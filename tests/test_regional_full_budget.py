import unittest
from types import SimpleNamespace
import numpy as np
import torch
from pipeline.model.full_attribute_update import FullUpdateConfig, encode_full_attribute_delta, decode_full_attribute_delta
from pipeline.frame_plan import FramePlanController
from pipeline.module import Module
from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate

class BudgetedFullAttributeTests(unittest.TestCase):
 def test_six_fields_and_optimizer_moments_stay_frozen_outside_selection(self):
  n=4;fields={name:torch.nn.Parameter(torch.ones(shape)) for name,shape in {'xyz':(n,3),'rotation':(n,3),'opacity':(n,1),'logscale':(n,3),'sh_dc':(n,1,3),'sh_rest':(n,3,3)}.items()};opt=torch.optim.Adam(list(fields.values()),lr=.1)
  sum(x.sum() for x in fields.values()).backward();opt.step();opt.zero_grad(set_to_none=True)
  m=SimpleNamespace(device=torch.device('cpu'),gaussians=SimpleNamespace(_xyz=torch.empty(n,3)),motion=SimpleNamespace(previous_rows=torch.arange(n),optimizer=opt),opacity_update=SimpleNamespace(rows=torch.arange(n)),birth_control={'regional_attribute_repair':{'invariant_checks':[]}})
  m._regional_attribute_parameters=lambda:fields
  m._plan_validate_regional_attribute_state=lambda s:FramePlanController._plan_validate_regional_attribute_state(m,s)
  mask=torch.tensor([True,False,True,False]);FramePlanController._plan_snapshot_regional_attributes(m,mask); before={k:v.detach().clone() for k,v in fields.items()};mom={k:opt.state[v]['exp_avg'].clone() for k,v in fields.items()}
  sum(x.sum() for x in fields.values()).backward();FramePlanController._plan_mask_regional_attribute_gradients(m);opt.step();FramePlanController._plan_restore_regional_attribute_frozen_rows(m,201)
  for k,v in fields.items():
   self.assertTrue(torch.equal(v.detach()[~mask],before[k][~mask]),k);self.assertTrue(torch.equal(opt.state[v]['exp_avg'][~mask],mom[k][~mask]),k);self.assertFalse(torch.equal(v.detach()[mask],before[k][mask]),k)
 def test_fua_roundtrip_preserves_dc_scale_and_higher_sh(self):
  ref={'logscale':torch.zeros(4,3),'sh_dc':torch.zeros(4,1,3),'sh_rest':torch.zeros(4,15,3)};sym={k:torch.zeros_like(v) for k,v in ref.items()}
  for v in sym.values():v[2].reshape(-1)[0]=2
  packet=encode_full_attribute_delta(ref,sym,frame_index=1,steps=(.005,.001,.001));out=decode_full_attribute_delta(packet,ref,frame_index=1)
  for k in ref:self.assertTrue(torch.equal(out[k][[0,1,3]],ref[k][[0,1,3]]));self.assertNotEqual(out[k][2].abs().sum().item(),0)
 def test_full_mode_alpha_forward_equals_decoder_and_keeps_gradient(self):
  update=QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True,mode='alpha',step=.01));update.setup(torch.tensor([[-1.2],[0.1],[2.0]]))
  with torch.no_grad():update.delta[:,0]=torch.tensor([.02,-.03,0.])
  cfg=FullUpdateConfig(enabled=True)
  m=SimpleNamespace(config=SimpleNamespace(full_update=cfg),full_update_active=True,opacity_update=update,trainer=SimpleNamespace(global_step=201),motion_steps=200)
  alpha=Module._render_opacity_alpha(m);decoded=update.canonical_logits().sigmoid();self.assertTrue(torch.equal(alpha,decoded));alpha.sum().backward();self.assertTrue(torch.equal(update.delta.grad,torch.ones_like(update.delta)))
if __name__=='__main__':unittest.main()

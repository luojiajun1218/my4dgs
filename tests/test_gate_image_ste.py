"""Hard tree frontier with an optional gate-only soft-path image derivative."""
import copy,tempfile,unittest
from pathlib import Path
import torch
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.spatial_gate import spatial_owner
from pipeline.model.rigid_tree_codec import decode_motion
from tests.test_frame_plan import motion_fixture

class GateImageSTETests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def model(self):
  initial=motion_fixture();cfg=copy.deepcopy(initial.config);cfg.spatial_gate.max_depth=2
  m=SpatialGatedMotion(cfg);m.setup(initial.base_xyz,initial.base_rotation,initial.extent,frame_index=1)
  m.observations.fill_(1)
  with torch.no_grad():
   m.region_translation[:,0]=(torch.arange(len(m.parents))%7)*.006
   m.gs_translation[:,1]=.003
   m.gate_logits.fill_(-.2)
  return m
 def test_hard_forward_gate_gradient_and_ineligible_nodes(self):
  m=self.model();self.assertEqual(m.spatial_paths.shape[1],2)
  hard_weights=m._path_weights(m.opened.float());self.assertTrue(torch.equal(hard_weights.sum(1),torch.ones(len(m.owner))))
  self.assertTrue(torch.equal((hard_weights>0).sum(1),torch.ones_like(m.owner)))
  self.assertTrue(torch.equal(spatial_owner(m.spatial_paths,m.opened),m.owner))
  m.refine_image_gate_ste=False
  x0,r0=m(151,training=True);loss0=x0.square().sum();g0=torch.autograd.grad(loss0,m.gate_logits,allow_unused=True,retain_graph=True)
  self.assertIsNone(g0[0])
  p0=torch.autograd.grad(loss0,(m.region_translation,m.gs_translation))
  m.refine_image_gate_ste=True
  x1,r1=m(151,training=True);self.assertTrue(torch.equal(x0,x1));self.assertTrue(torch.equal(r0,r1))
  g=torch.autograd.grad(x1.square().sum(),m.gate_logits,retain_graph=True)[0]
  self.assertTrue(bool(torch.isfinite(g).all()));self.assertGreater(float(g[m.split_eligible].abs().sum()),0.)
  self.assertTrue(torch.equal(g[~m.split_eligible],torch.zeros_like(g[~m.split_eligible])))
  p1=torch.autograd.grad(x1.square().sum(),(m.region_translation,m.gs_translation))
  for a,b in zip(p0,p1):self.assertTrue(torch.equal(a,b))
  for step in (150,181):
   m.refine_image_gate_ste=False;x0,r0=m(step,training=True)
   m.refine_image_gate_ste=True;x1,r1=m(step,training=True)
   self.assertTrue(torch.equal(x0,x1));self.assertTrue(torch.equal(r0,r1))
 def test_motion_image_gate_proxy_follows_early_joint_schedule(self):
  m=self.model();m.config.spatial_gate.joint_gate_start_step=131
  m.refine_image_gate_ste=False;x0,r0=m(131,training=True)
  self.assertIsNone(torch.autograd.grad(x0.square().sum(),m.gate_logits,allow_unused=True)[0])
  m.refine_image_gate_ste=True;x1,r1=m(131,training=True)
  self.assertTrue(torch.equal(x0,x1));self.assertTrue(torch.equal(r0,r1))
  g=torch.autograd.grad(x1.square().sum(),m.gate_logits)[0]
  self.assertGreater(float(g[m.split_eligible].abs().sum()),0.)
 def test_quantized_motion_survives_cut_switch_and_real_packet(self):
  m=self.model();m.refine_image_gate_ste=True;m(151,training=True)
  before=tuple(x.clone() for x in m._full_motion_symbols())
  with torch.no_grad():m.gate_logits[m.split_eligible]=2.
  m._select_frontier(freeze=False)
  after=m._full_motion_symbols()
  for a,b in zip(before,after):self.assertTrue(torch.equal(a,b))
  xyz,rot=m.materialized_pose();prior=copy.deepcopy(m.frame_reference_state)
  m.register_materialized_layout(xyz,rot,torch.empty(0,dtype=torch.long))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'motion.bin';m.export_motion(p);decoded=decode_motion(p.read_bytes(),prior)
  torch.testing.assert_close(decoded.xyz,xyz,rtol=0,atol=1e-6)
  torch.testing.assert_close(decoded.rotation,rot,rtol=0,atol=1e-6)
  self.assertTrue(torch.equal(decoded.owner,m.owner.cpu()))
if __name__=='__main__':unittest.main()

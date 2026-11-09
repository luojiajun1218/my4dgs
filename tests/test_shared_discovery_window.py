import copy
import tempfile
import unittest
from pathlib import Path
import torch
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion

class DiscoveryWindowTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def model(self,n):
  axis=torch.linspace(-1.,1.,8);xyz=torch.cartesian_prod(axis,axis,axis);q=torch.zeros(len(xyz),4);q[:,0]=1
  c=RigidTreeConfig(motion_xyz_step_extent=.001,gs_xyz_step_extent=.001,motion_rotation_step_rad=.001,gs_rotation_step_rad=.001,rate_loss_weight=.002,relative_motion_loss_weight=.01,rate_warmup_steps=0,rate_ramp_steps=1,motion_fit_window_steps=20);c.motion_steps=200;c.birth_steps=100;c.rotation_updates=False;c.amplitude_refit_steps=0;c.proximal_gs_regularization=True;c.differentiable_rate_support=True
  g=c.spatial_gate;g.max_depth=3;g.min_region_size=2;g.initial_logit=-2.;g.training_schedule='joint';g.learn_region_motion=True;g.start_step=g.gate_start_step=0;g.freeze_step=180;g.joint_image_gradients=g.shared_image_recode=g.shared_discovery=True;c.shared_discovery_steps=n;g.region_motion='relative';g.image_relaxation='stochastic_ste';g.frame_training_plan=True;c.validate();m=SpatialGatedMotion(c);m.setup(xyz,q,2.,frame_index=1);m.observations.fill_(1);return m
 def test_boundaries(self):
  for n in (50,75,100):
   with self.subTest(n=n):
    m=self.model(n);m.prepare_plan_step(n);self.assertTrue(m.shared_discovery_active);self.assertFalse(m.gs_translation.requires_grad);m._finish_joint_step();m.prepare_plan_step(n+1);self.assertFalse(m.shared_discovery_active);self.assertTrue(m.gs_translation.requires_grad)
    for step,enabled in ((100,True),(101,False),(150,False),(151,True),(180,True),(181,False)):
     m.prepare_plan_step(step);self.assertEqual(m.gate_logits.requires_grad,enabled)
 def test_gradient_and_adam_after_step50(self):
  m=self.model(50);m.prepare_plan_step(50);m._finish_joint_step();before=m.gs_translation.detach().clone();xyz,_=m(51,training=True);(xyz-(m.base_xyz+.013)).square().mean().backward();self.assertGreater(float(m.gs_translation.grad.abs().sum()),0.);m.optimizer_step();self.assertGreater(float((m.gs_translation-before).abs().sum()),0.)
 def test_recode_and_real_packet(self):
  m=self.model(50);m.prepare_plan_step(50);m._finish_joint_step();m.prepare_plan_step(51)
  with torch.no_grad():
   m.region_translation[:,0]=torch.arange(len(m.parents))%7*.002;m.gs_translation[:,1]=.0073;m.gate_logits[1:]=3.
  before=tuple(v.clone() for v in m._full_motion_symbols());old=m.owner.clone();m._select_frontier(freeze=False);self.assertTrue(bool(m.owner.ne(old).any()))
  for a,b in zip(before,m._full_motion_symbols()):self.assertTrue(torch.equal(a,b))
  ref=copy.deepcopy(m.frame_reference_state);xyz,rotation=m.materialized_pose();m.register_materialized_layout(xyz,rotation,torch.empty(0,dtype=torch.long))
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'motion.bin';info=m.export_motion(path);decoded=decode_motion(path.read_bytes(),ref);self.assertEqual(info['bytes'],path.stat().st_size);torch.testing.assert_close(decoded.xyz,xyz,rtol=0,atol=1e-6);torch.testing.assert_close(decoded.rotation,rotation,rtol=0,atol=1e-6)
 def test_invalid_window(self):
  for n in (0,101,True):
   with self.assertRaises(ValueError):self.model(n)

if __name__=='__main__':unittest.main()

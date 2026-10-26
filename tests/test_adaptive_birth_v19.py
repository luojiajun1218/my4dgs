"""Joint old/new opacity, equal population pressure and exact final replay."""
import tempfile
import unittest
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import torch

from tests.test_adaptive_birth import controller_fixture
from tests.test_reincarnation import M, T
from pipeline.model.sparse_updates import QuantizedOpacityUpdate
from pipeline.model.birth_quantization import model_raw


def joint_fixture(directory, empty=False):
    m,views=controller_fixture(directory,empty)
    m.config.birth.joint_old_opacity=True
    m.config.birth.population_loss_weight=.002
    m.trainer.global_step=200
    m.early_shared_steps=200
    m.early_birth_updates=0 if empty else 130
    m._early_select_population()
    return m,views


class JointPopulationTests(unittest.TestCase):
    def test_observed_old_and_new_have_equal_support_gradient_and_unseen_none(self):
        with tempfile.TemporaryDirectory() as directory:
            m,_=joint_fixture(directory)
            with torch.no_grad():
                m.opacity_update.alpha_reference.fill_(.1)
                m.opacity_update.delta.zero_()
                m.births._opacity.fill_(.1)
                m.motion.observations.fill_(1)
                m.birth_transaction.observations.fill_(1)
                m.motion.observations[0]=0
                m.birth_transaction.observations[0]=0
            self.assertEqual(m._adaptive_population_regularization(90)[0].item(),0)
            loss,_=m._adaptive_population_regularization(160)
            loss.backward()
            old_grad=m.opacity_update.delta.grad.flatten()
            new_grad=m.births._opacity.grad.flatten()
            self.assertEqual(old_grad[0].item(),0)
            self.assertEqual(new_grad[0].item(),0)
            self.assertGreater(old_grad[1].item(),0)
            torch.testing.assert_close(old_grad[1],new_grad[1])

    def test_post200_single_optimizer_old_and_new_fit_with_motion_frozen(self):
        for empty in (False,True):
            with self.subTest(empty=empty),tempfile.TemporaryDirectory() as directory:
                m,views=joint_fixture(directory,empty)
                self.assertTrue(m.opacity_update.delta.requires_grad)
                packet=m.opacity_update.capture()
                restored=QuantizedOpacityUpdate(m.opacity_update.config)
                restored.restore(packet,'cpu')
                self.assertTrue(restored.delta.requires_grad)
                torch.testing.assert_close(restored.alphas(),m.opacity_update.alphas())
                old=m.opacity_update.delta.detach().clone()
                xyz=m.gaussians._xyz.detach().clone()
                motion=[v.detach().clone() for v in m.motion.parameters()]
                m.training_step=MethodType(M.Module.training_step,m)
                trainer=NS(module=m,config=NS(motion_views_per_step=2))
                trainer._views_per_step=MethodType(T.Trainer._views_per_step,trainer)
                m.trainer.global_step=201;m._early_prepare_rates(201)
                with patch.object(m.motion.optimizer,'step',wraps=m.motion.optimizer.step) as shared,patch.object(m.opacity_update.optimizer,'step',side_effect=AssertionError('second Adam')):
                    T.Trainer._fit_views(trainer,NS(next=lambda:None),201,defer_metrics=True)
                self.assertEqual(shared.call_count,1)
                self.assertFalse(torch.equal(old,m.opacity_update.delta))
                torch.testing.assert_close(xyz,m.gaussians._xyz,rtol=0,atol=0)
                for before,after in zip(motion,m.motion.parameters()):
                    torch.testing.assert_close(before,after,rtol=0,atol=0)
                self.assertEqual(m.early_shared_steps,201)

    def test_final_opacity_and_child_predictor_replay_exact_after_retirement(self):
        from pipeline.model import rigid_tree_codec as codec
        from scripts.replay_rigid_tree import decode_frame
        from pipeline.model.residual_birth import RawView
        with tempfile.TemporaryDirectory() as directory:
            m,views=joint_fixture(directory)
            prior=m.motion.frame_reference_state
            reference=model_raw(m.gaussians)
            reference['xyz']=prior.xyz;reference['rotation']=prior.rotation
            reference['opacity']=m.opacity_update.reference.clone()
            parents=m.birth_transaction.sources
            nonparents=[i for i in range(len(m.gaussians._xyz)) if i not in parents.tolist()]
            with torch.no_grad():
                m.opacity_update.delta[parents[0]]+=.07
                m.opacity_update.delta[nonparents[0]]=-m.opacity_update.alpha_reference[nonparents[0]]
                m.motion.observations.fill_(1)
                m.birth_transaction.observations.fill_(1)
                m.births._opacity.fill_(.2)
            # The child's live predictor must follow the old alpha at300.
            camera=next(iter(views.values()))
            live=m.forward(camera,training=True)['render'].detach()
            raw=model_raw(m.gaussians)
            raw['opacity']=m.opacity_update.canonical_logits()
            newborn=m.make_birth_view(reference=raw)
            from pipeline.model.birth_quantization import materialize_quantized_gaussians
            raw,_=materialize_quantized_gaussians(raw,newborn)
            previous=m.renderer.gaussians
            m.renderer.gaussians=RawView(raw,m.config.sh_degree,m.gaussians)
            exact=m.renderer(camera,m.background)['render'].detach()
            m.renderer.gaussians=previous
            torch.testing.assert_close(live,exact,rtol=1e-5,atol=1e-6)
            m.trainer.global_step=300;m.early_shared_steps=300;m.early_birth_updates=230
            m.initial_reference={};m.stage_image_metrics=[]
            m.trainer.decoder_reference_path=m.output_dir/'reference.npz'
            m.trainer.decoder_reference_path.write_bytes(b'fixture')
            for name in ('finalize_frame','_freeze_base'):
                setattr(m,name,MethodType(getattr(M.Module,name),m))
            with patch.object(m.gaussians,'training_setup',return_value=None),patch.object(codec,'encode_motion',wraps=codec.encode_motion) as encode:
                m._early_finish()
            self.assertEqual(encode.call_count,1)
            self.assertGreaterEqual(m.frame_report['retired_count'],1)
            _,decoded=decode_frame(m.output_dir/'frame_storage/rigid_tree',1,prior,reference)
            for name,value in model_raw(m.gaussians).items():
                torch.testing.assert_close(value,decoded[name],rtol=0,atol=0)

    def test_config_rejects_population_pressure_without_joint_fitting(self):
        cfg=M.BirthConfig(radiance_mode='adaptive_residual_birth',population_loss_weight=.002)
        with self.assertRaisesRegex(ValueError,'joint_old_opacity'):cfg.validate(200)
        cfg.joint_old_opacity=True
        cfg.validate(200)


if __name__=='__main__':unittest.main()

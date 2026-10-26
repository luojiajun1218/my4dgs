import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.model.motion_evidence import MotionEvidence, PROBE_ALPHAS, probe_motion_evidence
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion
import test_motion_training as training_fixtures
import test_spatial_motion as codec_fixtures


class MotionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        template = training_fixtures.StochasticSharedResidualTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.motion_steps = 200
        cfg.amplitude_refit_steps = 100
        cfg.counterfactual_residual_support = True
        cfg.spatial_gate.shared_discovery = True
        cfg.spatial_gate.shared_image_recode = True
        cfg.validate()
        result = SpatialGatedMotion(cfg)
        result.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        result.last_step = 70
        result._select_frontier(freeze=True)
        with torch.no_grad():
            result.gs_translation[:, 0] = .02
        result(101, training=True)
        return result

    def test_quadrature_recovers_conditional_image_gain_with_a_shared_birth_source(self):
        # The child contributes to the same rendered pixel as its source.
        # A redundant source update is harmful after the child is introduced.
        parameter = torch.tensor([[.2, 0, 0], [.3, 0, 0]], requires_grad=True)
        sources = torch.tensor([0])
        target = torch.tensor([.1, .3])
        evidence = MotionEvidence(parameter)

        def photo(alpha):
            old = alpha * parameter[:, 0]
            new = .1 + old[sources]
            image = old.index_add(0, sources, new)
            return .5 * (image - target).square()

        for alpha in PROBE_ALPHAS:
            gradient, = torch.autograd.grad(photo(alpha).sum(), parameter)
            evidence.add(gradient, parameter.detach(), alpha)
        expected = (photo(0.) - photo(1.)).detach()
        torch.testing.assert_close(evidence.total / evidence.samples, expected)
        self.assertLess(float(expected[0]), 0.)
        self.assertGreater(float(expected[1]), 0.)
        self.assertIsNone(parameter.grad)

    def fake_module(self, motion):
        binding = SimpleNamespace(xyz=None)
        def bind(xyz, rotation, detach=True):
            binding.xyz = xyz.detach().clone()
        births = SimpleNamespace(bind_source_pose=bind)
        module = SimpleNamespace(motion=motion, motion_committed=False,
                                 early_initialized=True, births=births)
        def forward(view, training=False):
            xyz, rotation = motion(160, training=training)
            births.bind_source_pose(xyz, rotation)
            return {'render': xyz[:, 0]}
        module.forward = forward
        return module, binding

    def test_probe_reuses_the_view_preserves_gradients_rng_and_restores_birth_pose(self):
        motion = self.model()
        module, binding = self.fake_module(motion)
        motion.gs_translation.grad = torch.full_like(motion.gs_translation, .7)
        before = motion.gs_translation.grad.clone()
        rng = torch.get_rng_state().clone()
        target = motion.base_xyz[:, 0] + .03
        probe_motion_evidence(module, 'current_view', 160, lambda image: (image-target).square().mean())
        self.assertEqual(motion._motion_evidence.samples, 2)
        self.assertIsNone(motion._counterfactual_probe_scale)
        torch.testing.assert_close(binding.xyz, motion(160)[0])
        torch.testing.assert_close(motion.gs_translation.grad, before, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        probe_motion_evidence(module, 'current_view', 159, lambda image: image.sum())
        self.assertEqual(motion._motion_evidence.samples, 2)

    def test_probe_restores_physical_source_pose_after_a_loss_error(self):
        motion = self.model()
        module, binding = self.fake_module(motion)
        def fail(image):
            raise RuntimeError('synthetic loss failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic loss'):
            probe_motion_evidence(module, None, 160, fail)
        self.assertIsNone(motion._counterfactual_probe_scale)
        torch.testing.assert_close(binding.xyz, motion(160)[0])

    def test_selection_removes_redundant_rows_survives_decode_and_blocks_adam_regrowth(self):
        motion = self.model()
        evidence = motion._motion_evidence
        evidence.samples = 80
        evidence.seen.fill_(True)
        half = len(motion.owner)//2
        evidence.total[:half] = -.01
        evidence.total[half:] = .01
        parameter = motion.gs_translation
        motion.optimizer.state[parameter] = dict(step=torch.tensor(1.),
            exp_avg=torch.ones_like(parameter), exp_avg_sq=torch.ones_like(parameter))
        xyz, _ = motion(180, training=True)
        self.assertTrue(evidence.done)
        self.assertFalse(bool(motion.gs_translation[:half].ne(0).any()))
        self.assertTrue(bool(motion.gs_translation[half:, 0].ne(0).all()))
        self.assertFalse(bool(motion.optimizer.state[parameter]['exp_avg'][:half].ne(0).any()))
        decoded = decode_motion(codec_fixtures.JointImageRateTests.packet(motion), motion.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=0)
        (motion(181, training=True)[0]-motion.base_xyz-.05).square().mean().backward()
        motion.optimizer_step()
        self.assertFalse(bool(motion.gs_translation[:half].ne(0).any()))
        motion.retain_survivors(torch.arange(len(motion.owner))[::2])
        self.assertEqual(len(motion._motion_evidence.total), len(motion.owner))

    def test_resume_keeps_evidence_and_next_camera_contribution_exact(self):
        motion = self.model()
        module, _ = self.fake_module(motion)
        target = motion.base_xyz[:, 0] + .03
        loss = lambda image: (image-target).square().mean()
        probe_motion_evidence(module, None, 160, loss)
        restored = SpatialGatedMotion(copy.deepcopy(motion.config))
        restored.restore(copy.deepcopy(motion.capture()))
        other, _ = self.fake_module(restored)
        probe_motion_evidence(module, None, 161, loss)
        probe_motion_evidence(other, None, 161, loss)
        self.assertEqual(restored._motion_evidence.samples, 4)
        torch.testing.assert_close(restored._motion_evidence.total, motion._motion_evidence.total,
                                   rtol=0, atol=0)
        packet = copy.deepcopy(motion.capture())
        packet['conditional_motion_evidence']['objective'] = 'incompatible'
        with self.assertRaisesRegex(ValueError, 'objective'):
            restored.restore(packet)

    def test_finalization_requires_observations_and_configuration_rejects_wrong_coordinates(self):
        motion = self.model()
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            motion(180, training=True)
        for name in ('rotation_updates',):
            cfg = copy.deepcopy(motion.config)
            setattr(cfg, name, True)
            with self.assertRaises(ValueError):
                cfg.validate()
        cfg = copy.deepcopy(motion.config)
        cfg.amplitude_refit_steps = 20
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg = copy.deepcopy(motion.config)
        cfg.motion_fit_window_steps = True
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == '__main__':
    unittest.main()

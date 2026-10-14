"""Candidate checks kept outside the frozen running experiment."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

from test_sparse_opacity import OPACITY
torch.set_num_threads(2)

class AlphaOpacityTests(unittest.TestCase):
    def test_saturated_gradient_and_adam_fit(self):
        ref = torch.tensor([[30.], [20.], [-30.], [0.]])
        delta = OPACITY.SparseOpacityDelta(ref, .001, mode='alpha')
        optimizer = torch.optim.Adam([delta.delta], lr=.0125, eps=1e-15)
        target = torch.tensor([[.1], [.2], [.8], [.5]])
        loss = (delta.activated() - target).square().mean()
        loss.backward()
        torch.testing.assert_close(delta.delta.grad, torch.tensor([[.45], [.4], [-.4], [0.]]), rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        initial = float(loss.detach())
        for _ in range(100):
            loss = (delta.activated() - target).square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            delta.proximal_zero(optimizer, 0)
        self.assertLess(float(loss.detach()), initial / 100)
        packet = delta.packet(2)
        self.assertEqual(packet[:4], b'OPD2')
        decoded = OPACITY.decode_opacity_delta(packet, ref, frame_index=2)
        self.assertEqual(decoded.mode, 'alpha')
        self.assertEqual(decoded.logits.numpy().tobytes(), delta(False).numpy().tobytes())
        self.assertTrue(torch.equal(delta.activated(False), decoded.logits.sigmoid()))

    def test_endpoints_signed_zero_and_restore(self):
        ref = torch.tensor([[30.], [-30.], [-0.], [0.], [-2.]])
        delta = OPACITY.SparseOpacityDelta(ref, .001, mode='alpha')
        self.assertEqual(delta(False).numpy().tobytes(), ref.numpy().tobytes())
        with torch.no_grad():
            delta.delta[:2] = torch.tensor([[-2.], [2.]])
        optimizer = torch.optim.Adam([delta.delta], lr=.0125)
        delta.proximal_zero(optimizer, 0)
        self.assertTrue(torch.equal(delta.activated(False)[:2], torch.tensor([[0.], [1.]])))
        decoded = OPACITY.decode_opacity_delta(delta.packet(4), ref)
        self.assertEqual(decoded.logits.numpy().tobytes(), delta(False).numpy().tobytes())
        restored = OPACITY.SparseOpacityDelta(ref + 3, .2, mode='alpha')
        restored.load_state_dict(delta.state_dict())
        self.assertEqual(restored(False).numpy().tobytes(), delta(False).numpy().tobytes())
        self.assertTrue(torch.equal(restored._alpha, delta._alpha))

    def test_no_outward_packet_at_saturation(self):
        ref = torch.tensor([[100.], [-100.]])
        delta = OPACITY.SparseOpacityDelta(ref, .001, mode='alpha')
        with torch.no_grad():
            delta.delta[:] = torch.tensor([[1.], [-1.]])
        self.assertIsNone(delta.packet(1))
        self.assertTrue(torch.equal(delta(False), ref))

    def test_sparse_identity_and_extreme_codec(self):
        ref = torch.tensor([[30.], [-30.], [-0.], [1.]])
        packet = OPACITY.encode_opacity_delta(torch.tensor([[-32768], [32767], [0], [-1]]),
                    ref, step=.001, frame_index=1, mode='alpha')
        decoded = OPACITY.decode_opacity_delta(packet, ref)
        self.assertEqual(decoded.logits[2].numpy().tobytes(), ref[2].numpy().tobytes())
        self.assertTrue(torch.equal(decoded.logits[:2], torch.tensor([[-100.], [100.]])))
        self.assertTrue(torch.isfinite(decoded.logits).all())
        with self.assertRaises(ValueError):
            OPACITY.decode_opacity_delta(packet, ref + 1)
        with self.assertRaises(ValueError):
            OPACITY.SparseOpacityDelta(ref, mode='bad')


class AlphaOpacityModuleTests(__import__('test_sparse_opacity_module').SparseOpacityModuleTests):
    def fixture(self, directory, with_birth=True):
        module, source, birth, view = super().fixture(directory, with_birth)
        module.config.old_opacity.mode = 'alpha'
        module.config.old_opacity.step = .001
        module.config.old_opacity.learning_rate = .0125
        module._initialize_old_opacity()
        return module, source, birth, view

class LegacyOpacityCheckpointTests(unittest.TestCase):
    def test_missing_mode_restores_logit_and_rejects_alpha(self):
        import tempfile
        from unittest.mock import patch
        from test_sparse_opacity_module import SparseOpacityModuleTests
        case = SparseOpacityModuleTests()
        with tempfile.TemporaryDirectory() as directory:
            old, source, birth, view = case.fixture(Path(directory) / 'old')
            case.train_opacity(old, view, 201, 210)
            checkpoint = dict(global_step=210)
            old.on_save_checkpoint(checkpoint)
            checkpoint['rigid_tree_module_config']['old_opacity'].pop('mode')
            restored, _, birth, _ = case.fixture(Path(directory) / 'restored')
            with patch.object(source, 'BirthGaussianModel', birth):
                restored.on_load_checkpoint(checkpoint)
            self.assertEqual(restored.old_opacity_delta.mode, 'logit')
            self.assertEqual(restored.old_opacity_delta(False).numpy().tobytes(), old.old_opacity_delta(False).numpy().tobytes())
            alpha, _, birth, _ = case.fixture(Path(directory) / 'alpha')
            alpha.config.old_opacity.mode = 'alpha'
            with patch.object(source, 'BirthGaussianModel', birth), self.assertRaisesRegex(ValueError, 'module configuration'):
                alpha.on_load_checkpoint(checkpoint)

class AlphaOpacityGPUChecks(unittest.TestCase):
    mode = 'alpha'
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA numerical check')
    def test_canonical_cpu_cuda_packets_and_active_gradient(self):
        generator = torch.Generator().manual_seed(414)
        reference = torch.randn(4096, 1, generator=generator) * 20
        reference[:4] = torch.tensor([[30.], [-30.], [-0.], [0.]])
        values = torch.randn(4096, 1, generator=generator) * .2
        values[:4] = torch.tensor([[-.01], [.01], [0.], [0.]])
        cpu = OPACITY.SparseOpacityDelta(reference, .001, mode=self.mode)
        gpu = OPACITY.SparseOpacityDelta(reference.cuda(), .001, mode=self.mode)
        with torch.no_grad():
            cpu.delta.copy_(values)
            gpu.delta.copy_(values.cuda())
        self.assertEqual(cpu(False).numpy().tobytes(), gpu(False).cpu().numpy().tobytes())
        decoded = OPACITY.decode_opacity_delta(gpu.packet(1), reference)
        self.assertEqual(decoded.logits.numpy().tobytes(), cpu(False).numpy().tobytes())
        self.assertTrue(torch.equal(gpu.activated(False), decoded.logits.cuda().sigmoid()))
        gpu.activated().square().sum().backward()
        self.assertGreater(float(gpu.delta.grad[0]), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA renderer check')
    def test_real_rasterizer_saturated_alpha_has_image_gradient(self):
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
        from libgs.data.utils.graphics import getProjectionMatrix
        import math
        projection = getProjectionMatrix(znear=.01, zfar=100, fovX=1., fovY=1.).transpose(0, 1).cuda()
        settings = GaussianRasterizationSettings(image_height=64, image_width=64,
            tanfovx=math.tan(.5), tanfovy=math.tan(.5), bg=torch.zeros(3, device='cuda'),
            scale_modifier=1., viewmatrix=torch.eye(4, device='cuda'), projmatrix=projection,
            sh_degree=0, campos=torch.zeros(3, device='cuda'), prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(settings)
        means = torch.tensor([[0., 0., 2.]], device='cuda')
        delta = OPACITY.SparseOpacityDelta(torch.tensor([[30.]], device='cuda'), .001, mode=self.mode)
        image = rasterizer(means3D=means, means2D=torch.zeros_like(means, requires_grad=True),
            shs=None, colors_precomp=torch.ones(1, 3, device='cuda'), opacities=delta.activated(),
            scales=torch.full((1, 3), .1, device='cuda'), rotations=torch.tensor([[1., 0., 0., 0.]], device='cuda'),
            cov3D_precomp=None)[0]
        image.square().mean().backward()
        self.assertGreater(float(delta.delta.grad[0]), 0)


class SaturatedAlphaGPUChecks(AlphaOpacityGPUChecks):
    mode = 'saturated_alpha'


class SaturatedAlphaTests(unittest.TestCase):
    def test_ordinary_forward_and_gradient_match_original_logit(self):
        reference = torch.tensor([[-0.], [1.], [-2.], [3.], [30.]])
        old = OPACITY.SparseOpacityDelta(reference, .01)
        corrected = OPACITY.SparseOpacityDelta(reference, .01, mode='saturated_alpha')
        values = torch.tensor([[0.], [.12], [-.23], [.09], [-.4]])
        with torch.no_grad():
            old.delta.copy_(values)
            corrected.delta.copy_(values)
        self.assertEqual(old(False)[:4].numpy().tobytes(), corrected(False)[:4].numpy().tobytes())
        weights = torch.tensor([[.33], [1.7], [-.142], [6.62], [1.]])
        (old.activated() * weights).sum().backward()
        (corrected.activated() * weights).sum().backward()
        torch.testing.assert_close(old.delta.grad[:4], corrected.delta.grad[:4], rtol=0, atol=0)
        self.assertEqual(float(corrected.delta.grad[4]), .25)

    def test_mixed_packet_reference_restore_and_immutable_eligibility(self):
        reference = torch.tensor([[-0.], [1.], [30.], [6.], [-2.]])
        delta = OPACITY.SparseOpacityDelta(reference, .01, mode='saturated_alpha')
        with torch.no_grad():
            delta.delta[:] = torch.tensor([[0.], [.12], [-4.], [-2.], [.21]])
        packet = delta.packet(3)
        self.assertEqual(packet[:4], b'OPD3')
        decoded = OPACITY.decode_opacity_delta(packet, reference, frame_index=3)
        self.assertEqual(decoded.mode, 'saturated_alpha')
        self.assertEqual(decoded.logits.numpy().tobytes(), delta(False).numpy().tobytes())
        self.assertEqual(float(decoded.logits[2]), -100.)
        self.assertEqual(float(decoded.logits[1]), float(reference[1] + 12 * delta.step))
        # Eligibility remains based on the preceding decode, even after falling below .99.
        self.assertTrue(bool(delta._saturated[3]))
        self.assertLess(float(delta.activated(False)[3]), .99)
        restored = OPACITY.SparseOpacityDelta(reference + 1, .2, mode='saturated_alpha')
        restored.load_state_dict(delta.state_dict())
        self.assertTrue(torch.equal(restored._saturated, delta._saturated))
        self.assertEqual(restored(False).numpy().tobytes(), delta(False).numpy().tobytes())


class SaturatedAlphaModuleTests(__import__('test_sparse_opacity_module').SparseOpacityModuleTests):
    def fixture(self, directory, with_birth=True):
        import numpy as np
        module, source, birth, view = super().fixture(directory, with_birth)
        module.config.old_opacity.mode = 'saturated_alpha'
        # Put a truly saturated old point through the real hooks, birth predictor and restore.
        with torch.no_grad():
            module.gaussians._opacity[0] = 30.
        reference = module.trainer.decoder_reference_path
        with np.load(reference, allow_pickle=False) as archive:
            raw = {key: archive[key].copy() for key in archive.files}
        raw['opacity'][0] = 30.
        np.savez_compressed(reference, **raw)
        module._initialize_old_opacity()
        return module, source, birth, view

if __name__ == '__main__':
    unittest.main()

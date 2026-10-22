"""Quantized opacity learning, sparse updates, current-alpha retirement, and exact replay."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.module import Module
from pipeline.trainer import Trainer
from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
from pipeline.model.gaussian import GaussianModel
from pipeline.model.legacy_opacity_codec import decode_opacity_delta
from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate, encode_opacity_delta
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from scripts.replay_rigid_tree import decode_frame
from _stream_fixtures import Rows, reference_raw, motion_model, export_frame


class OpacityUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_opacity_symbols_keep_bits_and_image_gradient(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, step=.1))
        reference = torch.tensor([[-0.], [1.], [-2.]])
        update.setup(reference)
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.02], [.13], [-.08]])
        logits = update()
        self.assertEqual(int(logits.detach().view(torch.int32)[0]),
                         int(reference.view(torch.int32)[0]))
        logits.sigmoid().sum().backward()
        self.assertTrue(bool(update.delta.grad.gt(0).all()))
        self.assertTrue(torch.isfinite(update.delta.grad).all())
        torch.testing.assert_close(update.canonical_logits(), logits.detach(), rtol=0, atol=0)

    def test_opacity_rate_only_pushes_nonzero_quantized_updates(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, step=.05))
        update.setup(torch.zeros(3, 1))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.01], [.03], [-.1]])
        terms = update.rate_terms(1.)
        self.assertEqual(int(terms['active_rows']), 2)
        terms['loss'].backward()
        self.assertEqual(float(update.delta.grad[0]), 0.)
        self.assertGreater(float(update.delta.grad[1]), 0.)
        self.assertLess(float(update.delta.grad[2]), 0.)

    def test_proximal_opacity_adam_keeps_only_image_second_moments(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, proximal_regularization=True,
                                                            l1_weight=.005))
        update.setup(torch.tensor([[0.], [1.], [-1.]]))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.03], [-.1], [.2]])
        image_loss = (update().sigmoid() - .6).square().mean()
        gradient, = torch.autograd.grad(image_loss, [update.delta], retain_graph=True)
        rate = update.rate_terms(.01)['loss']
        self.assertFalse(rate.requires_grad)
        (image_loss + rate).backward()
        update.optimizer_step(0., proximal=False)
        torch.testing.assert_close(update.optimizer.state[update.delta]['exp_avg_sq'],
            gradient.square() * .001, atol=1e-15, rtol=1e-6)

    def test_proximal_opacity_clears_weak_latents_and_retains_large_update(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, proximal_regularization=True))
        update.setup(torch.zeros(3,1))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.01],[.03],[2.]])
        update.optimizer.state[update.delta] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(update.delta), exp_avg_sq=torch.full_like(update.delta, 1e-11))
        update.delta.grad = torch.zeros_like(update.delta)
        update.optimizer_step(1e-6, proximal=True)
        self.assertEqual(float(update.delta[0]),0.)
        self.assertEqual(float(update.delta[1]),0.)
        self.assertEqual(float(update.delta[2]),2.)

    def test_continuous_opacity_penalty_cannot_create_opposite_updates(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha',
            proximal_regularization=True, l1_weight=.005, learning_rate=.01))
        update.setup(torch.zeros(3, 1))
        with torch.no_grad():
            update.delta[:, 0] = torch.tensor([.1, -.1, 1e-5])
        update.optimizer.state[update.delta] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(update.delta), exp_avg_sq=torch.full_like(update.delta, 1e-8))
        update.delta.grad = torch.zeros_like(update.delta)
        update.optimizer_step(0., lr_factor=.2)
        self.assertGreater(float(update.delta[0]), 0.)
        self.assertLess(float(update.delta[0]), .1)
        self.assertLess(float(update.delta[1]), 0.)
        self.assertGreater(float(update.delta[1]), -.1)
        self.assertEqual(float(update.delta[2]), 0.)
        self.assertAlmostEqual(update.optimizer.param_groups[0]['lr'], .002)

    def test_legacy_opacity_config_defaults_and_proximal_resume(self):
        old = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True))
        old.setup(torch.zeros(3,1))
        saved = copy.deepcopy(old.capture())
        saved['config'].pop('proximal_regularization')
        restored = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True))
        restored.restore(saved,torch.device('cpu'))
        torch.testing.assert_close(old(),restored(),atol=0,rtol=0)
        cfg = OpacityUpdateConfig(enabled=True,proximal_regularization=True)
        one = QuantizedOpacityUpdate(cfg); one.setup(torch.zeros(3,1))
        one().sigmoid().sum().backward(); one.optimizer_step(0.)
        two = QuantizedOpacityUpdate(copy.deepcopy(cfg));two.restore(copy.deepcopy(one.capture()),torch.device('cpu'))
        for current in (one,two):
            current().sigmoid().sum().backward();current.optimizer_step(1e-8,proximal=True)
        torch.testing.assert_close(one.delta,two.delta,atol=0,rtol=0)

    def test_opacity_refit_keeps_zero_rows_frozen_and_removes_amplitude_bias(self):
        cfg = OpacityUpdateConfig(enabled=True, mode='alpha', step=.01,
            proximal_regularization=True, l1_weight=100., amplitude_refit_steps=20)
        update = QuantizedOpacityUpdate(cfg)
        update.setup(torch.zeros(3, 1))
        with torch.no_grad():
            update.delta[:, 0] = torch.tensor([.004, .1, -.1])
        update.optimizer.state[update.delta] = dict(step=torch.tensor(1.),
            exp_avg=torch.ones_like(update.delta), exp_avg_sq=torch.ones_like(update.delta))
        update.prepare_step(80, 100)
        self.assertIsNone(update.refit_support)
        before = update.symbols().ne(0)
        update.prepare_step(81, 100)
        self.assertEqual(float(update.delta[0]), 0.)
        self.assertNotIn(update.delta, update.optimizer.state)
        self.assertFalse(update.rate_terms(100.)['loss'].requires_grad)
        for _ in range(3):
            # Huge rate/L1 penalties would erase all updates without refitting.
            (-update.alphas().sum()).backward()
            update.optimizer_step(100., proximal=True)
            self.assertEqual(float(update.delta[0]), 0.)
            self.assertFalse(bool((update.symbols().ne(0) & ~before).any()))
        self.assertGreater(float(update.delta[1]), .1)

    def test_opacity_refit_resume_and_survivor_selection_are_exact(self):
        cfg = OpacityUpdateConfig(enabled=True, mode='alpha', step=.01, amplitude_refit_steps=20)
        one = QuantizedOpacityUpdate(cfg)
        one.setup(torch.zeros(3, 1))
        with torch.no_grad():
            one.delta[:, 0] = torch.tensor([0., .1, -.1])
        one.prepare_step(81, 100)
        one.alphas().sum().backward(); one.optimizer_step(0.)
        two = QuantizedOpacityUpdate(copy.deepcopy(cfg))
        two.restore(copy.deepcopy(one.capture()), torch.device('cpu'))
        for update in (one, two):
            update.prepare_step(82, 100)
            update.alphas().sum().backward(); update.optimizer_step(1., proximal=True)
            update.commit(torch.tensor([2, 0]))
        torch.testing.assert_close(one.delta, two.delta, atol=0, rtol=0)
        torch.testing.assert_close(one.canonical_logits(), two.canonical_logits(), atol=0, rtol=0)
        self.assertEqual(float(two.delta[0]), 0.)
        with tempfile.TemporaryDirectory() as directory:
            packet = two.export(Path(directory)/'opacity_002.bin', 2)
            decoded = decode_opacity_delta(Path(packet['path']).read_bytes(), two.reference[two.rows], frame_index=2)
            torch.testing.assert_close(decoded.logits, two.canonical_logits(), atol=0, rtol=0)

    def test_opacity_refit_validation_and_legacy_default(self):
        for value in (True, -1, .5):
            with self.assertRaises(ValueError):
                OpacityUpdateConfig(enabled=True, amplitude_refit_steps=value).validate()
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True))
        update.setup(torch.zeros(2, 1))
        saved = copy.deepcopy(update.capture())
        saved.pop('refit_support')
        saved['config'].pop('amplitude_refit_steps')
        update.restore(saved, torch.device('cpu'))
        update.prepare_step(100, 100)
        self.assertIsNone(update.refit_support)
        bad = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, amplitude_refit_steps=100))
        bad.setup(torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            bad.prepare_step(1, 100)

    def test_birth_opacity_joint_fit_retains_previous_frame_reference_and_resumes(self):
        cfg = OpacityUpdateConfig(enabled=True, mode='alpha', step=.01,
            joint_with_birth=True, proximal_regularization=True)
        one = QuantizedOpacityUpdate(cfg)
        one.setup(torch.tensor([[1.], [-1.], [0.]]))
        one.commit(torch.tensor([2, 0]))
        self.assertTrue(one.delta.requires_grad)
        reference = one.reference.clone()
        one.alphas().sum().backward(); one.optimizer_step(0.)
        self.assertEqual(float(one.delta[1]), 0.)  # Deleted before birth.
        two = QuantizedOpacityUpdate(copy.deepcopy(cfg))
        two.restore(copy.deepcopy(one.capture()), torch.device('cpu'))
        for current in (one, two):
            self.assertTrue(current.delta.requires_grad)
            current.alphas().sum().backward(); current.optimizer_step(0.)
            current.finish()
        torch.testing.assert_close(one.delta, two.delta, rtol=0, atol=0)
        torch.testing.assert_close(two.reference, reference, rtol=0, atol=0)
        finished = QuantizedOpacityUpdate(copy.deepcopy(cfg))
        finished.restore(copy.deepcopy(two.capture()), torch.device('cpu'))
        self.assertFalse(finished.delta.requires_grad)
        with tempfile.TemporaryDirectory() as directory:
            info = two.export(Path(directory)/'opacity_002.bin', 2)
            decoded = decode_opacity_delta(Path(info['path']).read_bytes(), reference[two.rows], frame_index=2)
            torch.testing.assert_close(decoded.logits, two.canonical_logits(), atol=0, rtol=0)

    def test_alpha_mode_has_unsaturated_image_gradient_and_zero_bits(self):
        cfg=OpacityUpdateConfig(enabled=True,mode='alpha',step=.01,learning_rate=.01)
        update=QuantizedOpacityUpdate(cfg)
        update.setup(torch.tensor([[-0.],[7.],[-7.]]))
        output=update.alphas()
        output.sum().backward()
        torch.testing.assert_close(update.delta.grad,torch.ones(3,1),atol=0,rtol=0)
        self.assertEqual(int(update().detach().view(torch.int32)[0]),int(update.reference.view(torch.int32)[0]))

    def test_alpha_mode_render_grid_and_exact_canonical_packet(self):
        cfg=OpacityUpdateConfig(enabled=True,mode='alpha',step=.01)
        update=QuantizedOpacityUpdate(cfg)
        update.setup(torch.tensor([[0.],[5.],[-5.],[1.]]))
        with torch.no_grad():update.delta[:,0]=torch.tensor([.024,-.995,.025,0.])
        expected=(update.alpha_reference+update.symbols()*update.step).clamp(0,1)
        torch.testing.assert_close(update.alphas(),expected,atol=1e-7,rtol=0)
        packet=encode_opacity_delta(update.reference,update.symbols(),frame_index=2,step=cfg.step,mode='alpha')
        decoded=decode_opacity_delta(packet,update.reference)
        self.assertEqual(decoded.mode,'alpha_canonical')
        torch.testing.assert_close(decoded.logits,update.canonical_logits(),atol=0,rtol=0)
        self.assertLess(float(decoded.logits[1].sigmoid()),.01)

    def test_alpha_mode_bounds_and_checkpoint_state(self):
        cfg=OpacityUpdateConfig(enabled=True,mode='alpha',step=.01)
        update=QuantizedOpacityUpdate(cfg);update.setup(torch.tensor([[7.],[-7.],[0.]]))
        with torch.no_grad():update.delta[:,0]=torch.tensor([20.,-20.,.11])
        update.delta.grad=torch.zeros_like(update.delta)
        update.optimizer_step(0.)
        self.assertTrue(bool(update.delta.ge(-update.alpha_reference).all()))
        self.assertTrue(bool(update.delta.le(1-update.alpha_reference).all()))
        restored=QuantizedOpacityUpdate(copy.deepcopy(cfg));restored.restore(copy.deepcopy(update.capture()),torch.device('cpu'))
        torch.testing.assert_close(update.alphas(),restored.alphas(),atol=0,rtol=0)
        torch.testing.assert_close(update.canonical_logits(),restored.canonical_logits(),atol=0,rtol=0)

    def test_alpha_packet_is_invariant_to_survivor_selection_and_simd_tails(self):
        generator = torch.Generator().manual_seed(295)
        reference = torch.randn(4099, 1, generator=generator) * 4
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        update.setup(reference)
        with torch.no_grad():
            update.delta.copy_(torch.randn(4099, 1, generator=generator) * .08)
        full = update.canonical_logits(full=True)
        for count in (1, 7, 31, 32, 101, 3001, 4099):
            rows = torch.randperm(4099, generator=generator)[:count]
            selected_ref = reference[rows]
            packet = encode_opacity_delta(selected_ref, update.symbols()[rows],
                                           frame_index=2, step=.01, mode='alpha')
            if packet is None:
                decoded = selected_ref
            else:
                decoded = decode_opacity_delta(packet, selected_ref).logits
            torch.testing.assert_close(decoded.view(torch.int32),
                                       full[rows].view(torch.int32), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_gpu_opacity_materialization_matches_cpu_packet_bits(self):
        generator = torch.Generator().manual_seed(83)
        reference = torch.randn(1024, 1, generator=generator)
        reference[0] = -0.
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, step=.05))
        update.setup(reference.cuda())
        with torch.no_grad():
            update.delta.copy_(torch.randn(1024, 1, generator=generator).cuda() * .2)
            update.delta[0] = .01
        rendered = update().detach().cpu()
        torch.testing.assert_close(rendered.view(torch.int32),
                                   update.canonical_logits().view(torch.int32), rtol=0, atol=0)

    def test_gap_packet_exact_roundtrip_checksums_and_invalid_symbols(self):
        reference = torch.linspace(-4., 4., 300)[:, None]
        symbols = torch.zeros_like(reference)
        symbols[[0, 150, 299], 0] = torch.tensor([-200., 1., 32000.])
        packet = encode_opacity_delta(reference, symbols, frame_index=2, step=.05)
        decoded = decode_opacity_delta(packet, reference, frame_index=2)
        expected = reference.clone()
        expected[decoded.rows, 0] += decoded.symbols.float() * decoded.step
        torch.testing.assert_close(decoded.logits, expected, rtol=0, atol=0)
        damaged = bytearray(packet)
        damaged[-1] ^= 1
        with self.assertRaises(ValueError):
            decode_opacity_delta(damaged, reference, frame_index=2)
        self.assertIsNone(encode_opacity_delta(reference, torch.zeros_like(symbols), frame_index=2, step=.05))
        for bad in [.5, 32768., float('nan')]:
            invalid = symbols.clone()
            invalid[0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                encode_opacity_delta(reference, invalid, frame_index=2, step=.05)

    def test_current_opacity_retires_previously_opaque_point_and_replays_births(self):
        raw = reference_raw()
        raw['opacity'][0] = torch.logit(torch.tensor(.99))
        motion = motion_model(raw)
        initial = copy.deepcopy(motion.frame_reference_state)
        opacity_cfg = OpacityUpdateConfig(enabled=True, step=.05)
        update = QuantizedOpacityUpdate(opacity_cfg)
        update.setup(raw['opacity'])
        with torch.no_grad():
            update.delta[0] = -12.
            update.delta[1] = .1
        restored = QuantizedOpacityUpdate(copy.deepcopy(opacity_cfg))
        restored.restore(copy.deepcopy(update.capture()), torch.device('cpu'))
        torch.testing.assert_close(restored(), update(), rtol=0, atol=0)
        holder = SimpleNamespace(opacity_update=restored, gaussians=Rows(raw), motion=motion,
            config=SimpleNamespace(opacity_update=opacity_cfg), device=torch.device('cpu'),
            birth_control=dict(previous_birth_count=3), retired_reference=None)
        Module._commit_opacity_and_retire(holder)
        self.assertEqual(holder.birth_control['retirement']['retired_count'], 1)
        self.assertEqual(len(holder.gaussians._xyz), 7)
        torch.testing.assert_close(motion.previous_rows, torch.arange(1, 8))
        loaded_motion = SpatialGatedMotion(copy.deepcopy(motion.config))
        loaded_motion.restore(copy.deepcopy(motion.capture()))
        self.assertEqual(len(loaded_motion.gs_translation), 7)
        holder.frame_finalized = False
        holder.initial_reference = dict(original_count=8, retained_count=8,
                                        removed_count=0, threshold=0.)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            worker = SimpleNamespace(output_dir=directory/'resume', module=holder,
                datamodule=SimpleNamespace(cameras_extent=2.))
            Trainer._write_reference(worker, 0, initial)
            with np.load(worker.decoder_reference_path) as reference:
                for name in RAW_FIELDS:
                    torch.testing.assert_close(torch.from_numpy(reference[name]), raw[name], rtol=0, atol=0)
            current_raw = model_raw(holder.gaussians)
            expected = export_frame(directory, loaded_motion, current_raw, torch.tensor([0]))
            info = restored.export(directory/'opacity_001.bin', 1)
            self.assertEqual(info['rows'], 1)  # Deleted point's large alpha delta is not transmitted.
            manifest = dict(opacity_update='opacity_001.bin', opacity_update_timing='before_births')
            (directory/'frame_001.json').write_text(json.dumps(manifest))
            decoded, replayed = decode_frame(directory, 1, initial, raw)
            self.assertEqual(len(decoded.xyz), 8)
            for name in RAW_FIELDS:
                torch.testing.assert_close(replayed[name], expected[name], rtol=0, atol=0)
            manifest['opacity_update_timing'] = 'after_births'
            (directory/'frame_001.json').write_text(json.dumps(manifest))
            _, wrong = decode_frame(directory, 1, initial, raw)
            self.assertFalse(torch.equal(wrong['opacity'], expected['opacity']))

    def test_alpha_override_uses_quantized_values_with_gradient(self):
        model = GaussianModel(1)
        model._opacity = torch.nn.Parameter(torch.tensor([[2.]]), requires_grad=False)
        override = torch.tensor([[-1.]], requires_grad=True)
        model.rigid_opacity = override
        model.get_opacity.sum().backward()
        self.assertGreater(float(override.grad), 0.)
        self.assertIsNone(model._opacity.grad)
        model.rigid_opacity = None
        torch.testing.assert_close(model.get_opacity, model._opacity.sigmoid())

    def test_materialized_opacity_invalidates_previous_render_cache(self):
        raw = reference_raw()
        raw['opacity'][0] = 0.
        model = GaussianModel(1)
        from pipeline.model.birth_quantization import RAW_ATTRIBUTES
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(model, attr, torch.nn.Parameter(raw[name].clone(), requires_grad=False))
        before = model.get_opacity.clone()
        config = OpacityUpdateConfig(enabled=True)
        update = QuantizedOpacityUpdate(config)
        update.setup(raw['opacity'])
        with torch.no_grad():
            update.delta[0] = .1
        holder = SimpleNamespace(opacity_update=update, gaussians=model,
            motion=motion_model(raw), config=SimpleNamespace(opacity_update=config),
            device=torch.device('cpu'), birth_control={}, retired_reference=None)
        Module._commit_opacity_and_retire(holder)
        self.assertFalse(torch.equal(model.get_opacity, before))
        torch.testing.assert_close(model.get_opacity, update.canonical_logits().sigmoid(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()

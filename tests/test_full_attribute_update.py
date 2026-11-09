"""CPU checks for full-attribute training, codec, benchmark and replay."""
import copy
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

from pipeline.frame_plan import FramePlanController
from pipeline.model import full_attribute_update as fua
from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
from pipeline.model.full_attribute_update import (FIELDS, HEADER, MAGIC, SPARSE_MAGIC,
    FullAttributeUpdate, FullUpdateConfig, FullUpdateSpatialGatedMotion,
    _encode_body, _reference_hash, decode_full_attribute_delta, encode_full_attribute_delta)
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.rigid_tree_codec import DecodedState, encode_motion
from pipeline.model.sparse_features import (FeatureDCUpdateConfig, QuantizedFeatureDCUpdate, RegionalFeatureDCUpdate,
    REGIONAL_MAGIC, _regional_digest, decode_feature_dc_delta)
from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from pipeline.module import Module
from pipeline.trainer import Trainer
from scripts import benchmark_rigid_residual_tree as benchmark
from scripts.replay_rigid_tree import decode_frame
from tests.test_frame_plan import controller_fixture, checkpoint_harness, cpu_gaussian_setup



class SixFieldRenderer:
    """Make every attribute observable, including rotation and higher SH."""
    def __init__(self, base):
        self.base = base
        self.gaussians = base.gaussians
        self.collect_contributions = False

    def __call__(self, view, background, override_color=None):
        self.base.gaussians = self.gaussians
        result = self.base(view, background, override_color=override_color)
        gs = self.gaussians
        extra = .0001 * (gs.get_rotation[:, 1:].sum() + gs.get_features[:, 1:].sum())
        result['render'] = result['render'] + extra
        return result


def attach_full_methods(module):
    for name in ('full_update_prepare_step', 'full_update_after_step',
                 'full_update_finish_pose_attributes',
                 'full_update_reference'):
        setattr(module, name, MethodType(getattr(Module, name), module))


def key_fixture(directory):
    """Compatibility helper backed by the supported per-frame repair fixture."""
    module, _ = fixture(directory)
    return module, module.transaction_camera_views()


def force_retire_at_events(module):
    cleanup = module._plan_cleanup
    @torch.no_grad()
    def retire(step):
        if step in (200, 250):
            old_row = 0 if step == 200 or not len(module.births.source_rows) else int(module.births.source_rows[0])
            original = module.opacity_update.rows[old_row]
            module.opacity_update.delta[original] = -module.opacity_update.alpha_reference[original]
        return cleanup(step)
    module._plan_cleanup = retire


class FullAttributeCodecTests(unittest.TestCase):

    def test_zero_full_update_can_only_be_skipped_while_frozen_and_unchanged(self):
        reference = dict(logscale=torch.randn(4, 3), sh_dc=torch.randn(4, 1, 3),
                         sh_rest=torch.randn(4, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = FullAttributeUpdate(FullUpdateConfig())
        update.setup(reference, optimizer, (.01, .01, .01))
        self.assertFalse(update.initial_zero_frozen())
        for parameter in update.parameters.values():
            parameter.requires_grad_(False)
        self.assertTrue(update.initial_zero_frozen())
        for name, result in update.apply(reference).items():
            self.assertTrue(torch.equal(result, reference[name]))
        with torch.no_grad():
            update.parameters['sh_dc'][0, 0, 0] = .01
        self.assertFalse(update.initial_zero_frozen())

    def test_roundtrip_zero_preservation_reference_and_corruption(self):
        ref = dict(logscale=torch.tensor([[-0., 1., 2.], [3., 4., 5.]]),
                   sh_dc=torch.zeros(2, 1, 3), sh_rest=torch.zeros(2, 3, 3))
        symbols = {k: torch.zeros_like(v) for k, v in ref.items()}
        steps = (.005, .001, .001)
        self.assertIsNone(encode_full_attribute_delta(ref, symbols, frame_index=10, steps=steps))
        symbols['logscale'][1, 0] = -13
        symbols['sh_dc'][0, 0, 0] = 12
        symbols['sh_rest'][1, 2, 1] = 7
        packet = encode_full_attribute_delta(ref, symbols, frame_index=10, steps=steps)
        result = decode_full_attribute_delta(packet, ref, frame_index=10)
        self.assertTrue(torch.signbit(result['logscale'][0, 0]))
        for name, step in zip(FIELDS, steps):
            self.assertTrue(torch.equal(result[name], torch.where(symbols[name].ne(0),
                                           ref[name] + symbols[name] * step, ref[name])))
        changed = {k: v.clone() for k, v in ref.items()}
        changed['sh_dc'][0, 0, 1] = 1
        with self.assertRaises(ValueError):
            decode_full_attribute_delta(packet, changed, frame_index=10)
        damaged = bytearray(packet); damaged[-1] ^= 1
        with self.assertRaises(ValueError):
            decode_full_attribute_delta(damaged, ref, frame_index=10)

    def test_shared_adam_retirement_and_checkpoint(self):
        ref = dict(logscale=torch.zeros(5, 3), sh_dc=torch.zeros(5, 1, 3),
                   sh_rest=torch.zeros(5, 3, 3))
        anchor = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([anchor])
        update = FullAttributeUpdate(FullUpdateConfig(enabled=True))
        update.setup(ref, optimizer, (.01, .02, .003))
        loss = sum(value.sum() for value in update.apply(ref).values())
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        keep = torch.tensor([4, 1, 3])
        previous = {k: v.detach()[keep].clone() for k, v in update.parameters.items()}
        oldstate = {k: optimizer.state[v]['exp_avg'][keep].clone() for k, v in update.parameters.items()}
        update.retain(keep, optimizer)
        for name in FIELDS:
            self.assertTrue(torch.equal(previous[name], update.parameters[name]))
            self.assertTrue(torch.equal(oldstate[name], optimizer.state[update.parameters[name]]['exp_avg']))
        packet = update.capture(optimizer)
        restored_opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        restored = FullAttributeUpdate(update.config)
        restored.restore(packet, restored_opt, 'cpu', 3)
        for name in FIELDS:
            self.assertTrue(torch.equal(restored.parameters[name], update.parameters[name]))
            self.assertEqual(int(restored_opt.state[restored.parameters[name]]['step']), 1)
            for key in ('exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(restored_opt.state[restored.parameters[name]][key],
                                            optimizer.state[update.parameters[name]][key]))
            old_group = next(g for g in optimizer.param_groups if g.get('name') == 'early_full_' + name)
            new_group = next(g for g in restored_opt.param_groups if g.get('name') == 'early_full_' + name)
            self.assertEqual({k: v for k, v in old_group.items() if k != 'params'},
                             {k: v for k, v in new_group.items() if k != 'params'})

    def test_sparse_and_dense_choose_shortest_and_read_legacy_dense(self):
        torch.manual_seed(9)
        ref = dict(logscale=torch.zeros(3000, 3), sh_dc=torch.zeros(3000, 1, 3),
                   sh_rest=torch.zeros(3000, 3, 3))
        ref['logscale'][0, 0] = -0.
        steps = (.005, .001, .001)
        sparse = {k: torch.zeros_like(v) for k, v in ref.items()}
        for index, name in enumerate(FIELDS):
            sparse[name][1000 * index + 3].flatten()[0] = 25 + index
        sparse['logscale'][0, 1] = 17
        sparse['sh_rest'][-1, -1, -1] = -27
        packet = encode_full_attribute_delta(ref, sparse, frame_index=10, steps=steps)
        self.assertEqual(packet[:4], SPARSE_MAGIC)
        decoded = decode_full_attribute_delta(packet, ref, frame_index=10)
        self.assertTrue(torch.signbit(decoded['logscale'][0, 0]))
        for name, step in zip(FIELDS, steps):
            self.assertTrue(torch.equal(decoded[name], torch.where(sparse[name].ne(0),
                ref[name] + sparse[name] * step, ref[name])))
        body = b''.join(sparse[name].to(torch.int16).numpy().astype('<i2').tobytes() for name in FIELDS)
        legacy = _encode_body(MAGIC, body, 10, 3000, 3, steps, _reference_hash(ref))
        self.assertLessEqual(len(packet), len(legacy))
        for name in FIELDS:
            self.assertTrue(torch.equal(decode_full_attribute_delta(legacy, ref, frame_index=10)[name],
                                        decoded[name]))
        dense = {k: torch.randint(-32767, 32768, v.shape).float() for k, v in ref.items()}
        packet = encode_full_attribute_delta(ref, dense, frame_index=10, steps=steps)
        self.assertEqual(packet[:4], MAGIC)
        decoded = decode_full_attribute_delta(packet, ref, frame_index=10)
        for name, step in zip(FIELDS, steps):
            self.assertTrue(torch.equal(decoded[name], torch.where(dense[name].ne(0),
                ref[name] + dense[name] * step, ref[name])))

    def test_sparse_decoder_rejects_bad_gaps_rows_and_trailing_bytes(self):
        ref = dict(logscale=torch.zeros(3, 3), sh_dc=torch.zeros(3, 1, 3),
                   sh_rest=torch.zeros(3, 3, 3))
        steps = (.005, .001, .001)
        good_row = struct.pack('<I', 1) + b'\x01' + struct.pack('<hhh', 1, 0, 0)
        empty = struct.pack('<I', 0)
        bad = [struct.pack('<I', 1) + b'\x00' + struct.pack('<hhh', 1, 0, 0) + empty * 2,
               struct.pack('<I', 1) + b'\x04' + struct.pack('<hhh', 1, 0, 0) + empty * 2,
               struct.pack('<I', 2) + b'\x01\x00' + struct.pack('<hhhhhh', 1, 0, 0, 1, 0, 0) + empty * 2,
               struct.pack('<I', 1) + b'\x81\x00' + struct.pack('<hhh', 1, 0, 0) + empty * 2,
               struct.pack('<I', 1) + b'\x80' + empty * 2,
               struct.pack('<I', 1) + b'\x01' + struct.pack('<hhh', 0, 0, 0) + empty * 2,
               struct.pack('<I', 1) + b'\x01' + struct.pack('<hhh', -32768, 0, 0) + empty * 2,
               good_row + empty * 2 + b'\x00']
        for body in bad:
            packet = _encode_body(SPARSE_MAGIC, body, 10, 3, 3, steps, _reference_hash(ref))
            with self.subTest(body=body), self.assertRaises(ValueError):
                decode_full_attribute_delta(packet, ref, frame_index=10)

    def test_sparse_formats_materialize_exact_bits_without_mutating_reference(self):
        torch.manual_seed(4242)
        steps = (.005, .001, .002)
        for rest in (0, 3, 15):
            ref = dict(logscale=torch.randn(257, 3), sh_dc=torch.randn(257, 1, 3),
                       sh_rest=torch.randn(257, rest, 3))
            symbols = {name: torch.zeros_like(value, dtype=torch.int16) for name, value in ref.items()}
            for name in FIELDS:
                if not symbols[name][1].numel():
                    continue
                # Zero components in a changed row must retain signed-zero bits.
                ref[name][1].view(-1)[0] = -0.
                symbols[name][1].view(-1)[1] = 32767
                symbols[name][127].view(-1)[0] = -32767
                symbols[name][-1].view(-1)[-1] = 7
            original = {name: value.clone() for name, value in ref.items()}
            body = fua._sparse_body([symbols[name].numpy() for name in FIELDS])
            planar = fua._sparse_zigzag_byteplane(body, len(ref['logscale']), rest)
            for magic, payload in ((SPARSE_MAGIC, body), (fua.PLANAR_MAGIC, planar)):
                with self.subTest(rest=rest, magic=magic):
                    packet = _encode_body(magic, payload, 7, 257, rest, steps, _reference_hash(ref))
                    decoded = decode_full_attribute_delta(packet, ref, frame_index=7)
                    for name, step in zip(FIELDS, steps):
                        integer = symbols[name].float()
                        expected = torch.where(integer.ne(0), ref[name] + integer * step, ref[name])
                        self.assertTrue(torch.equal(decoded[name].view(torch.int32), expected.view(torch.int32)))
                        self.assertTrue(torch.equal(ref[name].view(torch.int32), original[name].view(torch.int32)))
                    decoded['logscale'][0, 0] = 123.
                    self.assertTrue(torch.equal(ref['logscale'].view(torch.int32), original['logscale'].view(torch.int32)))

    def test_reordered_survivors_bind_reference_and_roundtrip(self):
        ref = dict(logscale=torch.arange(15).reshape(5, 3).float(),
                   sh_dc=torch.arange(15).reshape(5, 1, 3).float(),
                   sh_rest=torch.arange(45).reshape(5, 3, 3).float())
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = FullAttributeUpdate(FullUpdateConfig(enabled=True))
        update.setup(ref, optimizer, (.01, .01, .01))
        with torch.no_grad():
            update.parameters['sh_dc'][4, 0, 0] = .003
            update.parameters['sh_rest'][1, 2, 2] = -.004
        rows = torch.tensor([4, 1, 3])
        update.retain(rows, optimizer)
        kept = {name: value[rows] for name, value in ref.items()}
        packet = encode_full_attribute_delta(kept, update.symbols(), frame_index=10, steps=update.steps)
        decoded = decode_full_attribute_delta(packet, kept, frame_index=10)
        expected = update.canonical(kept)
        for name in FIELDS:
            self.assertTrue(torch.equal(decoded[name], expected[name]))
        swapped = {name: value[torch.tensor([1, 0, 2])] for name, value in kept.items()}
        with self.assertRaises(ValueError):
            decode_full_attribute_delta(packet, swapped, frame_index=10)

    def test_rate_and_prox_use_attribute_rows_keep_free_bins_and_allow_regrowth(self):
        ref = dict(logscale=torch.zeros(3, 3), sh_dc=torch.zeros(3, 1, 3),
                   sh_rest=torch.zeros(3, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = FullAttributeUpdate(FullUpdateConfig(enabled=True, per_attribute_l1_weight=0.))
        update.setup(ref, optimizer, (.01, .01, .01))
        sum(v.sum() for v in update.apply(ref).values()).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for value in update.parameters.values(): value.zero_()
            update.parameters['logscale'][0, 0] = .002
            update.parameters['logscale'][1, :2] = .01
            update.parameters['sh_dc'][2, 0, 0] = .002
            update.parameters['sh_rest'][1, 0, 0] = -.002
        terms = update.rate_terms(1.)
        self.assertEqual(int(terms['fields']['logscale']['active_rows']), 1)
        self.assertEqual(int(terms['fields']['logscale']['active_scalars']), 2)
        self.assertEqual(int(terms['fields']['sh_dc']['active_rows']), 1)
        self.assertEqual(int(terms['fields']['sh_rest']['active_rows']), 1)
        self.assertEqual(int(terms['raw_bytes_surrogate']), HEADER.size + 12 + 7 + 7 + 19)
        self.assertEqual(float(terms['loss']), float(terms['raw_bytes_surrogate']))
        free = update.parameters['logscale'][0, 0].detach().clone()
        update.after_step(optimizer, 1., proximal=True)
        self.assertEqual(float(update.parameters['logscale'][0, 0]), float(free))
        self.assertFalse(bool(update.parameters['logscale'][1].any()))
        self.assertFalse(bool(update.parameters['sh_dc'][2].any()))
        self.assertFalse(bool(update.parameters['sh_rest'][1].any()))
        # No candidate is frozen; the next image gradient can grow pruned rows.
        update.apply(ref)['logscale'][1].sum().backward()
        self.assertTrue(bool(update.parameters['logscale'].grad[1].ne(0).all()))




    def test_old_checkpoint_with_inactive_attribute_gate_can_resume(self):
        ref = dict(logscale=torch.zeros(2, 3), sh_dc=torch.zeros(2, 1, 3),
                   sh_rest=torch.zeros(2, 3, 3))
        config = FullUpdateConfig(enabled=True)
        original = FullAttributeUpdate(config)
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        original.setup(ref, optimizer, (.01, .01, .01))
        packet = original.capture(optimizer)
        packet['config'].update(learnable_gate=False, point_gate_enabled=False,
            gate_learning_rate=.1, temporal_focus_strength=0., tail_rmse_mix=0.)
        restored = FullAttributeUpdate(config)
        restored.restore(packet, torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]), 'cpu', 2)
        for name in FIELDS:
            torch.testing.assert_close(restored.parameters[name], original.parameters[name])
        packet['config']['learnable_gate'] = True
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            restored.restore(packet, torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]), 'cpu', 2)

    def test_l1_prox_matches_adam_metric_each_step(self):
        ref = dict(logscale=torch.zeros(2, 3), sh_dc=torch.zeros(2, 1, 3),
                   sh_rest=torch.zeros(2, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = FullAttributeUpdate(FullUpdateConfig(enabled=True, per_attribute_l1_weight=.005))
        update.setup(ref, optimizer, (.01, .01, .01))
        sum(v.sum() for v in update.apply(ref).values()).backward()
        optimizer.step()
        before = {name: value.detach().clone() for name, value in update.parameters.items()}
        expected = {}
        for name, value in update.parameters.items():
            state = optimizer.state[value]
            group = next(g for g in optimizer.param_groups if g.get('name') == 'early_full_' + name)
            metric = (state['exp_avg_sq'] / (1 - group['betas'][1] ** int(state['step']))).sqrt() + group['eps']
            threshold = group['lr'] * .005 / value.numel() / metric
            expected[name] = before[name].sign() * (before[name].abs() - threshold).clamp_min(0)
        update.after_step(optimizer, 0., proximal=False)
        for name in FIELDS:
            torch.testing.assert_close(update.parameters[name], expected[name], rtol=0, atol=0)


def make_stream(directory, *, magic):
    count = 512
    xyz = torch.zeros(count, 3)
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1
    owner = torch.ones(count, dtype=torch.long)
    parents = torch.tensor([-1, 0])
    prior = DecodedState(xyz, rotation, owner, parents, 0)
    raw = dict(xyz=xyz, rotation=rotation, logscale=torch.zeros(count, 3),
               opacity=torch.zeros(count, 1), sh_dc=torch.zeros(count, 1, 3),
               sh_rest=torch.zeros(count, 3, 3))
    region = torch.zeros(2, 3, dtype=torch.int16)
    points = torch.zeros(count, 3, dtype=torch.int16)
    motion = encode_motion(prior, 1, 1., torch.arange(count), owner, parents,
        region, region, points, points, torch.empty(0, dtype=torch.long),
        torch.empty(0, 3), torch.empty(0, 4), .001, .001)
    reference = {name: raw[name] for name in fua.FIELDS}
    if magic == fua.SPARSE_MAGIC:
        symbols = {name: torch.zeros_like(value) for name, value in reference.items()}
        for value in symbols.values():
            value[-1].flatten()[0] = 7
    else:
        generator = torch.Generator().manual_seed(12)
        symbols = {name: torch.randint(-100, 101, value.shape, generator=generator).float()
                   for name, value in reference.items()}
    arrays = [symbols[name].to(torch.int16).numpy().astype('<i2', copy=False)
              for name in fua.FIELDS]
    body = (b''.join(a.tobytes() for a in arrays) if magic == fua.MAGIC
            else fua._sparse_body(arrays))
    if magic == fua.PLANAR_MAGIC:
        body = fua._sparse_zigzag_byteplane(body, count, 3)
    packet = fua._encode_body(magic, body, 1, count, 3, (.005, .001, .001),
                              fua._reference_hash(reference))
    assert packet[:4] == magic
    directory = Path(directory)
    (directory / 'motion_001.bin').write_bytes(motion)
    (directory / 'fullattr_001.bin').write_bytes(packet)
    manifest = dict(frame=1, motion='motion_001.bin', birth=None, full_update=True,
                    full_attribute_update='fullattr_001.bin',
                    full_attribute_update_timing='after_regional_dc_before_births',
                    opacity_update_timing='before_births')
    index = directory / 'frame_001.json'
    index.write_text(json.dumps(manifest))
    payload = len(motion) + len(packet)
    summary = dict(motion=dict(bytes=len(motion)), birth=None,
                   full_attribute_update=dict(archive_bytes=len(packet)),
                   packet_bytes=payload, manifest_bytes=index.stat().st_size,
                   encoded_bytes=payload + index.stat().st_size,
                   complete_dynamic_update_bytes=payload + index.stat().st_size,
                   shared_reference_bytes=11, decoder_reference='reference_frame0.npz')
    return prior, raw, manifest, summary, packet


class FullAttributeBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_replay_full_attributes_leave_previous_state_independent(self):
        for magic in (fua.MAGIC, fua.SPARSE_MAGIC, fua.PLANAR_MAGIC):
            with self.subTest(magic=magic), tempfile.TemporaryDirectory() as directory:
                stream = Path(directory)
                prior, raw, _, _, packet = make_stream(stream, magic=magic)
                before = {name: value.clone() for name, value in raw.items()}
                _, current = decode_frame(stream, 1, prior, raw)
                for name in RAW_FIELDS:
                    self.assertTrue(torch.equal(raw[name].view(torch.int32), before[name].view(torch.int32)), name)
                    self.assertNotEqual(current[name].data_ptr(), raw[name].data_ptr(), name)
                    current[name].fill_(123)
                    self.assertTrue(torch.equal(raw[name].view(torch.int32), before[name].view(torch.int32)), name)

                damaged = bytearray(packet)
                damaged[-1] ^= 1
                (stream / 'fullattr_001.bin').write_bytes(damaged)
                with self.assertRaises(ValueError):
                    decode_frame(stream, 1, prior, raw)
                for name in RAW_FIELDS:
                    self.assertTrue(torch.equal(raw[name].view(torch.int32), before[name].view(torch.int32)), name)

    def test_actual_fua1_fua2_fua3_sizes_and_aggregate_include_manifest(self):
        for magic in (fua.MAGIC, fua.SPARSE_MAGIC, fua.PLANAR_MAGIC):
            with self.subTest(magic=magic), tempfile.TemporaryDirectory() as directory:
                stream = Path(directory)
                _, _, manifest, summary, packet = make_stream(stream, magic=magic)
                measured = benchmark._actual_frame_storage(
                    stream, 1, manifest, summary, 11, 'reference_frame0.npz')
                self.assertEqual(measured['full_attribute_update_bytes'], len(packet))
                self.assertEqual(measured['payload_bytes'], summary['packet_bytes'])
                self.assertEqual(measured['complete_update_bytes'], summary['encoded_bytes'])
                total = benchmark._summarize_sequence_storage([dict(frame=1, **measured)], 11)
                dynamic = total['dynamic_packets']
                self.assertEqual(dynamic['full_attribute_update_bytes_total'], len(packet))
                self.assertEqual(dynamic['complete_update_bytes_total'], summary['encoded_bytes'])
                self.assertEqual(total['total_reference_and_complete_update_bytes'],
                                 11 + summary['encoded_bytes'])
                bad = {**summary, 'full_attribute_update': dict(archive_bytes=len(packet) + 1)}
                with self.assertRaisesRegex(ValueError, 'storage summary'):
                    benchmark._actual_frame_storage(stream, 1, manifest, bad, 11, 'reference_frame0.npz')
                without = {**manifest, 'full_attribute_update': None}
                with self.assertRaisesRegex(ValueError, 'not declared'):
                    benchmark._actual_frame_storage(stream, 1, without, summary, 11, 'reference_frame0.npz')
                wrong_order = {**manifest, 'full_attribute_update_timing': 'before_regional_dc'}
                with self.assertRaisesRegex(ValueError, 'must follow'):
                    benchmark._actual_frame_storage(stream, 1, wrong_order, summary, 11, 'reference_frame0.npz')

    def test_timed_playback_reads_all_fua_formats_before_upload(self):
        for magic in (fua.MAGIC, fua.SPARSE_MAGIC, fua.PLANAR_MAGIC):
            with self.subTest(magic=magic), tempfile.TemporaryDirectory() as directory:
                stream = Path(directory)
                prior, raw, _, _, packet = make_stream(stream, magic=magic)
                events = []
                tick = [0]
                original_decode = fua.decode_full_attribute_delta
                def clock():
                    events.append('clock')
                    tick[0] += 1
                    return float(tick[0])
                def decode(*args, **kwargs):
                    events.append('decode_full')
                    return original_decode(*args, **kwargs)
                def install(module, values, device):
                    events.append('upload')
                    module.raw = values
                class FakeModule:
                    def forward(self, view):
                        events.append('render')
                        return {'render': self.raw['sh_dc'][0].clone()}
                module = FakeModule()
                with patch.object(benchmark, 'check_gpu_exclusivity', return_value={}), \
                        patch.object(torch.cuda, 'synchronize'), \
                        patch.object(benchmark, '_install_raw', side_effect=install), \
                        patch.object(benchmark.time, 'perf_counter', side_effect=clock), \
                        patch.object(fua, 'decode_full_attribute_delta', side_effect=decode):
                    measured = benchmark._time_playback(module, SimpleNamespace(), stream, prior, raw,
                                                       end_frame=1, rounds=1, device='cpu')
                self.assertEqual(events, ['clock', 'decode_full', 'clock', 'upload', 'render', 'clock'])
                self.assertEqual(measured['passes'][0]['frames'][0]['file_decode_seconds'], 1.)
                self.assertEqual(measured['passes'][0]['frames'][0]['end_to_end_seconds'], 2.)
                expected = original_decode(packet, {name: raw[name] for name in fua.FIELDS}, frame_index=1)
                for name in fua.FIELDS:
                    torch.testing.assert_close(measured['final_raw'][name], expected[name], rtol=0, atol=0)


STEPS = (.005, .001, .001)


def reference(count=8, rest=3):
    result = dict(logscale=torch.zeros(count, 3), sh_dc=torch.zeros(count, 1, 3),
                  sh_rest=torch.zeros(count, rest, 3))
    result['logscale'][0, 0] = -0.
    return result


def legacy_packets(ref, symbols, frame=10):
    arrays = [symbols[name].to(torch.int16).numpy().astype('<i2', copy=False) for name in fua.FIELDS]
    dense = b''.join(array.tobytes() for array in arrays)
    sparse = fua._sparse_body(arrays)
    planar = fua._sparse_zigzag_byteplane(sparse, len(ref['logscale']), ref['sh_rest'].shape[1])
    args = (frame, len(ref['logscale']), ref['sh_rest'].shape[1], STEPS, fua._reference_hash(ref))
    return [fua._encode_body(magic, body, *args) for magic, body in
            zip((fua.MAGIC, fua.SPARSE_MAGIC, fua.PLANAR_MAGIC), (dense, sparse, planar))]


def bits(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


class FUA3CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shortest_full_packet_and_legacy_compatibility(self):
        gen = torch.Generator().manual_seed(20261104)
        for count, active_mod, magnitude in ((8, 1, 32767), (4096, 127, 30),
                                             (300, 3, 20), (500, 1, 4)):
            ref = reference(count)
            symbols = {name: torch.randint(-magnitude, magnitude + 1, value.shape,
                                            generator=gen).float() for name, value in ref.items()}
            for value in symbols.values():
                value[torch.arange(count) % active_mod != 0] = 0
            for name in fua.FIELDS:
                symbols[name][0].flatten()[0] = -32767
                symbols[name][-1].flatten()[-1] = 32767
            old_dense, old_sparse, new = legacy_packets(ref, symbols)
            packet = fua.encode_full_attribute_delta(ref, symbols, frame_index=10, steps=STEPS)
            self.assertLessEqual(len(packet), min(len(old_dense), len(old_sparse)))
            self.assertEqual(packet, min((old_dense, old_sparse, new), key=len))
            results = [fua.decode_full_attribute_delta(p, ref, frame_index=10)
                       for p in (old_dense, old_sparse, new, packet)]
            for name, step in zip(fua.FIELDS, STEPS):
                expected = torch.where(symbols[name].ne(0), ref[name] + symbols[name] * step, ref[name])
                for result in results:
                    self.assertEqual(bits(result[name]), bits(expected))

    def test_empty_fields_extremes_and_signed_range(self):
        ref = reference(4, rest=0)
        zero = {name: torch.zeros_like(value) for name, value in ref.items()}
        self.assertIsNone(fua.encode_full_attribute_delta(ref, zero, frame_index=10, steps=STEPS))
        empty_body = struct.pack('<III', 0, 0, 0)
        empty = fua._encode_body(fua.PLANAR_MAGIC, empty_body, 10, 4, 0, STEPS, fua._reference_hash(ref))
        decoded = fua.decode_full_attribute_delta(empty, ref, frame_index=10)
        for name in fua.FIELDS:
            self.assertEqual(bits(decoded[name]), bits(ref[name]))
        symbols = {name: value.clone() for name, value in zero.items()}
        symbols['logscale'][0, :2] = torch.tensor([-32767., 32767.])
        for packet in legacy_packets(ref, symbols):
            out = fua.decode_full_attribute_delta(packet, ref, frame_index=10)
            self.assertEqual(bits(out['logscale']), bits(torch.where(
                symbols['logscale'].ne(0), ref['logscale'] + symbols['logscale'] * STEPS[0], ref['logscale'])))
        symbols['logscale'][0, 0] = -32768
        with self.assertRaises(ValueError):
            fua.encode_full_attribute_delta(ref, symbols, frame_index=10, steps=STEPS)
        # Historical FUA1 decoding accepted the full signed int16 domain;
        # changing the new encoder must not narrow that old read contract.
        legacy_body = b''.join(symbols[name].to(torch.int16).numpy().astype('<i2').tobytes()
                               for name in fua.FIELDS)
        legacy = fua._encode_body(fua.MAGIC, legacy_body, 10, 4, 0, STEPS, fua._reference_hash(ref))
        legacy_result = fua.decode_full_attribute_delta(legacy, ref, frame_index=10)
        self.assertEqual(bits(legacy_result['logscale']), bits(torch.where(
            symbols['logscale'].ne(0), ref['logscale'] + symbols['logscale'] * STEPS[0], ref['logscale'])))
        # FUA2 has always rejected -32768; FUA3 must not create a bypass.
        planar = struct.pack('<I', 1) + b'\x01' + bytes([255, 0, 0, 255, 0, 0]) + struct.pack('<II', 0, 0)
        packet = fua._encode_body(fua.PLANAR_MAGIC, planar, 10, 4, 0, STEPS, fua._reference_hash(ref))
        with self.assertRaises(ValueError):
            fua.decode_full_attribute_delta(packet, ref, frame_index=10)

    def test_corrupt_headers_streams_gaps_and_rows_are_rejected(self):
        ref = reference(3)
        symbols = {name: torch.zeros_like(value) for name, value in ref.items()}
        symbols['logscale'][1, 1] = -7
        good = legacy_packets(ref, symbols)[2]
        for broken in (good[:3], good[:-1], good + b'\0', b'BAD!' + good[4:],
                       good[:fua.HEADER.size] + b'not-zlib'):
            with self.subTest(length=len(broken)), self.assertRaises(ValueError):
                fua.decode_full_attribute_delta(broken, ref, frame_index=10)
        altered = {name: value.clone() for name, value in ref.items()}
        altered['sh_dc'][0, 0, 0] = 1
        with self.assertRaises(ValueError):
            fua.decode_full_attribute_delta(good, altered, frame_index=10)
        with self.assertRaises(ValueError):
            fua.decode_full_attribute_delta(good, ref, frame_index=11)
        empty = struct.pack('<I', 0)
        row = bytes([2, 0, 0, 0, 0, 0])  # zigzag(1), three low then high bytes
        bad_bodies = [struct.pack('<I', 4) + b'\1' * 4 + row * 4 + empty * 2,
                      struct.pack('<I', 1) + b'\0' + row + empty * 2,
                      struct.pack('<I', 1) + b'\4' + row + empty * 2,
                      struct.pack('<I', 1) + b'\x81\0' + row + empty * 2,
                      struct.pack('<I', 1) + b'\1' + bytes(6) + empty * 2,
                      struct.pack('<I', 1) + b'\1' + row[:-1],
                      struct.pack('<I', 1) + b'\1' + row + empty * 2 + b'\0']
        for body in bad_bodies:
            packet = fua._encode_body(fua.PLANAR_MAGIC, body, 10, 3, 3, STEPS, fua._reference_hash(ref))
            with self.subTest(body=body), self.assertRaises(ValueError):
                fua.decode_full_attribute_delta(packet, ref, frame_index=10)





class AnalyticStats:
    def __init__(self, weights):
        self.weights = weights.detach()

    def accumulate(self, image_error):
        return ((self.weights * image_error[None]).sum((1, 2)).detach(),
                self.weights.sum((1, 2)).detach())


class SmallRenderer:
    """Tiny differentiable compositor with controlled synthetic observation support."""
    collect_contributions = False
    collect_error_stats = False

    def __init__(self, module):
        self.module, self.gaussians = module, module.gaussians
        self.stats_calls = 0

    def __call__(self, view, bg, override_color=None):
        g = self.gaussians
        xyz, scale = g.get_xyz, g.get_scaling
        y, x = torch.meshgrid(torch.linspace(-.15, .65, 8), torch.linspace(-.15, .65, 8), indexing='ij')
        d = ((x[None]-xyz[:, 0, None, None])/scale[:, 0, None, None]).square() + ((y[None]-xyz[:, 1, None, None])/scale[:, 1, None, None]).square()
        alpha = g.get_opacity[:, 0, None, None] * torch.exp(-.5*d)
        trans = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1-alpha), 0), 0)[:-1]
        weights = alpha*trans
        colour = .5 + .28209479177387814*g.get_features[:, 0] if override_color is None else override_color
        image = torch.einsum('nhw,nc->chw', weights, colour)
        # Depend explicitly on orientation and higher SH so those six-field
        # gradients remain observable in the otherwise tiny toy compositor.
        image = image + .01*(g.get_rotation[:, 1:].sum() + g.get_features[:, 1:].sum())
        count = len(xyz)
        package = dict(render=image, radii=torch.ones(count),
            visibility_filter=torch.ones(count, dtype=torch.bool),
            motion_contribution_filter=weights.flatten(1).sum(1).gt(0))
        if self.collect_error_stats:
            self.stats_calls += 1
            owners = self.module.motion.owner
            # Hide all but one owner from the measurement. This exercises
            # both triggered and unselected rows using real compositor weights.
            support = owners.eq(owners.min()).to(weights)[:, None, None]
            package['regional_error_stats'] = AnalyticStats(weights * support)
        return package


def fixture(directory, *, threshold=.03):
    m, views = controller_fixture(directory, shared_colour=False)
    m.config.birth.population_loss_weight = 0.
    m.config.birth.storage_loss_enabled = False
    m.config.birth.regional_color_l1_threshold = threshold
    m.config.birth.regional_color_min_support = .001
    m.config.birth.regional_color_min_views = 2
    m.config.birth.keep_incumbent_fixed_after_motion = False
    m.config.birth.keep_incumbent_alpha_after_motion = False
    m.config.rigid_tree.spatial_gate.final_joint_refine_steps = 0
    m.config.full_update = FullUpdateConfig(enabled=True, regional_repair=True,
                                             storage_loss_enabled=True, per_attribute_l1_weight=0.)
    m.full_update_active = True
    # Preserve an especially low-alpha point: a no-deletion run must retain it.
    with torch.no_grad():
        m.gaussians._opacity[-1] = -20.
    m.motion = FullUpdateSpatialGatedMotion(replace(m.config.rigid_tree, rotation_updates=True))
    m.motion.regional_repair = True
    m.motion.original_rotation_updates = False
    m.motion.setup(m.gaussians._xyz, m.gaussians._rotation, 1., frame_index=1)
    m.opacity_update.setup(m.gaussians._opacity)
    m._early_attach_opacity()
    for name in ('full_update_prepare_step', 'full_update_after_step',
                 'full_update_finish_pose_attributes', 'full_update_reference'):
        setattr(m, name, MethodType(getattr(Module, name), m))
    m.full_attribute_update = FullAttributeUpdate(m.config.full_update)
    args = m.config.gaussian
    m.full_attribute_update.setup(m.full_update_reference(), m.motion.optimizer,
                                  (args.scaling_lr, args.feature_lr, args.feature_lr/20.))
    m._plan_regional_attribute_state = None
    m._plan_error_owner = None
    m.renderer = SmallRenderer(m)
    m.finalize_frame = MethodType(Module.finalize_frame, m)
    checkpoint_harness(m)
    runner = SimpleNamespace(module=m,
        config=SimpleNamespace(motion_views_per_step=4, birth_views_per_step=4),
        _views_per_step=lambda step: 4)
    return m, runner


def fit_step(module, runner, step):
    module.trainer.global_step = step
    module.pre_training_step()
    Trainer._fit_views(runner, None, step)




# ----- Regional DC -----




# ----- Image loss -----


class ImageLossTests(unittest.TestCase):
    def test_rmse_has_finite_zero_and_emphasizes_larger_pixel_errors(self):
        from pipeline.module import stabilized_rmse
        target = torch.zeros(1, 1, 2)
        exact = target.clone().requires_grad_()
        loss = stabilized_rmse(exact, target)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(exact.grad).all())
        self.assertTrue(torch.equal(exact.grad, torch.zeros_like(exact)))
        image = torch.tensor([[[.4, .02]]], requires_grad=True)
        loss = .75 * (image - target).abs().mean() + .25 * stabilized_rmse(image, target)
        loss.backward()
        self.assertGreater(float(image.grad[0, 0, 0]), float(image.grad[0, 0, 1]))
        self.assertGreater(float(image.grad[0, 0, 1]), 0.)


if __name__ == "__main__":
    unittest.main()

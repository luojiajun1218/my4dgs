"""CPU checks for full-attribute training, codec, benchmark and replay."""
import copy
import hashlib
import io
import json
import shutil
import struct
import tempfile
import unittest
import warnings
import zlib
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from pipeline.frame_plan import gate_owner_error_decisions
from pipeline.model import full_attribute_update as fua
from pipeline.model.birth_quantization import RAW_FIELDS, model_raw, reference_digest
from pipeline.model.full_attribute_update import (FIELDS, HEADER, MAGIC, SPARSE_MAGIC,
    FullAttributeUpdate, FullUpdateConfig, FullUpdateSpatialGatedMotion,
    _encode_body, _reference_hash, decode_full_attribute_delta, encode_full_attribute_delta)
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.rigid_tree_codec import DecodedState, encode_motion
from pipeline.model.sparse_features import (FeatureDCUpdateConfig, QuantizedFeatureDCUpdate,
    REGIONAL_MAGIC, _regional_digest, decode_feature_dc_delta)
from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from pipeline.module import Module
from pipeline.trainer import Trainer
from scripts import benchmark_rigid_residual_tree as benchmark, repack_fua3
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
    def test_legacy_repair_policy_config_validation(self):
        self.assertFalse(FullUpdateConfig().repair_only_after_motion)
        for invalid in (0, 1, 'true', None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                FullUpdateConfig(repair_only_after_motion=invalid).validate()

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
DIAGNOSTICS = Path('/mnt/SSD500GB/my4dgs_diagnostics')


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
        self.assertEqual(repack_fua3.repack_packet(empty, frame_index=10), empty)
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
        self.assertEqual(repack_fua3.repack_packet(legacy, frame_index=10), legacy)
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

    def test_four_real_packets_preserve_metadata_symbols_and_full_packet(self):
        for scene in ('sear', 'cook'):
            analysis = DIAGNOSTICS / f'{scene}_two_stage_30_analysis_20261104/analysis.json'
            if not analysis.exists():
                self.skipTest('Local formal-run analysis not present')
            run = Path(json.loads(analysis.read_text())['run'])
            for frame in (10, 20):
                with self.subTest(scene=scene, frame=frame):
                    packet = (run / 'frame_storage/rigid_tree' / f'fullattr_{frame:03d}.bin').read_bytes()
                    old = fua.HEADER.unpack_from(packet)
                    self.assertEqual(old[0], fua.SPARSE_MAGIC)
                    converted = repack_fua3.repack_packet(packet, frame_index=frame)
                    new = fua.HEADER.unpack_from(converted)
                    self.assertEqual(new[0], fua.PLANAR_MAGIC)
                    self.assertLess(len(converted), len(packet))
                    self.assertEqual(old[1:-1], new[1:-1])  # includes frame, shape, steps, len, prior hash
                    sparse = zlib.decompress(packet[fua.HEADER.size:])
                    planar = zlib.decompress(converted[fua.HEADER.size:])
                    self.assertEqual(fua._sparse_zigzag_byteplane(planar, old[2], old[3], inverse=True), sparse)
                    self.assertEqual(fua._sparse_zigzag_byteplane(sparse, old[2], old[3]), planar)
                    # Rebind both to the same synthetic reference solely to run
                    # the actual materializing decoder; packet prior remains intact above.
                    ref = reference(old[2], old[3])
                    prior = fua._reference_hash(ref)
                    re_old = fua._encode_body(fua.SPARSE_MAGIC, sparse, frame, old[2], old[3], old[4:7], prior)
                    re_new = fua._encode_body(fua.PLANAR_MAGIC, planar, frame, old[2], old[3], old[4:7], prior)
                    a = fua.decode_full_attribute_delta(re_old, ref, frame_index=frame)
                    b = fua.decode_full_attribute_delta(re_new, ref, frame_index=frame)
                    for name in fua.FIELDS:
                        self.assertEqual(bits(a[name]), bits(b[name]))
                    self.assertEqual(packet[:fua.HEADER.size] + zlib.compress(sparse, 6), packet)
                    print(f'{scene} {frame}: FUA2={len(packet)} FUA3={len(converted)}', flush=True)

    def test_real_checkpoints_decode_old_and_fua3_to_identical_six_fields(self):
        for scene in ('sear', 'cook'):
            analysis = DIAGNOSTICS / f'{scene}_two_stage_30_analysis_20261104/analysis.json'
            if not analysis.exists():
                self.skipTest('Local formal-run analysis not present')
            run = Path(json.loads(analysis.read_text())['run'])
            stream = run / 'frame_storage/rigid_tree'
            for frame in (10, 20):
                with self.subTest(scene=scene, frame=frame):
                    ckpt_path = run / f'rigid-ckpt-{frame - 1}-300.pth'
                    if not ckpt_path.exists():
                        self.skipTest('Local predecessor checkpoint not present')
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    g = ckpt['gaussians']
                    raw = dict(xyz=g[1], sh_dc=g[2], sh_rest=g[3],
                               logscale=g[4], rotation=g[5], opacity=g[6])
                    state = DecodedState.restore(ckpt['rigid_tree_motion']['codec_previous_state'])
                    prev = json.loads((stream / f'frame_{frame - 1:03d}.json').read_text())
                    manifest = json.loads((stream / f'frame_{frame:03d}.json').read_text())
                    self.assertEqual(state.frame_index, frame - 1)
                    self.assertEqual(state.digest().hex(), prev['decoded_state_sha256'])
                    self.assertEqual(reference_digest(raw), prev['raw_state_sha256'])
                    old_state, old_raw = decode_frame(stream, frame, state, raw)
                    original_decode = fua.decode_full_attribute_delta
                    seen = []
                    def decode_repacked(packet, actual_reference, *, frame_index):
                        converted = repack_fua3.repack_packet(packet, frame_index=frame_index)
                        self.assertEqual(converted[:4], fua.PLANAR_MAGIC)
                        seen.append(True)
                        return original_decode(converted, actual_reference, frame_index=frame_index)
                    with patch.object(fua, 'decode_full_attribute_delta', side_effect=decode_repacked):
                        new_state, new_raw = decode_frame(stream, frame, state, raw)
                    self.assertEqual(seen, [True])
                    repack_fua3._check_frame(manifest, old_state, old_raw, new_state, new_raw)
                    self.assertEqual(len(new_state.survivor_rows), manifest['old_count'])
                    self.assertEqual(len(new_state.xyz), manifest['total_count'])

    def test_repack_tool_tiny_complete_run_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            stream = source / 'frame_storage/rigid_tree'
            stream.mkdir(parents=True)
            (source / 'frame_summary').mkdir()
            config = Path(__file__).resolve().parents[1] / 'config/sear_two_stage_region_fullattr_k10_30_20261104.yaml'
            shutil.copyfile(config, source / 'config.yaml')
            prior, raw, manifest, summary, _ = make_stream(stream, magic=fua.SPARSE_MAGIC)
            count = len(prior.xyz)
            ref = {name: raw[name] for name in fua.FIELDS}
            gen = torch.Generator().manual_seed(44)
            symbols = {name: torch.randint(-4, 5, v.shape, generator=gen).float() for name, v in ref.items()}
            packet = legacy_packets(ref, symbols, frame=1)[1]
            self.assertEqual(repack_fua3.repack_packet(packet, frame_index=1)[:4], fua.PLANAR_MAGIC)
            (stream / 'fullattr_001.bin').write_bytes(packet)
            ref_path = stream / 'reference_frame0.npz'
            np.savez(ref_path, **{name: raw[name].numpy() for name in raw},
                     owner=prior.owner.numpy(), parents=prior.parents.numpy(), frame_index=0)
            state, expected = decode_frame(stream, 1, prior, raw)
            manifest.update(reference='frame_storage/rigid_tree/reference_frame0.npz',
                            decoded_state_sha256=state.digest().hex(),
                            raw_state_sha256=reference_digest(expected), total_count=count)
            manifest_path = stream / 'frame_001.json'
            manifest_path.write_text(json.dumps(manifest))
            summary['full_attribute_update'] = dict(archive_bytes=len(packet), format='FUA2', path=str(stream/'fullattr_001.bin'))
            summary['full_attribute_rate_diagnostics'] = {'actual_packet_bytes': len(packet), 'loss': .123}
            payload = (stream / 'motion_001.bin').stat().st_size + len(packet)
            summary.update(packet_bytes=payload, manifest_bytes=manifest_path.stat().st_size,
                           encoded_bytes=payload + manifest_path.stat().st_size,
                           complete_dynamic_update_bytes=payload + manifest_path.stat().st_size,
                           shared_reference_bytes=ref_path.stat().st_size,
                           decoder_reference=manifest['reference'])
            (source / 'frame_summary/frame_001.json').write_text(json.dumps(summary))
            before = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).digest()
                      for p in source.rglob('*') if p.is_file()}
            training_source = Path(__file__).resolve().parents[1]
            with self.assertRaises(ValueError):
                repack_fua3.repack_run(source, source / 'must_not_exist', 1, training_source=training_source)
            self.assertFalse((source / 'must_not_exist').exists())
            output = root / 'artifact'
            actual_decode = repack_fua3.decode_frame
            def empty_births_as_none(*args):
                state, values = actual_decode(*args)
                self.assertEqual(len(state.xyz), len(state.survivor_rows))
                state.birth_sources = None
                return state, values
            # This packet has zero births. Exercise the replay API's other
            # valid representation of that condition: birth_sources=None.
            with patch.object(repack_fua3, 'decode_frame', side_effect=empty_births_as_none):
                report = repack_fua3.repack_run(source, output, 1, training_source=training_source)
            self.assertTrue(report['verified'])
            self.assertTrue(report['frames'][0]['bit_exact_six_fields_and_sources'])
            self.assertEqual(report['frames'][0]['birth_count'], 0)
            self.assertEqual(report['frames'][0]['repacked_format'], 'FUA3')
            self.assertLess(report['frames'][0]['repacked_fua_bytes'], report['frames'][0]['original_fua_bytes'])
            self.assertFalse((output / 'REPACK_INCOMPLETE').exists())
            self.assertEqual((source / 'config.yaml').read_bytes(), (output / 'source_config.yaml').read_bytes())
            source_config = OmegaConf.to_container(OmegaConf.load(source / 'config.yaml'), resolve=False)
            new_config = OmegaConf.to_container(OmegaConf.load(output / 'config.yaml'), resolve=False)
            self.assertEqual(new_config.pop('experiment_name'), source_config.pop('experiment_name') + '_fua3_repack_20261104')
            self.assertEqual(new_config, source_config)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                benchmark_config = benchmark._load_config(output)
            self.assertTrue(benchmark_config.module.full_update.enabled)
            self.assertEqual(benchmark_config.module.full_update.interval, 10)
            self.assertEqual(report['new_training_steps'], 0)
            self.assertFalse(report['psnr_retested'])
            self.assertEqual(report['source_training_config_sha256'],
                             hashlib.sha256((source / 'config.yaml').read_bytes()).hexdigest())
            self.assertEqual(report['repack_script_sha256'],
                             hashlib.sha256(Path(repack_fua3.__file__).read_bytes()).hexdigest())
            artifact_summary = json.loads((output / 'frame_summary/frame_001.json').read_text())
            self.assertNotIn('full_attribute_rate_diagnostics', artifact_summary)
            self.assertEqual(artifact_summary['source_full_attribute_rate_diagnostics'],
                             summary['full_attribute_rate_diagnostics'])
            self.assertFalse(artifact_summary['lossless_repack_provenance']['psnr_retested'])
            self.assertEqual(artifact_summary['lossless_repack_provenance']['repacked_fua_bytes'],
                             report['frames'][0]['repacked_fua_bytes'])
            after = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).digest()
                     for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            with self.assertRaises(ValueError):
                repack_fua3.repack_run(source, output, 1, training_source=training_source)
            interrupted = root / 'interrupted'
            with patch.object(repack_fua3, '_check_frame', side_effect=ValueError('test failed validation')):
                with self.assertRaises(ValueError):
                    repack_fua3.repack_run(source, interrupted, 1, training_source=training_source)
            self.assertTrue((interrupted / 'REPACK_INCOMPLETE').is_file())
            self.assertFalse((interrupted / 'frame_summary/frame_001.json').exists())
            self.assertFalse((interrupted / 'fua3_repack_report.json').exists())


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


class GateOwnerRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reliability_equal_view_weight_and_exact_threshold(self):
        owner = torch.tensor([1, 1, 4, 7])
        den = torch.tensor([[0., 32., 0., 0., 320., 0., 0., 31.],
                            [0., 64., 0., 0., 32., 0., 0., 32.],
                            [0., 0., 0., 0., 1., 0., 0., 0.],
                            [0., 0., 0., 0., 1., 0., 0., 0.]])
        num = den * torch.tensor([[0., .01, 0., 0., .07, 0., 0., .9],
                                  [0., .05, 0., 0., .01, 0., 0., .9],
                                  [0., 1., 0., 0., 1., 0., 0., .9],
                                  [0., 1., 0., 0., 1., 0., 0., .9]])
        result = gate_owner_error_decisions(owner, num, den, threshold=.03, min_support=32., min_views=2)
        self.assertEqual(result['region_id'].tolist(), [1, 4, 7])
        self.assertAlmostEqual(float(result['region_score'][0]), .03, places=7)
        self.assertAlmostEqual(float(result['region_score'][1]), .04, places=7)
        self.assertTrue(torch.isnan(result['region_score'][2]))
        # Use a margin for the equality check: float32 .01/.05 representation.
        exact = gate_owner_error_decisions(owner, num.double(), den.double(), threshold=.031,
                                            min_support=32., min_views=2)
        self.assertEqual(exact['eligible'].tolist(), [False, False, True, False])

    def test_six_fields_no_deletion_checkpoint_and_exact_packets(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            m, runner = fixture(directory)
            prior, initial = copy.deepcopy(m.motion.frame_reference_state), model_raw(m.gaussians)
            worker = SimpleNamespace(output_dir=Path(directory), module=m, datamodule=m.datamodule)
            Trainer._write_reference(worker, 0, prior)
            m.trainer.decoder_reference_path = worker.decoder_reference_path
            clones = {}
            originals = None
            region_snapshot = None
            for step in range(1, 301):
                fit_step(m, runner, step)
                if step in (201, 250, 300):
                    factor = m.config.rigid_tree.motion_lr_final_ratio ** ((step-201)/99.)
                    base = dict(logscale=m.config.gaussian.scaling_lr,
                                sh_dc=m.config.gaussian.feature_lr,
                                sh_rest=m.config.gaussian.feature_lr/20.)
                    for group in m.motion.optimizer.param_groups:
                        name = group.get('name', '')
                        if name.startswith('early_full_'):
                            self.assertAlmostEqual(group['lr'], base[name[len('early_full_'):]]*factor)
                if step == 199:
                    with torch.no_grad():
                        delta = m.opacity_update.delta
                        delta[-1] = 0
                        for value in m.motion.optimizer.state[delta].values():
                            if torch.is_tensor(value) and value.shape == delta.shape:
                                value[-1] = 0
                if step < 200:
                    self.assertTrue(all(not p.requires_grad for p in m.full_attribute_update.parameters.values()))
                    self.assertTrue(all(torch.count_nonzero(p) == 0 for p in m.full_attribute_update.parameters.values()))
                    self.assertFalse(m.motion.gs_rotation.requires_grad)
                if step == 200:
                    state = m._plan_regional_attribute_state
                    self.assertGreater(int(state['eligible'].sum()), 0)
                    self.assertGreater(int((~state['eligible']).sum()), 0)
                    self.assertFalse(bool(state['eligible'][-1]))
                    self.assertEqual(m.renderer.stats_calls, 4)
                    originals = {k: v.detach().clone() for k, v in m._regional_attribute_parameters().items()}
                    raw_200 = {k: v.detach().cpu().clone() for k, v in m._adaptive_reference().items()}
                    region_snapshot = {name: (getattr(m.motion, name).detach().clone(),
                        copy.deepcopy(m.motion.optimizer.state.get(getattr(m.motion, name), {})))
                        for name in ('gate_logits', 'region_translation', 'region_rotation')}
                if 201 <= step <= 300 and step != 300:
                    self.assertEqual(len(m.gaussians._xyz), len(initial['xyz']))
                    self.assertEqual(len(m.births._xyz), 0)
                    state = m._plan_regional_attribute_state
                    frozen = ~state['eligible']
                    for name, param in m._regional_attribute_parameters().items():
                        self.assertTrue(torch.equal(param.detach()[frozen], originals[name][frozen]), (step, name))
                    for name, (value, adam) in region_snapshot.items():
                        p = getattr(m.motion, name)
                        self.assertTrue(torch.equal(p, value), (step, name))
                        self.assertEqual(m.motion.optimizer.state.get(p, {}).keys(), adam.keys())
                        for k in adam:
                            if torch.is_tensor(adam[k]):
                                self.assertTrue(torch.equal(m.motion.optimizer.state[p][k], adam[k]), (step, name, k))
                if step in (200, 250):
                    packet = dict(global_step=step, current_frame=1)
                    Module.on_save_checkpoint(m, packet)
                    clone, clone_runner = fixture(directory)
                    clone.trainer.decoder_reference_path = worker.decoder_reference_path
                    Module.on_load_checkpoint(clone, copy.deepcopy(packet))
                    clones[step + 1] = (clone, clone_runner)
                if step in clones:
                    clone, clone_runner = clones[step]
                    fit_step(clone, clone_runner, step)
                    for name, parameter in m._regional_attribute_parameters().items():
                        self.assertTrue(torch.equal(parameter, clone._regional_attribute_parameters()[name]), (step, name))
                        for key, value in m.motion.optimizer.state.get(parameter, {}).items():
                            actual = clone.motion.optimizer.state.get(clone._regional_attribute_parameters()[name], {}).get(key)
                            if torch.is_tensor(value): self.assertTrue(torch.equal(value, actual), (step, name, key))
                    current_groups = {group.get('name', f'motion_{i}'): group
                                      for i, group in enumerate(m.motion.optimizer.param_groups)}
                    restored_groups = {group.get('name', f'motion_{i}'): group
                                       for i, group in enumerate(clone.motion.optimizer.param_groups)}
                    self.assertEqual(current_groups.keys(), restored_groups.keys())
                    for name, group in current_groups.items():
                        self.assertEqual(group['lr'], restored_groups[name]['lr'], (step, name))
            self.assertTrue(m.frame_finalized)
            self.assertEqual(m.frame_report['birth_count'], 0)
            self.assertEqual(m.frame_report['retired_count'], 0)
            self.assertEqual(len(m.gaussians._xyz), len(initial['xyz']))
            self.assertLess(float(m.gaussians._opacity[-1].sigmoid()), .01)
            decoded, raw = decode_frame(Path(directory)/'frame_storage'/'rigid_tree', 1, prior, initial)
            self.assertEqual(decoded.digest(), m.motion.codec_previous_state.digest())
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(raw[name], model_raw(m.gaussians)[name].cpu()), name)
                self.assertFalse(torch.equal(raw[name][state['eligible'].cpu()],
                                             raw_200[name][state['eligible'].cpu()]), name)
            manifest = json.loads((Path(directory)/'frame_storage'/'rigid_tree'/'frame_001.json').read_text())
            self.assertEqual(manifest['birth_events'], [])
            self.assertEqual(manifest['training_step_accounting']['cleanup_steps'], [])
            self.assertTrue(manifest['no_old_deletions'])
            self.assertAlmostEqual(manifest['regional_attribute_repair']['learning_rates_step_201']['logscale'],
                                   m.config.gaussian.scaling_lr)
            self.assertAlmostEqual(manifest['regional_attribute_repair']['learning_rates_step_300']['logscale'],
                                   m.config.gaussian.scaling_lr * m.config.rigid_tree.motion_lr_final_ratio)
            audit = m.birth_control['regional_attribute_repair']
            for check in audit['image_gradient_checks']:
                for name, row in check['fields'].items():
                    self.assertTrue(row['has_gradient'], (check['step'], name))
                    self.assertGreater(row['selected_norm'], 0., (check['step'], name))
            with np.load(Path(directory)/audit['diagnostics']) as evidence:
                np.testing.assert_array_equal(evidence['eligible_rows'], state['eligible'].numpy())
                self.assertEqual(tuple(evidence['point_numerator'].shape), (4, len(initial['xyz'])))
                self.assertTrue(np.all(evidence['unassigned_l1_sum'] >= 0))
            self.assertEqual(m.renderer.stats_calls, 4)

    def test_no_trigger_never_creates_birth_or_removes_old_rows(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            m, runner = fixture(directory, threshold=100.)
            prior = copy.deepcopy(m.motion.frame_reference_state)
            worker = SimpleNamespace(output_dir=Path(directory), module=m, datamodule=m.datamodule)
            Trainer._write_reference(worker, 0, prior)
            m.trainer.decoder_reference_path = worker.decoder_reference_path
            initial_count = len(m.gaussians._xyz)
            for step in range(1, 301):
                fit_step(m, runner, step)
                if step == 200:
                    self.assertEqual(int(m._plan_regional_attribute_state['eligible'].sum()), 0)
                    saved = {name: p.detach().clone() for name, p in m._regional_attribute_parameters().items()}
                if step == 299:
                    for name, parameter in m._regional_attribute_parameters().items():
                        self.assertTrue(torch.equal(parameter, saved[name]), name)
            self.assertEqual(m.renderer.stats_calls, 4)
            self.assertEqual(len(m.gaussians._xyz), initial_count)
            self.assertEqual(m.frame_report['birth_count'], 0)
            self.assertEqual(m.frame_report['retired_count'], 0)

    def test_historical_regional_colour_decodes_but_training_is_rejected(self):
        cfg = FeatureDCUpdateConfig(enabled=True, sharing='spatial_region', gate_joint=True)
        with self.assertRaisesRegex(ValueError, 'retired'):
            QuantizedFeatureDCUpdate(cfg)
        reference = torch.zeros(3, 1, 3)
        owner = torch.tensor([1, 2, 1])
        scalar = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='logit', step=.01))
        scalar.setup(torch.zeros(9, 1))
        with torch.no_grad():
            scalar.delta[3] = .02
            scalar.delta[8] = -.03
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.bin'
            scalar.export(path, 1)
            packet = REGIONAL_MAGIC + _regional_digest(reference, owner, 3) + path.read_bytes()
            result = decode_feature_dc_delta(packet, reference, frame_index=1, owner=owner, node_count=3)
        expected = torch.tensor([[[.02, 0., 0.]], [[0., 0., -.03]], [[.02, 0., 0.]]])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()

"""Motion training state/codec checks, without scene data or a renderer."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

M = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_motion')


def test_accumulation_restore_and_final_export():
    torch.manual_seed(723)
    xyz = torch.randn(31, 3, dtype=torch.float64)
    config = M.RDTreeConfig(max_depth=4, min_leaf_size=2, rho_xyz=1., rho_rotation=1.,
                            quant_xyz=.01, quant_rotation=.01, accumulation_steps=4,
                            step_fraction=.2, normalized_rate=.001)
    model = M.RDTreeMotion(config)
    model.setup(xyz)
    gradients = [torch.randn(31, 6, dtype=torch.float64) * .1 for _ in range(8)]
    resumed = None
    for index, gradient in enumerate(gradients):
        if index == 2:
            resumed = M.RDTreeMotion(config)
            resumed.restore(model.capture())
            assert resumed.pending_count == 2
        before = model.latent.detach().clone()
        for candidate in ([model, resumed] if resumed is not None else [model]):
            loss = (candidate.field() * gradient).sum() + candidate.envelope_loss()
            loss.backward()
            candidate.optimizer.step()
            candidate.optimizer.zero_grad()
        if (index + 1) % 4:
            torch.testing.assert_close(model.latent, before, rtol=0, atol=0)
        if resumed is not None:
            torch.testing.assert_close(model.latent, resumed.latent, rtol=0, atol=0)
            torch.testing.assert_close(model.pending_gradient, resumed.pending_gradient, rtol=0, atol=0)
    assert model.updates == 2 and model.dirty
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.rdp'
        stats = model.export(path)
        assert not model.dirty
        protocol = json.loads(path.with_suffix('.rdp.json').read_text())
        field = M.decode_tree_motion(path.read_bytes(), xyz.numpy(), protocol)
        torch.testing.assert_close(torch.as_tensor(field), model.field().detach(), rtol=0, atol=0)
        assert stats['motion_bytes'] == path.stat().st_size + path.with_suffix('.rdp.json').stat().st_size
        assert stats['export_verification'] == 'independent_tree_rebuild'
        model.settings.rebuild_tree_on_export = False
        fast_path = Path(directory) / 'fast.rdp'
        fast_stats = model.export(fast_path)
        assert fast_stats['export_verification'] == 'existing_candidate_tree'
        assert fast_path.read_bytes() == path.read_bytes()
        try:
            model.export(path)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing packet was overwritten')


def test_accumulated_gradient_matches_batch_mean():
    torch.manual_seed(83)
    xyz = torch.randn(12, 3, dtype=torch.float64)
    config = M.RDTreeConfig(max_depth=3, min_leaf_size=2, rho_xyz=2., rho_rotation=3.,
                            accumulation_steps=4, step_fraction=.2)
    model = M.RDTreeMotion(config)
    model.setup(xyz)
    gradients = torch.randn(4, 12, 6, dtype=torch.float64)
    for gradient in gradients:
        model.field().backward(gradient)
        model.optimizer.step()
        model.optimizer.zero_grad()
    expected = -.2 / model.rho * gradients.mean(0)
    torch.testing.assert_close(model.latent, expected, rtol=1e-12, atol=1e-12)
    assert model.projection_calls == 1  # No repeated DP within the batch.


def run_all():
    test_accumulation_restore_and_final_export()
    test_accumulated_gradient_matches_batch_mean()
    test_invalid_settings_and_partial_export()
    test_fixed_cut_restore_and_legacy_config()
    test_structure_warmup_switch_and_resume()


def test_structure_warmup_switch_and_resume():
    config = M.RDTreeConfig(max_depth=2, min_leaf_size=1, accumulation_steps=1,
                            structure_warmup_updates=2)
    model = M.RDTreeMotion(config)
    model.setup(torch.arange(12, dtype=torch.float64).reshape(4, 3))
    assert model.effective_fixed_cut_depth == 2
    assert len(model.project().split_nodes) == 3
    for update in range(2):
        (model.field().sum() * 0 + model.envelope_loss()).backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        resumed = M.RDTreeMotion(config)
        resumed.restore(model.capture())
        assert resumed.capture()['packet'] == model.capture()['packet']
        assert resumed.effective_fixed_cut_depth == model.effective_fixed_cut_depth
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'warmup.rdp'
            if update == 0:
                assert len(model.project().split_nodes) == 3
                try:
                    model.export(path)
                except RuntimeError:
                    assert not path.exists()
                else:
                    raise AssertionError('Provisional fixed-cut state exported')
            else:
                assert model.effective_fixed_cut_depth is None
                assert not model.project().split_nodes
                model.export(path)
                assert path.read_bytes() == model.capture()['packet']
    for value in (-1, 1.5, True):
        try:
            M.RDTreeMotion(M.RDTreeConfig(structure_warmup_updates=value))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid structure warmup accepted')
    invalid_configs = (
        M.RDTreeConfig(structure_warmup_updates=1, fixed_cut_depth=2),
        SimpleNamespace(num_stage1_steps=4, rd_tree=M.RDTreeConfig(
            accumulation_steps=4, structure_warmup_updates=2)),
    )
    for invalid_config in invalid_configs:
        try:
            M.RDTreeMotion(invalid_config)
        except ValueError:
            pass
        else:
            raise AssertionError('Conflicting or over-budget warmup accepted')


def test_fixed_cut_restore_and_legacy_config():
    xyz = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    for depth in (None, 0, 1, 4):
        config = M.RDTreeConfig(max_depth=3, min_leaf_size=1, fixed_cut_depth=depth)
        model = M.RDTreeMotion(config)
        model.setup(xyz)
        state = model.capture()
        if depth is None:
            # Old checkpoints omit the new optional selector.
            del state['config']['fixed_cut_depth']
            del state['config']['structure_warmup_updates']
        resumed = M.RDTreeMotion(config)
        resumed.restore(state)
        assert resumed.capture()['packet'] == state['packet']
        assert len(model.project().split_nodes) == {None: 0, 0: 0, 1: 1, 4: 4}[depth]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixed.rdp'
            model.export(path)
            protocol = json.loads(path.with_suffix('.rdp.json').read_text())
            decoded = M.decode_tree_motion(path.read_bytes(), xyz.numpy(), protocol)
            torch.testing.assert_close(torch.from_numpy(decoded), model.field().detach(),
                                       rtol=0, atol=0)
        wrong = M.RDTreeMotion(M.RDTreeConfig(max_depth=3, min_leaf_size=1,
                                             fixed_cut_depth=2))
        try:
            wrong.restore(state)
        except ValueError:
            pass
        else:
            raise AssertionError('Different fixed cut restored silently')


def test_invalid_settings_and_partial_export():
    for config in (M.RDTreeConfig(accumulation_steps=0), M.RDTreeConfig(rho_xyz=0.)):
        try:
            M.RDTreeMotion(config)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid optimization configuration was accepted')
    model = M.RDTreeMotion(M.RDTreeConfig(max_depth=1, min_leaf_size=1))
    model.setup(torch.zeros(4, 3, dtype=torch.float64))
    model.field().sum().backward()
    model.optimizer.step()
    model.optimizer.zero_grad()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'unfinished.rdp'
        try:
            model.export(path)
        except RuntimeError:
            assert not path.exists()
        else:
            raise AssertionError('Partial accumulated batch was exported as a completed frame')

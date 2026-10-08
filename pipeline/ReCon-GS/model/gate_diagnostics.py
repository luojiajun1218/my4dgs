"""Read-only gate traces and isolated stop/refine counterfactual fits.

Candidates change the sampled node's shared motion and its reachable descendants.
Ancestors, sibling regions, all GS attributes, births and tree topology stay fixed.
The fit uses image loss; the comparison uses actual encoded bytes after fitting.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import copy
import json
import math
from pathlib import Path
import random
import time
from typing import List, Optional

import numpy as np
import torch

from .binary_refinement import _conditional_transform
from .node_motion import node_conditional_transform
from .storage_aware_binary import CODEC_HEADER_BYTES, StorageAwareBinaryMotion


@dataclass
class GateDiagnosticsConfig:
    enabled: bool = False
    frames: List[int] = field(default_factory=list)
    trace_steps: List[int] = field(default_factory=list)
    probe_steps: List[int] = field(default_factory=list)
    nodes_per_state: int = 1  # Per level and reachable-open/closed/unreachable group.
    seed: int = 0
    camera_names: List[str] = field(default_factory=list)
    fit_steps: int = 20
    fit_lr_scale: float = 1.0
    precision_policies: List[str] = field(default_factory=lambda: ['fixed'])
    fixed_precision_bits: Optional[int] = None  # None preserves each node's selection.
    candidate_mode: str = 'stop_refine'
    stop_parent_precision_bits: List[int] = field(default_factory=list)
    refine_parent_precision_bits: List[int] = field(default_factory=list)
    refine_descendant_precision_bits: List[int] = field(default_factory=list)
    descendant_precision_scope: str = 'reachable_descendants'
    include_original_precision_candidates: bool = True
    evaluation_camera_names: List[str] = field(default_factory=list)
    view_sequence: str = 'round_robin'
    save_probe_snapshots: bool = False
    record_component_gradients: bool = False

    def __post_init__(self):
        # libgs.to_structured may leave OmegaConf ListConfig fields in dataclasses.
        # Keep this public config JSON-safe, including checkpoint metadata via asdict.
        for name in ('frames', 'trace_steps', 'probe_steps', 'camera_names', 'precision_policies',
                     'stop_parent_precision_bits', 'refine_parent_precision_bits',
                     'refine_descendant_precision_bits', 'evaluation_camera_names'):
            setattr(self, name, list(getattr(self, name)))

    def validate(self, motion_config=None):
        if self.candidate_mode not in {'stop_refine', 'joint_precision'}:
            raise ValueError('candidate_mode must be stop_refine or joint_precision')
        if self.descendant_precision_scope not in {'reachable_descendants', 'immediate_children'}:
            raise ValueError('Invalid descendant_precision_scope')
        if self.view_sequence not in {'round_robin', 'seeded_balanced_cycle'}:
            raise ValueError('Invalid diagnostic view_sequence')
        for name in ('include_original_precision_candidates', 'save_probe_snapshots', 'record_component_gradients'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be bool')
        if (len(set(self.evaluation_camera_names)) != len(self.evaluation_camera_names) or
                any(not isinstance(name, str) or not name for name in self.evaluation_camera_names) or
                set(self.camera_names) & set(self.evaluation_camera_names)):
            raise ValueError('Fitting/check camera names must be unique and disjoint')
        if 'cam00.png' in self.camera_names + self.evaluation_camera_names:
            raise ValueError('cam00 is reserved for the original final evaluation')
        if self.candidate_mode == 'joint_precision':
            if self.precision_policies != ['fixed'] or self.fixed_precision_bits is not None:
                raise ValueError('Joint precision comparisons use explicit fixed parent/descendant choices')
            for name in ('stop_parent_precision_bits', 'refine_parent_precision_bits', 'refine_descendant_precision_bits'):
                values = getattr(self, name)
                if (not values or len(set(values)) != len(values) or
                        any(isinstance(value, bool) or not isinstance(value, int) or
                            (motion_config is not None and value not in motion_config.precision_bits) for value in values)):
                    raise ValueError(f'{name} must use unique configured precision bits')
        if not isinstance(self.enabled, bool):
            raise ValueError('gate_diagnostics.enabled must be bool')
        for name in ('frames', 'trace_steps', 'probe_steps'):
            values = getattr(self, name)
            if (len(set(values)) != len(values) or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 1
                    for value in values)):
                raise ValueError(f'gate_diagnostics.{name} requires unique positive integers')
        for name in ('nodes_per_state', 'fit_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'gate_diagnostics.{name} must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('gate_diagnostics.seed must be a nonnegative integer')
        if (isinstance(self.fit_lr_scale, bool) or
                not math.isfinite(self.fit_lr_scale) or self.fit_lr_scale <= 0):
            raise ValueError('gate_diagnostics.fit_lr_scale must be finite and positive')
        if (not self.precision_policies or self.precision_policies[0] != 'fixed' or
                len(set(self.precision_policies)) != len(self.precision_policies) or
                any(value not in {'fixed', 'learned'} for value in self.precision_policies)):
            raise ValueError('precision_policies must start with fixed, optionally followed by learned')
        if (len(set(self.camera_names)) != len(self.camera_names) or
                any(not isinstance(name, str) or not name for name in self.camera_names)):
            raise ValueError('gate_diagnostics.camera_names must contain unique camera filenames')
        if self.fixed_precision_bits is not None and (
                isinstance(self.fixed_precision_bits, bool) or
                not isinstance(self.fixed_precision_bits, int) or
                (motion_config is not None and
                 self.fixed_precision_bits not in motion_config.precision_bits)):
            raise ValueError('fixed_precision_bits must be a configured precision')
        if self.enabled and not self.frames:
            raise ValueError('Enabled diagnostics require explicit frames')
        if self.enabled and self.probe_steps and not self.camera_names:
            raise ValueError('Candidate fitting requires explicit training camera_names')
        if motion_config is not None and any(
                step > motion_config.motion_steps for step in self.trace_steps + self.probe_steps):
            raise ValueError('Online diagnostics must run before motion is committed')

    def due(self, frame, step):
        return (self.enabled and frame in self.frames and
                step in self.trace_steps + self.probe_steps)


@contextmanager
def isolated_rng(device):
    """Data loading and rendering diagnostics cannot advance the training RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def node_masks(motion, level, node):
    """Return bool [nodes_at_depth] masks for this node and all descendants."""
    if (level < 0 or level >= len(motion.counts) or
            node < 0 or node >= motion.counts[level]):
        raise ValueError('Diagnostic node is outside the tree')
    masks = [torch.zeros(count, dtype=torch.bool, device=motion.base_xyz.device)
             for count in motion.counts]
    masks[level][node] = True
    for depth in range(level + 1, len(motion.counts)):
        masks[depth] = masks[depth - 1][motion.parents[depth - 1]]
    return masks


def _gradient(parameter, node):
    grad = None if parameter.grad is None else parameter.grad[node].detach()
    return dict(requires_grad=parameter.requires_grad,
                available=grad is not None,
                l2=None if grad is None else float(grad.norm()),
                values=None if grad is None else grad.cpu().reshape(-1).tolist())


def _component_row(parameter, row, parameter_name, components):
    result = {}
    for name in ('image', 'rate_proxy'):
        tensor = components[name].get(parameter_name)
        gradient = None if tensor is None else tensor[row]
        result[name] = dict(available=gradient is not None,
            l2=None if gradient is None else float(gradient.norm()),
            values=None if gradient is None else gradient.reshape(-1).cpu().tolist())
    return result


def node_trace(motion, step, level, node, gradient_components=None):
    gates = motion.refinement_masks(step)
    reached = motion._paths(gates)
    occupancy = motion._occupancy_values()
    position = motion._refine_position(level, node)
    precision, symbols, saturation = motion._selected_symbols(step)
    descendants = []
    masks = node_masks(motion, level, node)
    for depth in range(level, len(masks)):
        for row in torch.where(masks[depth])[0].tolist():
            descendants.append(dict(level=depth, node=row,
                reachable=bool(reached[depth][row]) and bool(occupancy[depth][row]),
                translation_gradient=_gradient(motion.region_translation[depth], row),
                rotation_gradient=_gradient(motion.region_rotation[depth], row),
                precision_gradient=_gradient(motion.precision_logits[depth], row)))
            if gradient_components is not None:
                descendants[-1]['gradient_components'] = {
                    name: _component_row(parameter, row, f'{field}.{depth}', gradient_components)
                    for name, field, parameter in (
                        ('translation', 'region_translation', motion.region_translation[depth]),
                        ('rotation', 'region_rotation', motion.region_rotation[depth]),
                        ('precision', 'precision_logits', motion.precision_logits[depth]))}
    result = dict(level=level, node=node, gate_position=position,
        members=int((motion.index[level] == node).sum()),
        reachable=bool(reached[level][node]) and bool(occupancy[level][node]),
        gate_open=bool(gates[level][position]),
        gate_logit=float(motion.refine_logits[level][position].detach()),
        gate_probability=float(motion.gate_probabilities(step)[level][position].detach()),
        gate_gradient=_gradient(motion.refine_logits[level], position),
        precision_bits=motion.config.precision_bits[int(precision[level][node])],
        motion_nonzero=bool(symbols[level][node].ne(0).any()),
        saturated=bool(saturation[level][node]), descendants=descendants)
    if gradient_components is not None:
        result['gate_gradient_components'] = _component_row(
            motion.refine_logits[level], position, f'refine_logits.{level}', gradient_components)
        result['gradient_component_context'] = gradient_components['context']
    return result


def sample_nodes(motion, step, frame, config, *, reachable_only=False):
    """Image-blind stratification; private RNG, deterministic per frame/group."""
    gates = motion.refinement_masks(step)
    paths, occupancy = motion._paths(gates), motion._occupancy_values()
    result = []
    for level, gate in enumerate(gates):
        rows = motion.active_layout['refine_rows'][level]
        reached = (paths[level][rows] > 0) & occupancy[level][rows]
        groups = [('reachable_closed', reached & ~gate),
                  ('reachable_open', reached & gate)]
        if not reachable_only:
            groups.append(('unreachable', ~reached & occupancy[level][rows]))
        for group_index, (group, mask) in enumerate(groups):
            population = rows[mask].detach().cpu().numpy()
            rng = np.random.default_rng(np.random.SeedSequence(
                [config.seed, frame, level, group_index]))
            chosen = rng.choice(population, min(config.nodes_per_state, len(population)), replace=False)
            for node in sorted(map(int, chosen)):
                result.append(dict(level=level, node=node, stratum=group,
                                   population=len(population)))
    return result


def exact_cost(motion, step, *, birth_bytes=0, subtree=None):
    """Use the production bit writer; padding applies to the whole frame."""
    payload, payload_bits, precision, symbols = motion._encode_payload(step)
    gates, occupancy = motion.refinement_masks(step), motion._occupancy_values()
    paths = motion._paths(gates)
    totals = dict(presence_bits=0, refine_flag_bits=0, precision_selector_bits=0, symbol_bits=0)
    local = dict(totals)
    bits = motion.base_xyz.new_tensor(motion.config.precision_bits)
    for level, count in enumerate(motion.counts):
        reached = paths[level].bool() & occupancy[level]
        nonzero = symbols[level].ne(0).any(-1)
        flags = torch.zeros_like(reached)
        if level < len(gates):
            active = torch.zeros(count, device=flags.device, dtype=torch.int32)
            active.index_add_(0, motion.parents[level], occupancy[level + 1].to(torch.int32))
            flags = active > 0
        for target, mask in ((totals, reached), (local, reached & subtree[level]
                             if subtree is not None else torch.zeros_like(reached))):
            costs = dict(presence_bits=int(mask.sum()), refine_flag_bits=int((mask & flags).sum()),
                precision_selector_bits=int((mask & nonzero).sum()) * motion.selector_bits,
                symbol_bits=int((bits[precision[level]] * (mask & nonzero)).sum()) * 6)
            for key, value in costs.items():
                target[key] += value
    if sum(totals.values()) != payload_bits:
        raise AssertionError('Diagnostic cost disagrees with the production encoder')
    return dict(motion_bytes=CODEC_HEADER_BYTES + len(payload),
        birth_bytes=int(birth_bytes), total_bytes=CODEC_HEADER_BYTES + len(payload) + int(birth_bytes),
        header_bytes=CODEC_HEADER_BYTES, payload_bits=payload_bits,
        padding_bits=(-payload_bits) % 8, components=totals,
        subtree_components=local if subtree is not None else None)


class DiagnosticCandidate(StorageAwareBinaryMotion):
    """Independent parameters with fixed topology/gates and optional precision STE."""

    def __init__(self, source, step, level, node, opened, config, precision_policy,
                 *, parent_bits=None, descendant_bits=None):
        super().__init__(copy.deepcopy(source.config))
        self.setup(source.base_xyz, source.base_rotation, source.extent,
                   frame_index=1, layout_override=source.active_layout)
        self.load_state_dict(source.state_dict())
        self.diagnostic_gates = [value.detach().clone() for value in source.refinement_masks(step)]
        self.diagnostic_precision = [value.detach().argmax(-1).clone()
            for value in source._precision_weights(step, differentiable=False)]
        paths = source._paths(self.diagnostic_gates)
        if not bool(paths[level][node]) or not bool(source._occupancy_values()[level][node]):
            raise ValueError('Stop/refine comparison requires a reachable occupied node')
        position = self._refine_position(level, node)
        self.diagnostic_gates[level][position] = opened
        self.subtree_masks = node_masks(self, level, node)
        reached = self._paths(self.diagnostic_gates)
        self.train_masks = [mask & value.bool() & occupied for mask, value, occupied in
            zip(self.subtree_masks, reached, self._occupancy_values())]
        self.precision_policy = precision_policy
        self.diagnostic_step = step
        if config.fixed_precision_bits is not None:
            index = self.config.precision_bits.index(config.fixed_precision_bits)
            for mask, choice in zip(self.subtree_masks, self.diagnostic_precision):
                choice[mask] = index
        if parent_bits is not None:
            self.diagnostic_precision[level][node] = self.config.precision_bits.index(parent_bits)
        if opened and descendant_bits is not None:
            index = self.config.precision_bits.index(descendant_bits)
            for depth in range(level + 1, len(self.counts)):
                if config.descendant_precision_scope == 'immediate_children' and depth > level + 1:
                    break
                self.diagnostic_precision[depth][self.train_masks[depth]] = index
        # Both policies begin with exactly the fixed candidate's hard selections.
        # For learned precision, preserve relative original logits where possible.
        with torch.no_grad():
            for logits, choice, mask in zip(self.precision_logits, self.diagnostic_precision, self.train_masks):
                if config.fixed_precision_bits is not None:
                    rows = torch.where(mask)[0]
                    logits[rows] = 0.
                    logits[rows, choice[rows]] = 1.
                mismatch = mask & (torch.softmax(logits / self._temperature(step), -1).argmax(-1) != choice)
                rows = torch.where(mismatch)[0]
                logits[rows, choice[rows]] = logits[rows].amax(-1) + 1.
        groups, self.masked_parameters = [], []
        for values, lr in ((self.region_translation, self.config.region_translation_lr),
                           (self.region_rotation, self.config.region_rotation_lr),
                           (self.precision_logits, self.config.precision_lr)):
            precision_group = values is self.precision_logits
            parameters = []
            for value, mask in zip(values, self.train_masks):
                enabled = bool(mask.any()) and (not precision_group or precision_policy == 'learned')
                value.requires_grad_(enabled)
                if enabled:
                    parameters.append(value)
                    self.masked_parameters.append((value, mask))
            if parameters:
                groups.append(dict(params=parameters, lr=lr * config.fit_lr_scale,
                    eps=self.config.gate_adam_eps if precision_group else self.config.adam_eps))
        for parameter in self.refine_logits:
            parameter.requires_grad_(False)
        self.fit_optimizer = torch.optim.Adam(groups, betas=tuple(self.config.adam_betas),
                                               eps=self.config.adam_eps)

    def refinement_masks(self, step):
        return self.diagnostic_gates

    def _precision_weights(self, step, *, differentiable=True):
        output = []
        for level, logits in enumerate(self.precision_logits):
            if self.precision_policy == 'learned':
                probability = torch.softmax(logits / self._temperature(self.diagnostic_step), dim=-1)
                choice = torch.where(self.train_masks[level], probability.detach().argmax(-1),
                                     self.diagnostic_precision[level])
            else:
                probability, choice = None, self.diagnostic_precision[level]
            hard = torch.nn.functional.one_hot(choice, len(self.config.precision_bits)).to(logits)
            output.append(hard + (probability - probability.detach()) * self.train_masks[level][:, None]
                          if differentiable and probability is not None else hard)
        return output

    def _precision_indices(self, step):
        return [value.argmax(-1) for value in self._precision_weights(step, differentiable=False)]

    def _region_motion(self, step=None, *, hard=False, active_rows=None):
        # Always form the full set of residuals; fixed paths decide participation.
        weights = self._precision_weights(step, differentiable=not hard)
        output = []
        for level, (translation, rotation) in enumerate(zip(self.region_translation, self.region_rotation)):
            candidates, _, _ = self._quantized_candidates(torch.cat((translation, rotation), -1))
            output.append(torch.einsum('nk,knc->nc', weights[level], candidates))
        return output

    def forward(self, step=None, training=False):
        regions, paths = self._region_motion(), self._paths(self.diagnostic_gates)
        if self.config.motion_compute == 'reference':
            return _conditional_transform(self.base_xyz, self.base_rotation, self.active_layout, regions, paths)
        return node_conditional_transform(self.base_xyz, self.base_rotation,
            self.active_layout, regions, paths, selected_only=True)

    def fit_step(self, loss, observer=None):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite diagnostic image loss')
        self.fit_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter, mask in self.masked_parameters:
            if parameter.grad is not None:
                parameter.grad[~mask] = 0
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite diagnostic gradient')
        observed = observer() if observer is not None else None
        self.fit_optimizer.step()
        return observed


def _evaluate(candidate, views, image_loss, step, birth_bytes, evaluation_views=()):
    rows = []
    check_rows = []
    with torch.no_grad():
        xyz, rotation = candidate()
        for destination, selected, role in ((rows, views, 'candidate_fitting_training_view'),
                                          (check_rows, evaluation_views, 'nonfitting_training_view')):
            for view in selected:
                if hasattr(image_loss, 'evaluate'):
                    metrics = image_loss.evaluate(view, xyz, rotation)
                    loss, quality = metrics['image_loss'], float(metrics['psnr'])
                    quality_value = quality if math.isfinite(quality) else None
                else:
                    loss, quality, quality_value = image_loss(view, xyz, rotation), None, None
                destination.append(dict(camera=view.path.name, role=role, image_loss=float(loss),
                    psnr=quality_value, psnr_is_infinite=quality is not None and math.isinf(quality)))
    def quality_mean(values):
        return (sum(row['psnr'] for row in values) / len(values)
                if values and all(row['psnr'] is not None for row in values) else None)
    return dict(image_loss=sum(row['image_loss'] for row in rows) / len(rows), views=rows,
                psnr=quality_mean(rows), evaluation_views=check_rows,
                evaluation_image_loss=(sum(row['image_loss'] for row in check_rows) / len(check_rows)
                                       if check_rows else None), evaluation_psnr=quality_mean(check_rows),
                cost=exact_cost(candidate, step, birth_bytes=birth_bytes, subtree=candidate.subtree_masks))


def compare_node(source, step, level, node, views, image_loss, config,
                 *, reference_bytes, rate_weight, birth_bytes=0, evaluation_views=(),
                 view_plan=None, gradient_components=None):
    """Fit each arm from the same state; never modify source tensors or optimizers."""
    if not views or reference_bytes <= 0 or not math.isfinite(reference_bytes):
        raise ValueError('Comparison requires views and a finite positive reference byte count')
    if not math.isfinite(rate_weight) or rate_weight < 0:
        raise ValueError('Rate weight must be finite and nonnegative')
    if config.candidate_mode == 'joint_precision':
        return compare_joint_node(source, step, level, node, views, image_loss, config,
            reference_bytes=reference_bytes, rate_weight=rate_weight, birth_bytes=birth_bytes,
            evaluation_views=evaluation_views, view_plan=view_plan, gradient_components=gradient_components)
    results = []
    for policy in config.precision_policies:
        arms = {}
        for opened, name in ((False, 'stop'), (True, 'refine')):
            _synchronize(source.base_xyz.device)
            started = time.perf_counter()
            candidate = DiagnosticCandidate(source, step, level, node, opened, config, policy)
            before = _evaluate(candidate, views, image_loss, step, birth_bytes)
            fit_losses, first_gradient = [], None
            _synchronize(source.base_xyz.device)
            fit_started = time.perf_counter()
            for fit_step in range(config.fit_steps):
                view = views[fit_step % len(views)]
                xyz, rotation = candidate()
                loss = image_loss(view, xyz, rotation)
                fit_losses.append(dict(step=fit_step + 1, camera=view.path.name, image_loss=float(loss.detach())))
                if fit_step == 0:
                    first_gradient = candidate.fit_step(loss, lambda: node_trace(candidate, step, level, node))
                else:
                    candidate.fit_step(loss)
                del loss, xyz, rotation
            _synchronize(source.base_xyz.device)
            fit_seconds = time.perf_counter() - fit_started
            after = _evaluate(candidate, views, image_loss, step, birth_bytes)
            _synchronize(source.base_xyz.device)
            arms[name] = dict(before=before, after=after, fit_losses=fit_losses,
                first_fit_gradient=first_gradient, fit_seconds=fit_seconds,
                total_seconds=time.perf_counter() - started,
                trainable_nodes=sum(int(mask.sum()) for mask in candidate.train_masks))
            del candidate
        stop, refine = arms['stop']['after'], arms['refine']['after']
        byte_delta = refine['cost']['total_bytes'] - stop['cost']['total_bytes']
        gain = stop['image_loss'] - refine['image_loss'] - rate_weight * byte_delta / reference_bytes
        results.append(dict(precision_policy=policy, arms=arms,
            distortion_gain=stop['image_loss'] - refine['image_loss'], byte_delta=byte_delta,
            rate_weight=rate_weight, reference_bytes=reference_bytes, gain=gain,
            supports_refine=gain > 0, fit_steps=config.fit_steps,
            fit_objective='image_loss_only', comparison_objective='image_loss_plus_actual_bytes',
            descendant_gate_policy='preserve_original; only sampled gate is forced',
            frozen='ancestors, siblings, topology, gates, GS attributes and newborn GS',
            camera_order=[view.path.name for view in views]))
    return dict(level=level, node=node, original=node_trace(source, step, level, node), comparisons=results)


def clone_motion(source):
    clone = StorageAwareBinaryMotion(copy.deepcopy(source.config))
    clone.restore(source.capture())
    return clone


def collect_gradient_components(source, step, view, image_loss, *, reference_bytes, rate_weight):
    """Recompute on independent motion/renderer graphs; main backward stays untouched."""
    clone = clone_motion(source)
    named = dict(clone.named_parameters())
    active = [(name, parameter) for name, parameter in named.items() if parameter.requires_grad]
    xyz, rotation = clone(step)
    image = image_loss(view, xyz, rotation)
    image_grads = torch.autograd.grad(image, [p for _, p in active], allow_unused=True)
    rate = rate_weight * sum(clone.storage_parts(step, hard=False).values()) / reference_bytes
    rate_grads = (torch.autograd.grad(rate, [p for _, p in active], allow_unused=True)
                  if rate.requires_grad else [None] * len(active))
    result = dict(image={name: None for name in named}, rate_proxy={name: None for name in named},
        context=dict(camera=view.path.name, image_loss=float(image.detach()),
            rate_proxy_loss=float(rate.detach()), rate_weight=rate_weight,
            semantics='Same current training camera on independent graphs. Rate is the differentiable '
                'training proxy, not the actual bitstream. Original total gradients remain separately labeled.'))
    for (name, _), a, b in zip(active, image_grads, rate_grads):
        result['image'][name] = None if a is None else a.detach().cpu().clone()
        result['rate_proxy'][name] = None if b is None else b.detach().cpu().clone()
    return result


def candidate_specs(config):
    specs = []
    if config.include_original_precision_candidates:
        specs.extend(dict(name=name, opened=opened, parent_bits=None, descendant_bits=None)
                     for name, opened in (('stop_original', False), ('refine_original', True)))
    specs.extend(dict(name=f'stop_parent{bits}', opened=False, parent_bits=bits, descendant_bits=None)
                 for bits in config.stop_parent_precision_bits)
    specs.extend(dict(name=f'refine_parent{parent}_descendant{child}', opened=True,
                      parent_bits=parent, descendant_bits=child)
                 for parent in config.refine_parent_precision_bits
                 for child in config.refine_descendant_precision_bits)
    return specs


def private_view_plan(config, frame, step, level, node):
    seed_parts = [config.seed, frame, step, level, node]
    count = len(config.camera_names)
    if not count:
        raise ValueError('Candidate fitting requires cameras')
    if config.view_sequence == 'round_robin':
        order = [index % count for index in range(config.fit_steps)]
    else:
        rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
        order = []
        while len(order) < config.fit_steps:
            order.extend(map(int, rng.permutation(count)))
        order = order[:config.fit_steps]
    return dict(camera_names=list(config.camera_names), indices=order,
                seed_parts=seed_parts, rule=config.view_sequence)


def _fit_joint_candidate(candidate, spec, step, level, node, views, evaluation_views,
                         image_loss, config, plan, birth_bytes):
    _synchronize(candidate.base_xyz.device)
    started = time.perf_counter()
    before = _evaluate(candidate, views, image_loss, step, birth_bytes, evaluation_views)
    _synchronize(candidate.base_xyz.device)
    fit_started = time.perf_counter()
    fit_losses, first_gradient = [], None
    for index, camera_index in enumerate(plan['indices']):
        view = views[camera_index]
        xyz, rotation = candidate()
        loss = image_loss(view, xyz, rotation)
        fit_losses.append(dict(step=index + 1, camera=view.path.name, image_loss=float(loss.detach())))
        observer = (lambda: node_trace(candidate, step, level, node)) if index == 0 else None
        observed = candidate.fit_step(loss, observer)
        if index == 0:
            first_gradient = observed
        del loss, xyz, rotation
    _synchronize(candidate.base_xyz.device)
    fit_seconds = time.perf_counter() - fit_started
    after = _evaluate(candidate, views, image_loss, step, birth_bytes, evaluation_views)
    assignment, trainable = [], []
    for depth, mask in enumerate(candidate.train_masks):
        rows = torch.where(mask)[0].tolist()
        for row in rows:
            assignment.append(dict(level=depth, node=row,
                role='sampled_parent' if depth == level else 'reachable_descendant',
                bits=candidate.config.precision_bits[int(candidate.diagnostic_precision[depth][row])]))
        if rows:
            trainable.extend(dict(parameter=f'{field}.{depth}', rows=rows)
                             for field in ('region_translation', 'region_rotation'))
    _synchronize(candidate.base_xyz.device)
    return dict(spec=spec, before=before, after=after, fit_losses=fit_losses,
        first_fit_gradient=first_gradient, first_fit_gradient_loss='candidate_image_loss_only',
        fit_seconds=fit_seconds, total_seconds=time.perf_counter() - started,
        precision_assignment=assignment, trainable_parameters=trainable,
        trainable_nodes=sum(int(mask.sum()) for mask in candidate.train_masks))


def compare_joint_node(source, step, level, node, views, image_loss, config, *,
                       reference_bytes, rate_weight, birth_bytes=0, evaluation_views=(),
                       view_plan=None, gradient_components=None):
    plan = view_plan or private_view_plan(config, 0, step, level, node)
    if (plan['camera_names'] != [view.path.name for view in views] or
            len(plan['indices']) != config.fit_steps or any(
                index < 0 or index >= len(views) for index in plan['indices'])):
        raise ValueError('Private candidate view plan does not match fitting views/budget')
    original = node_trace(source, step, level, node, gradient_components)
    baseline_candidate = DiagnosticCandidate(source, step, level, node,
        original['gate_open'], config, 'fixed')
    baseline = _evaluate(baseline_candidate, views, image_loss, step, birth_bytes, evaluation_views)
    del baseline_candidate
    results = []
    for spec in candidate_specs(config):
        candidate = DiagnosticCandidate(source, step, level, node, spec['opened'], config, 'fixed',
            parent_bits=spec['parent_bits'], descendant_bits=spec['descendant_bits'])
        result = _fit_joint_candidate(candidate, spec, step, level, node, views, evaluation_views,
                                      image_loss, config, plan, birth_bytes)
        del candidate
        byte_delta = result['after']['cost']['total_bytes'] - baseline['cost']['total_bytes']
        penalty = rate_weight * byte_delta / reference_bytes
        result.update(image_gain=baseline['image_loss'] - result['after']['image_loss'],
            byte_delta=byte_delta, rate_penalty=penalty,
            gain=baseline['image_loss'] - result['after']['image_loss'] - penalty)
        result['evaluation_gain'] = (None if not evaluation_views else
            baseline['evaluation_image_loss'] - result['after']['evaluation_image_loss'] - penalty)
        per_view = []
        for key in ('views', 'evaluation_views'):
            for before, after in zip(baseline[key], result['after'][key]):
                image_gain = before['image_loss'] - after['image_loss']
                per_view.append(dict(camera=before['camera'], role=before['role'],
                    image_gain=image_gain, rate_gain=image_gain - penalty, byte_delta=byte_delta,
                    psnr_gain=(after['psnr'] - before['psnr']
                               if before['psnr'] is not None and after['psnr'] is not None else None)))
        result['per_view_gains'] = per_view
        results.append(result)
    selected = max(results, key=lambda row: row['gain'])
    check_selected = (max(results, key=lambda row: row['evaluation_gain']) if evaluation_views else None)
    stop = max((row for row in results if not row['spec']['opened']), key=lambda row: row['gain'])
    refine = max((row for row in results if row['spec']['opened']), key=lambda row: row['gain'])
    return dict(level=level, node=node, original=original, schema='joint_precision_candidates_v1',
        baseline=baseline, candidates=results, view_plan=plan,
        selected_by_fitting_rate_score=selected['spec']['name'],
        selected_by_check_rate_score=None if check_selected is None else check_selected['spec']['name'],
        best_stop=stop['spec']['name'], best_refine=refine['spec']['name'],
        best_refine_minus_stop_gain=refine['gain'] - stop['gain'],
        fit_steps=config.fit_steps, reference_bytes=reference_bytes, rate_weight=rate_weight,
        fit_objective='image_loss_only', comparison_objective='image_loss_plus_actual_whole_frame_bytes',
        precision_space=f'Sampled parent precision x uniform {config.descendant_precision_scope} precision; '
            'all other descendant gates retain their original choices.',
        frozen='ancestors, siblings, topology, gates, GS appearance and newborn GS',
        limitations=['Restricted local candidates, not a whole-tree optimum.',
            'Finite extra fitting budget does not establish convergence.',
            'Full-frame rendering couples regions; node gains cannot be added independently.',
            'Check views are ordinary training cameras excluded only from this extra candidate fitting.'])


def write_trace(path, *, frame, step, context, rows, camera=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(dict(frame=frame, step=step, context=context, camera=camera,
                                         **row), allow_nan=False) + '\n')


def run_probes(motion, *, frame, step, context, views, image_loss, config,
               reference_bytes, rate_weight, output_dir, birth_bytes=0, evaluation_views=(),
               samples=None, view_plans=None, gradient_components=None, snapshot_path=None):
    """Append one recoverable JSONL record per completed node comparison."""
    path = Path(output_dir) / f'frame_{frame:03d}_step_{step:04d}_{context}.jsonl'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic comparisons: {path}')
    if samples is None:
        samples = sample_nodes(motion, step, frame, config, reachable_only=True)
    if view_plans is None:
        view_plans = {f"{item['level']}:{item['node']}": private_view_plan(
            config, frame, step, item['level'], item['node']) for item in samples}
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = path.with_suffix('.plan.json')
    if plan.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic sample plan: {plan}')
    plan.write_text(json.dumps(dict(frame=frame, step=step, context=context,
        samples=samples, image_blind=True, camera_names=[view.path.name for view in views],
        seed=config.seed, precision_policies=list(config.precision_policies), view_plans=view_plans,
        candidate_mode=config.candidate_mode, candidates=candidate_specs(config)
        if config.candidate_mode == 'joint_precision' else None,
        evaluation_camera_names=[view.path.name for view in evaluation_views],
        source_snapshot=str(snapshot_path) if snapshot_path is not None else None), indent=2), encoding='utf-8')
    for sample in samples:
        record = compare_node(motion, step, sample['level'], sample['node'], views,
            image_loss, config, reference_bytes=reference_bytes, rate_weight=rate_weight,
            birth_bytes=birth_bytes, evaluation_views=evaluation_views,
            view_plan=view_plans[f"{sample['level']}:{sample['node']}"], gradient_components=gradient_components)
        write_trace(path, frame=frame, step=step, context=context,
                    rows=[dict(stage=motion.stage(step), sample=sample, **record)])
    return path

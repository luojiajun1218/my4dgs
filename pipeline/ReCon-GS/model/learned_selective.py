"""Learned tree cuts with full rigid candidates and straight-through gates."""
from dataclasses import dataclass, asdict, fields
from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F
from .selective_motion import SelectiveMotion, KEEP, SHARED, SPLIT
from .region_rigid import apply_region_rigid


@dataclass
class LearnedSelectiveConfig:
    fixed_leaf_ablation: bool = False
    max_depth: Optional[int] = None
    coarse_regions: Optional[int] = None
    max_leaf_regions: Optional[int] = None
    min_region_gs: Optional[int] = None
    warmup_steps: Optional[int] = None
    joint_steps: Optional[int] = None
    refine_steps: Optional[int] = None
    incremental_steps: Optional[int] = None
    enable_incremental: Optional[bool] = None
    translation_lr: Optional[float] = None
    rotation_lr: Optional[float] = None
    gate_lr: Optional[float] = None
    gate_seed: Optional[int] = None
    diagnose_soft_hard: Optional[bool] = None
    adam_betas: Optional[List[float]] = None
    adam_eps: Optional[float] = None
    initial_logits: Optional[List[float]] = None
    temperature_start: Optional[float] = None
    temperature_end: Optional[float] = None
    shared_cost: Optional[float] = None
    split_cost: Optional[float] = None
    structure_warmup_steps: Optional[int] = None
    structure_ramp_steps: Optional[int] = None
    clone_sigma_scale: Optional[float] = None
    enable_change_loss: Optional[bool] = None
    change_loss_weight: Optional[float] = None
    camera_match_atol: Optional[float] = None
    change_cache_device: Optional[str] = None

    def validate(self):
        missing = [f.name for f in fields(self) if getattr(self, f.name) is None]
        if missing:
            raise ValueError('learned_selective_motion requires explicit YAML values: ' + ', '.join(missing))
        positive_ints = ('max_depth', 'coarse_regions', 'max_leaf_regions', 'min_region_gs',
                         'refine_steps', 'structure_ramp_steps')
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('incremental_steps', 'structure_warmup_steps', 'warmup_steps', 'joint_steps'):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not isinstance(self.fixed_leaf_ablation, bool):
            raise ValueError('fixed_leaf_ablation must be a boolean')
        if self.fixed_leaf_ablation:
            if self.warmup_steps != 0 or self.joint_steps != 0:
                raise ValueError('Fixed leaf ablation requires warmup_steps=0 and joint_steps=0')
            if self.shared_cost != 0 or self.split_cost != 0 or self.enable_change_loss or self.change_loss_weight != 0 or self.diagnose_soft_hard:
                raise ValueError('Fixed leaf ablation disables structure costs, change loss and soft/hard diagnostics')
        elif self.warmup_steps < 1 or self.joint_steps < 1:
            raise ValueError('Learned selection requires positive warmup_steps and joint_steps')
        if isinstance(self.gate_seed, bool) or not isinstance(self.gate_seed, int) or self.gate_seed < 0:
            raise ValueError('gate_seed must be a nonnegative integer')
        if self.max_leaf_regions < self.coarse_regions:
            raise ValueError('max_leaf_regions must cover coarse_regions')
        if not isinstance(self.enable_incremental, bool) or self.enable_incremental != (self.incremental_steps > 0):
            raise ValueError('enable_incremental must agree with incremental_steps')
        if len(self.adam_betas) != 2 or not all(0 <= x < 1 for x in self.adam_betas):
            raise ValueError('adam_betas must have two values in [0, 1)')
        if len(self.initial_logits) != 3 or not torch.isfinite(torch.tensor(self.initial_logits)).all():
            raise ValueError('initial_logits must contain three finite KEEP/SHARED/SPLIT scores')
        initial_probability = torch.tensor(self.initial_logits, dtype=torch.float32).softmax(-1)
        if not bool(((initial_probability > 0) & (initial_probability < 1)).all()):
            raise ValueError('initial_logits must produce nondegenerate floating-point probabilities')
        for name in ('translation_lr', 'rotation_lr', 'gate_lr', 'adam_eps', 'temperature_start',
                     'temperature_end', 'clone_sigma_scale'):
            if not torch.isfinite(torch.tensor(getattr(self, name))) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('shared_cost', 'split_cost', 'change_loss_weight', 'camera_match_atol'):
            if not torch.isfinite(torch.tensor(getattr(self, name))) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not isinstance(self.enable_change_loss, bool) or self.enable_change_loss != (self.change_loss_weight > 0):
            raise ValueError('enable_change_loss must agree with a positive change_loss_weight')
        if not isinstance(self.diagnose_soft_hard, bool):
            raise ValueError('diagnose_soft_hard must be a boolean')
        if self.change_cache_device not in ('cpu', 'cuda'):
            raise ValueError('change_cache_device must be cpu or cuda')

    @property
    def motion_steps(self):
        return self.warmup_steps + self.joint_steps + self.refine_steps

    @property
    def total_steps(self):
        return self.motion_steps + self.incremental_steps


class LearnedSelectiveMotion(SelectiveMotion):
    def setup(self, xyz, raw_rotation):
        super().setup(xyz, raw_rotation)
        if 'selected_sources' in self._buffers:
            delattr(self, 'selected_sources')
        self.parents = {child: n for n, row in enumerate(self.children) for child in row}
        self.roots = [n for n in range(len(self.members)) if n not in self.parents]
        self.register_or_set('observation_count', torch.zeros(len(self.members), device=xyz.device, dtype=torch.long))
        if self.config.fixed_leaf_ablation:
            self.register_or_set('last_gates', F.one_hot(self.states, 3).to(xyz))
            self.register_or_set('selected_sources', torch.arange(len(self.members), device=xyz.device))
            self.selected = True
            self.gate_generator = None
        else:
            self.logits = nn.Parameter(xyz.new_tensor(self.config.initial_logits).expand(len(self.members), 3).clone())
            self.register_or_set('last_gates', torch.zeros_like(self.logits))
            self.gate_generator = torch.Generator(device=xyz.device)
            self.gate_generator.manual_seed(self.config.gate_seed)
        self.build_index_cache()
        self.last_step = 0
        self.last_metrics = {}
        self.make_optimizer()

    def make_optimizer(self):
        groups = [dict(params=[self.translation], lr=self.config.translation_lr),
                  dict(params=[self.rotation_vector], lr=self.config.rotation_lr)]
        if hasattr(self, 'logits') and not self.config.fixed_leaf_ablation:
            groups.append(dict(params=[self.logits], lr=self.config.gate_lr))
        self.optimizer = torch.optim.Adam(groups, betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)

    def stage(self, step):
        if self.config.fixed_leaf_ablation:
            return 'fixed_leaf', self.owner
        if step <= self.config.warmup_steps:
            return 'soft_warmup', self.owner
        if step <= self.config.warmup_steps + self.config.joint_steps:
            return 'hard_joint', self.owner
        return 'fixed_refine', self.owner

    def temperature(self, step):
        fraction = min(max((step - 1) / max(self.config.warmup_steps + self.config.joint_steps - 1, 1), 0), 1)
        return self.config.temperature_start * (self.config.temperature_end / self.config.temperature_start) ** fraction

    def probabilities(self):
        masked = self.logits.clone()
        leaves = torch.tensor([not row for row in self.children], device=masked.device)
        masked[leaves, SPLIT] = -torch.inf
        return masked.softmax(-1)

    def build_index_cache(self):
        device = self.base_xyz.device
        levels = torch.empty((max(self.depths) + 1, len(self.base_xyz)), device=device, dtype=torch.long)
        for n, indices in enumerate(self.members):
            levels[self.depths[n], indices] = n
            if not self.children[n]:
                levels[self.depths[n]:, indices] = n
        self.level_nodes = levels
        self.parent_index = torch.tensor([self.parents.get(n, n) for n in range(len(self.members))], device=device)
        self.depth_nodes = [torch.tensor([n for n, d in enumerate(self.depths) if d == depth], device=device)
                            for depth in range(max(self.depths) + 1)]

    @torch.no_grad()
    def observe(self, visibility):
        visible = visibility[:len(self.base_xyz)].long()
        support = torch.zeros_like(self.observation_count)
        for nodes in self.level_nodes:
            support.scatter_reduce_(0, nodes, visible, reduce='amax', include_self=True)
        self.observation_count.add_(support)

    def gates(self, step, training, soft=False):
        if self.config.fixed_leaf_ablation:
            return self.last_gates
        if self.selected and not soft:
            return F.one_hot(self.states, 3).to(self.logits)
        logits = self.logits.masked_fill(torch.tensor([not row for row in self.children],
                                                     device=self.logits.device)[:, None] &
                                         (torch.arange(3, device=self.logits.device)[None] == SPLIT), -torch.inf)
        if soft:
            gates = (logits / self.temperature(step)).softmax(-1)
        elif training:
            if self.stage(step)[0] == 'soft_warmup':
                gates = (logits.detach() / self.temperature(step)).softmax(-1)
            else:
                uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype, generator=self.gate_generator)
                tiny = torch.finfo(logits.dtype).eps
                noise = -torch.log(-torch.log(uniform.clamp(tiny, 1 - tiny)))
                soft = ((logits + noise) / self.temperature(step)).softmax(-1)
                hard = F.one_hot(soft.argmax(-1), 3).to(soft)
                gates = hard + (soft - soft.detach())
        else:
            gates = F.one_hot(logits.argmax(-1), 3).to(logits)
        # Missing geometric support supplies no gate evidence. Retain shared motion.
        fallback = F.one_hot(torch.full_like(self.states, SHARED), 3).to(gates)
        return torch.where((self.observation_count > 0)[:, None], gates, fallback)

    def motion_sources(self):
        if self.selected and 'selected_sources' in self._buffers:
            return self.selected_sources
        observed = (self.observation_count > 0).detach().cpu().tolist()
        sources = []
        for n in range(len(self.members)):
            source = n
            while source in self.parents and not observed[source]:
                source = self.parents[source]
            sources.append(source)
        return torch.tensor(sources, device=self.base_xyz.device)

    def transform(self, owner=None, keep=None, *, step=None, training=False, soft=False, update_diagnostics=None):
        step = self.last_step if step is None else step
        update_diagnostics = training if update_diagnostics is None else update_diagnostics
        if self.config.fixed_leaf_ablation:
            if training:
                self.last_step = step
            nodes = self.owner
            return apply_region_rigid(self.base_xyz, self.base_rotation, self.centers[nodes],
                                      self.translation[nodes], self.rotation_vector[nodes])
        gates = self.gates(step, training, soft=soft)
        sources = self.motion_sources()
        split_xyz, split_rot = self.base_xyz, self.base_rotation
        hard_forward = not soft and (self.selected or not training or self.stage(step)[0] == 'hard_joint')
        reference = F.normalize(self.base_rotation, dim=-1)
        for nodes in reversed(self.level_nodes):
            source = sources[nodes]
            moved_xyz, moved_rot = apply_region_rigid(self.base_xyz, self.base_rotation,
                self.centers[source], self.translation[source], self.rotation_vector[source])
            gate = gates[nodes]
            mixed_xyz = (self.base_xyz * gate[:, KEEP:KEEP + 1] +
                         moved_xyz * gate[:, SHARED:SHARED + 1] + split_xyz * gate[:, SPLIT:SPLIT + 1])
            options_rot = F.normalize(torch.stack((self.base_rotation, moved_rot, split_rot), dim=1), dim=-1)
            options_rot = torch.where((options_rot * reference[:, None]).sum(-1, keepdim=True) < 0,
                                      -options_rot, options_rot)
            quaternion_sum = (options_rot * gate[:, :, None]).sum(1)
            mixed_rot = F.normalize(quaternion_sum, dim=-1)
            mixed_rot = torch.where((quaternion_sum.norm(dim=-1, keepdim=True) == 0), reference, mixed_rot)
            if hard_forward:
                action = gate.detach().argmax(-1)[:, None]
                exact_xyz = torch.where(action == KEEP, self.base_xyz,
                                        torch.where(action == SHARED, moved_xyz, split_xyz))
                exact_rot = torch.where(action == KEEP, self.base_rotation,
                                        torch.where(action == SHARED, moved_rot, split_rot))
                mixed_xyz = exact_xyz.detach() + (mixed_xyz - mixed_xyz.detach())
                mixed_rot = exact_rot.detach() + (mixed_rot - mixed_rot.detach())
            split_xyz, split_rot = mixed_xyz, mixed_rot
        if update_diagnostics:
            self.last_gates.copy_(gates.detach())
            if training:
                self.last_step = step
            if not self.selected:
                self.update_cut(gates.detach())
        return split_xyz, split_rot

    @torch.no_grad()
    def cut_from_gates(self, gates):
        states = gates.argmax(-1)
        owner = torch.empty_like(self.owner)
        unresolved = torch.ones(len(self.base_xyz), device=self.base_xyz.device, dtype=torch.bool)
        for nodes in self.level_nodes:
            terminal = states[nodes] != SPLIT
            assign = unresolved & terminal
            owner[assign] = nodes[assign]
            unresolved &= ~terminal
        if bool(unresolved.any()):
            raise RuntimeError('Learned cut does not cover every Gaussian')
        return states, owner

    @torch.no_grad()
    def update_cut(self, gates):
        states, owner = self.cut_from_gates(gates)
        self.states.copy_(states)
        self.owner.copy_(owner)

    @torch.no_grad()
    def select(self):
        if self.config.fixed_leaf_ablation:
            return
        self.update_cut(self.gates(self.last_step, False))
        self.register_or_set('selected_sources', self.motion_sources().clone())
        self.selected = True
        self.logits.grad = None
        self.optimizer.state.pop(self.logits, None)

    def regularization(self, step):
        if self.config.fixed_leaf_ablation:
            shared = self.translation.new_tensor(sum(not row for row in self.children))
            split = self.translation.new_tensor(sum(bool(row) for row in self.children))
            return self.translation.sum() * 0, dict(fixed_shared_raw=shared, fixed_split_raw=split,
                                                   fixed_shared_per_root=shared / len(self.roots),
                                                   fixed_split_per_root=split / len(self.roots))
        probabilities = (F.one_hot(self.states, 3).to(self.logits) if self.selected else self.probabilities())
        fallback = F.one_hot(torch.full_like(self.states, SHARED), 3).to(probabilities)
        probabilities = torch.where((self.observation_count > 0)[:, None], probabilities, fallback)
        reach = torch.zeros_like(probabilities[:, 0])
        for depth, nodes in enumerate(self.depth_nodes):
            if depth == 0:
                values = torch.ones_like(reach[nodes])
            else:
                parents = self.parent_index[nodes]
                values = reach[parents] * probabilities[parents, SPLIT]
            reach = reach.index_copy(0, nodes, values)
        shared = (reach * probabilities[:, SHARED]).sum()
        split = (reach * probabilities[:, SPLIT]).sum()
        scale = min(max((step - self.config.structure_warmup_steps) / self.config.structure_ramp_steps, 0), 1)
        loss = scale * (self.config.shared_cost * shared + self.config.split_cost * split) / len(self.roots)
        return loss, dict(expected_shared_raw=shared, expected_split_raw=split,
                          expected_shared_per_root=shared / len(self.roots), expected_split_per_root=split / len(self.roots),
                          structure_scale=probabilities.new_tensor(scale), temperature=probabilities.new_tensor(self.temperature(step)),
                          probability_keep=probabilities[:, KEEP].mean(), probability_shared=probabilities[:, SHARED].mean(),
                          probability_split=probabilities[:, SPLIT].mean())

    @torch.no_grad()
    def optimizer_step(self, owner=None):
        if self.config.fixed_leaf_ablation:
            self.last_metrics.update(stage='fixed_leaf', step=self.last_step)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            return
        self.last_metrics['gate_grad_norm'] = float(self.logits.grad.norm()) if self.logits.grad is not None else 0.
        if self.selected or self.stage(self.last_step)[0] == 'soft_warmup':
            self.logits.grad = None
            self.optimizer.state.pop(self.logits, None)
        if self.stage(self.last_step)[0] == 'hard_joint':
            value = self.last_metrics['gate_grad_norm']
            self.last_metrics['joint_last_gate_grad_norm'] = value
            self.last_metrics['joint_max_gate_grad_norm'] = max(value, self.last_metrics.get('joint_max_gate_grad_norm', 0.))
        self.last_metrics.update(stage=self.stage(self.last_step)[0], step=self.last_step)
        if self.selected:
            active = torch.zeros(len(self.members), device=self.owner.device, dtype=torch.bool)
            nodes = self.owner.unique()
            nodes = nodes[self.states[nodes] == SHARED]
            active[self.motion_sources()[nodes]] = True
            for parameter in (self.translation, self.rotation_vector):
                if parameter.grad is not None:
                    parameter.grad[~active] = 0
                for value in self.optimizer.state.get(parameter, {}).values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value[~active] = 0
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def capture(self):
        packet = super().capture()
        if packet is not None:
            packet.update(mode='learned_selective', last_step=self.last_step, last_metrics=self.last_metrics,
                          gate_rng_state=None if self.gate_generator is None else self.gate_generator.get_state())
        return packet

    def restore(self, packet):
        if packet is not None:
            saved_config = dict(packet['config'])
            saved_config.setdefault('fixed_leaf_ablation', False)
            if saved_config['fixed_leaf_ablation'] != self.config.fixed_leaf_ablation:
                raise ValueError('Checkpoint fixed_leaf_ablation does not match configured selection policy')
            packet = dict(packet, config=saved_config)
        # Base restore builds tree/parameters before loading dynamic cut buffers.
        if packet is not None and not self.config.fixed_leaf_ablation and 'selected_sources' in packet['state']:
            saved = packet['state']['selected_sources']
            packet = dict(packet, state={k: v for k, v in packet['state'].items() if k != 'selected_sources'})
        else:
            saved = None
        super().restore(packet)
        if saved is not None:
            self.register_or_set('selected_sources', saved.clone())
        if packet is not None:
            self.parents = {c: n for n, row in enumerate(self.children) for c in row}
            self.roots = [n for n in range(len(self.members)) if n not in self.parents]
            self.build_index_cache()
            if not self.config.fixed_leaf_ablation:
                self.gate_generator.set_state(packet['gate_rng_state'].cpu())
            self.last_step = packet['last_step']
            self.last_metrics = packet.get('last_metrics', {})

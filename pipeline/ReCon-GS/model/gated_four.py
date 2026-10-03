"""Four nested motion levels with three gradient-selected refinement gates."""
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from ..utils import batched_knn
from .deformation import Adaptive_Grid, DeformConfig
from .nested import NestedFineGrid, weighted_gradient_diagnostics


def _factor_child_index_rows(indices: np.ndarray):
    """Factor repeated [child_0, child_1, ...] rows into a table and row ids."""
    if indices.ndim != 2:
        raise ValueError("Blend indices must have shape [rows, slots]")
    table, groups = np.unique(indices, axis=0, return_inverse=True)
    return table, groups.astype(np.int32, copy=False)


class GatedFourDeformation(nn.Module):
    """Keep L1 active and open L2, L3, L4 at configured training steps."""

    def __init__(self, config: DeformConfig):
        super().__init__()
        self.config = config
        self.gate_steps = (config.gate_step_l2, config.gate_step_l3, config.gate_step_l4)
        gaps = (self.gate_steps[0], self.gate_steps[1] - self.gate_steps[0],
                self.gate_steps[2] - self.gate_steps[1])
        if config.hierarchy_mode != "nested" or config.refinement_mode != "gradhet":
            raise ValueError("Four-level hierarchy requires nested and gradhet")
        if config.num_grid_levels != 4 or config.grid_level_ratio < 2 or config.max_gs_per_grid < 1:
            raise ValueError("Four-level hierarchy requires four levels and ratio >= 2")
        if not 0 < self.gate_steps[0] < self.gate_steps[1] < self.gate_steps[2] < config.num_stage1_steps:
            raise ValueError("Gate steps must increase within the motion stage")
        if not 1 <= config.gate_score_window <= min(gaps):
            raise ValueError("Gate score window must fit before each gate")
        if not 0 <= config.gate_keep_ratio <= 1 or not 0 <= config.fine_overlap_ratio <= 1:
            raise ValueError("Gate keep and overlap ratios must be in [0, 1]")

        self.grids = nn.ModuleList()
        self.register_parameter("delta", None)
        self.register_buffer("index", None)
        for level in (2, 3, 4):
            self.register_buffer(f"gate_mask_l{level}", None)
        for name in ("score_sum", "score_after_sum", "score_visible_sum", "score_observations"):
            self.register_buffer(name, None)
        self.score_steps = []
        self.active_gate_count = 0
        self.grids_num = 0
        self.optimizer = None

    @property
    def get_grid_num(self) -> int:
        return self.grids_num

    def _gate_mask(self, level: int) -> torch.Tensor:
        return getattr(self, f"gate_mask_l{level}")

    @torch.no_grad()
    def _reset_scores(self):
        self.score_steps = []
        if self.active_gate_count == 3:
            self.score_sum = self.score_after_sum = None
            self.score_visible_sum = self.score_observations = None
            return
        parent_count = self.grids[self.active_gate_count].vertex_num
        self.score_sum = self.delta.new_zeros(parent_count)
        self.score_after_sum = self.delta.new_zeros(parent_count)
        self.score_visible_sum = torch.zeros(parent_count, device=self.delta.device, dtype=torch.long)
        self.score_observations = torch.zeros(parent_count, device=self.delta.device, dtype=torch.long)

    @torch.no_grad()
    def _initialize(self, xyz: torch.Tensor, rebuild_base: bool):
        old_base = self.grids[0] if self.grids else None
        old_delta = self.delta.detach() if self.delta is not None else None
        if old_base is not None and not rebuild_base:
            base = old_base
            base.reset_p2v(xyz)
        else:
            fine_budget = math.ceil(xyz.shape[0] / self.config.max_gs_per_grid)
            base_count = max(1, fine_budget // self.config.grid_level_ratio ** 2)
            base = Adaptive_Grid(base_count, xyz).to(xyz.device)

        grids = [base]
        for _ in range(3):
            parent = grids[-1].p2v.flatten()
            grids.append(NestedFineGrid(xyz, parent, grids[-1].vertex_num,
                                        self.config.grid_level_ratio,
                                        self.config.fine_overlap_ratio))
        self.grids = nn.ModuleList(grids)
        self.grids_num = sum(grid.vertex_num for grid in grids)
        delta = xyz.new_zeros(self.grids_num, 7)
        delta[:, 3] = 1
        if old_base is not None:
            if rebuild_base:
                _, match = batched_knn(base.vertex, old_base.vertex, k=1)
                inherited = old_delta[match.flatten()]
            else:
                inherited = old_delta[:base.vertex_num]
            if self.config.momentum is not None:
                inherited = inherited * self.config.momentum
            delta[:base.vertex_num] = inherited
        self.delta = nn.Parameter(delta)

        offsets, index = [], []
        offset = 0
        for grid in grids:
            offsets.append(offset)
            index.append(grid.p2v.flatten() + offset)
            offset += grid.vertex_num
        self.index = torch.stack(index)
        self.offsets = tuple(offsets)
        for level in (2, 3, 4):
            setattr(self, f"gate_mask_l{level}",
                    torch.zeros(grids[level - 2].vertex_num, device=xyz.device, dtype=torch.bool))
        self.active_gate_count = 0
        self._reset_scores()
        self.optimizer = torch.optim.Adam([self.delta], lr=self.config.lr, eps=1e-15)

    def setup(self, xyz: torch.Tensor, reset_grid: bool = False):
        self._initialize(xyz, rebuild_base=reset_grid or not self.grids)

    def reset(self, xyz: torch.Tensor):
        self._initialize(xyz, rebuild_base=False)

    def reset_grid(self, xyz: torch.Tensor):
        self._initialize(xyz, rebuild_base=True)

    def gate_due(self, step: int) -> bool:
        return self.active_gate_count < 3 and step == self.gate_steps[self.active_gate_count]

    def collecting_gradients(self, step: int) -> bool:
        if self.active_gate_count == 3:
            return False
        end = self.gate_steps[self.active_gate_count]
        return end - self.config.gate_score_window < step <= end

    @torch.no_grad()
    def accumulate_gradient_score(self, gradient: torch.Tensor, visible: torch.Tensor,
                                  step: int = None):
        if gradient is None or self.active_gate_count == 3:
            raise RuntimeError("Missing gradient or no gate remains")
        parent = self.grids[self.active_gate_count].p2v.flatten()
        child = self.grids[self.active_gate_count + 1]
        before, after, count = weighted_gradient_diagnostics(
            gradient.detach(), parent, visible, self.grids[self.active_gate_count].vertex_num,
            child)
        observed = count >= 2
        self.score_sum[observed] += before[observed]
        self.score_after_sum[observed] += after[observed]
        self.score_visible_sum[observed] += count[observed]
        self.score_observations += observed.long()
        if step is not None:
            self.score_steps.append(step)

    @torch.no_grad()
    def activate_refinement(self, step: int) -> Dict[str, Any]:
        if not self.gate_due(step):
            raise ValueError("Gate activation must occur at its configured step")
        stage = self.active_gate_count
        parent, child = self.grids[stage], self.grids[stage + 1]
        score = self.score_sum / self.score_observations.clamp_min(1)
        after = self.score_after_sum / self.score_observations.clamp_min(1)
        count = torch.bincount(parent.p2v.flatten(), minlength=parent.vertex_num)
        child_count = torch.bincount(child.parent, minlength=parent.vertex_num)
        ancestor_active = torch.ones_like(count, dtype=torch.bool)
        if stage:
            ancestor_active = self._gate_mask(stage + 1)[parent.parent]
        valid = (count > 0) & ancestor_active
        eligible = valid & (child_count >= 2) & (self.score_observations > 0) & (score > 0)
        budget = int(valid.sum().item() * self.config.gate_keep_ratio)
        candidates = torch.where(eligible)[0]
        selected = candidates[torch.argsort(score[candidates], descending=True,
                                            stable=True)[:budget]]
        mask = self._gate_mask(stage + 2)
        mask.zero_()
        mask[selected] = True
        stats = dict(prototype_version=3, gate_step=step, parent_level=stage + 1,
                     opened_level=stage + 2, H_before=score, H_after=after,
                     H_reduction=score - after, count=count, child_count=child_count,
                     observed=self.score_observations.clone(),
                     visible_sum=self.score_visible_sum.clone(), eligible=eligible,
                     active=mask.clone(), parent_xyz=parent.vertex,
                     score_steps=list(self.score_steps), budget=budget,
                     selected=int(mask.sum().item()))
        self.active_gate_count += 1
        self._reset_scores()
        return stats

    def forward(self, xyz: torch.Tensor):
        identity = self.delta.new_zeros(7)
        identity[3] = 1
        delta = self.delta[self.index[0]]
        for stage in range(self.active_gate_count):
            child = self.grids[stage + 1]
            parent_index = self.grids[stage].p2v.flatten()
            residual = self.delta[child.blend_indices + self.offsets[stage + 1]] - identity
            blended = (residual * child.blend_weights[:, :, None]).sum(1)
            gate = self._gate_mask(stage + 2)[parent_index].to(delta.dtype)[:, None]
            delta = delta + blended * gate
        return delta[:, :3].contiguous(), torch.nn.functional.normalize(
            delta[:, 3:].contiguous(), dim=-1)

    def reg_loss(self):
        identity = self.delta.new_zeros(1, 7)
        identity[:, 3] = 1
        active = [torch.ones(self.grids[0].vertex_num, device=self.delta.device, dtype=torch.bool)]
        for stage in range(3):
            active.append(self._gate_mask(stage + 2)[self.grids[stage + 1].parent])
        # Keep the original three-level penalty scale as dormant L4 candidates
        # are added; the denominator stays fixed when gates open.
        reference_rows = sum(grid.vertex_num for grid in self.grids[:3])
        return ((self.delta - identity).abs() * torch.cat(active)[:, None]).sum() / (7 * reference_rows)

    def capture(self) -> Dict[str, Any]:
        if not self.grids:
            return {}
        return dict(prototype_version=3, grids=self.grids, delta=self.delta, index=self.index,
                    offsets=self.offsets, active_gate_count=self.active_gate_count,
                    gate_masks=[self._gate_mask(level) for level in (2, 3, 4)],
                    score_sum=self.score_sum, score_after_sum=self.score_after_sum,
                    score_visible_sum=self.score_visible_sum,
                    score_observations=self.score_observations, score_steps=self.score_steps,
                    optimizer=self.optimizer.state_dict())

    def restore(self, ckpt: Dict[str, Any]):
        if not ckpt:
            return
        if ckpt.get("prototype_version") != 3 or len(ckpt["grids"]) != 4:
            raise RuntimeError("Four-level resume requires a four-level checkpoint")
        self.grids = nn.ModuleList(list(ckpt["grids"]))
        self.delta = ckpt["delta"]
        self.index = ckpt["index"]
        self.offsets = tuple(ckpt["offsets"])
        self.grids_num = sum(grid.vertex_num for grid in self.grids)
        self.active_gate_count = ckpt["active_gate_count"]
        for level, mask in zip((2, 3, 4), ckpt["gate_masks"]):
            setattr(self, f"gate_mask_l{level}", mask)
        for name in ("score_sum", "score_after_sum", "score_visible_sum", "score_observations"):
            setattr(self, name, ckpt[name])
        self.score_steps = list(ckpt["score_steps"])
        self.optimizer = torch.optim.Adam([self.delta], lr=self.config.lr, eps=1e-15)
        self.optimizer.load_state_dict(ckpt["optimizer"])

    @torch.no_grad()
    def export(self, path: Path, index_encoding: str = "per_row") -> Dict[str, int]:
        """Save active anchors and sparse per-GS blends needed to reconstruct motion.

        ``shared_table`` factors identical child-index tuples within each frame;
        the default ``per_row`` keeps the existing v2 representation.
        """
        if index_encoding not in ("per_row", "shared_table"):
            raise ValueError("index_encoding must be 'per_row' or 'shared_table'")
        path.parent.mkdir(parents=True, exist_ok=True)
        active = [torch.ones(self.grids[0].vertex_num, device=self.delta.device, dtype=torch.bool)]
        for stage in range(3):
            active.append(self._gate_mask(stage + 2)[self.grids[stage + 1].parent])
        active = torch.cat(active)
        remap = torch.full((self.grids_num,), -1, device=self.delta.device, dtype=torch.long)
        remap[active] = torch.arange(int(active.sum()), device=self.delta.device)
        residual = self.delta.detach().clone()
        residual[self.grids[0].vertex_num:, 3] -= 1
        representation = ("nested_gated_four_sparse_v2" if index_encoding == "per_row"
                          else "nested_gated_four_sparse_v3_shared_index")
        arrays = dict(representation=np.array(representation),
                      delta=residual[active].cpu().numpy(),
                      base_index=remap[self.index[0]].int().cpu().numpy(),
                      active_gate_count=np.array(self.active_gate_count, dtype=np.int32))
        for stage in range(3):
            child = self.grids[stage + 1]
            level = stage + 2
            gate_rows = torch.where(self._gate_mask(level)[
                self.grids[stage].p2v.flatten()])[0]
            arrays[f"blend_rows_l{level}"] = gate_rows.int().cpu().numpy()
            index_rows = remap[
                child.blend_indices[gate_rows] + self.offsets[stage + 1]].int().cpu().numpy()
            if index_encoding == "per_row":
                arrays[f"blend_indices_l{level}"] = index_rows
            else:
                table, groups = _factor_child_index_rows(index_rows)
                if not np.array_equal(table[groups], index_rows):
                    raise RuntimeError(f"Shared index table failed reconstruction at level {level}")
                arrays[f"blend_index_table_l{level}"] = table
                arrays[f"blend_index_group_l{level}"] = groups
            arrays[f"blend_weights_l{level}"] = child.blend_weights[gate_rows].cpu().numpy()
        np.savez_compressed(path, **arrays)
        return dict(motion_bytes=path.stat().st_size, active_anchors=int(active.sum()),
                    candidate_anchors=self.grids_num,
                    active_l2_parents=int(self.gate_mask_l2.sum()),
                    active_l3_parents=int(self.gate_mask_l3.sum()),
                    active_l4_parents=int(self.gate_mask_l4.sum()))

    @staticmethod
    def decode_export_motion(path: Path) -> np.ndarray:
        """Reconstruct per-GS 7D motion from a sparse-v2/v3 exported archive."""
        with np.load(path, allow_pickle=False) as packet:
            representation = str(packet["representation"].item())
            if representation not in ("nested_gated_four_sparse_v2",
                                      "nested_gated_four_sparse_v3_shared_index"):
                raise ValueError(f"Unsupported gated-four export: {representation}")
            shared_index = representation == "nested_gated_four_sparse_v3_shared_index"
            delta = packet["delta"]
            base_index = packet["base_index"]
            if delta.ndim != 2 or delta.shape[1] != 7:
                raise ValueError("Exported delta must have shape [anchors, 7]")
            if base_index.ndim != 1 or (base_index.size and
                                        (base_index.min() < 0 or base_index.max() >= len(delta))):
                raise ValueError("Exported base indices are out of range")
            field = delta[base_index].copy()
            active_gate_count = int(packet["active_gate_count"].item())
            if not 0 <= active_gate_count <= 3:
                raise ValueError("Exported active gate count must be in [0, 3]")
            for level in range(2, 2 + active_gate_count):
                rows = packet[f"blend_rows_l{level}"]
                if shared_index:
                    table = packet[f"blend_index_table_l{level}"]
                    groups = packet[f"blend_index_group_l{level}"]
                    if (table.ndim != 2 or groups.ndim != 1 or
                            len(groups) != len(rows) or
                            (groups.size and (groups.min() < 0 or groups.max() >= len(table)))):
                        raise ValueError(f"Invalid shared blend-index table for level {level}")
                    indices = table[groups]
                else:
                    indices = packet[f"blend_indices_l{level}"]
                weights = packet[f"blend_weights_l{level}"]
                if (rows.ndim != 1 or indices.ndim != 2 or indices.shape[0] != len(rows) or
                        indices.shape[1] < 1 or weights.shape != indices.shape):
                    raise ValueError(f"Invalid sparse blend arrays for level {level}")
                if rows.size and (rows.min() < 0 or rows.max() >= len(field) or
                                  np.any(rows[1:] <= rows[:-1])):
                    raise ValueError(f"Exported blend rows for level {level} must be sorted and unique")
                if indices.size and (indices.min() < -1 or indices.max() >= len(delta)):
                    raise ValueError(f"Exported blend indices for level {level} are out of range")
                valid = indices >= 0
                residual = delta[np.maximum(indices, 0)]
                np.copyto(residual, 0, where=~valid[..., None])
                blend = np.einsum("rkg,rk->rg", residual, weights, optimize=True)
                field[rows] += blend
        return field

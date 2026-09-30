import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from absl import logging
from jaxtyping import Float
from torch import Tensor, nn
import time

from .nested import NestedFineGrid, weighted_gradient_diagnostics

from ..utils import average_quaternions, batched_knn, batched_keops_knn, uniform_grid_sampling_optimized
    
class Grid(nn.Module):
    def __init__(
        self,
        size: Tuple[int, int, int],
        xyz_min: Float[Tensor, "3"],
        xyz_max: Float[Tensor, "3"],
    ):
        super().__init__()

        size = size if torch.is_tensor(size) else torch.tensor(size)
        base = torch.tensor([size[1] * size[2], size[2], 1])

        self.register_buffer("size", size)
        self.register_buffer("base", base)
        self.register_buffer("xyz_min", xyz_min)
        self.register_buffer("xyz_max", xyz_max)

        logging.info(f"Set grid min: {xyz_min.tolist()}, max: {xyz_max.tolist()}")

    def normalize(self, xyz: Float[Tensor, "n 3"], clamp: bool = True):
        xyz = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)
        if clamp:
            return xyz.clamp(0, 1)
        return xyz

    def hash(self, xyz: Float[Tensor, "n 3"]) -> torch.Tensor:
        xyz_normed = self.normalize(xyz)
        index = ((xyz_normed * self.size - 0.5).clamp(0).int() * self.base).sum(dim=-1)
        return index
        
class Adaptive_Grid(nn.Module):
    def __init__(
        self,
        vertex_num: int,
        xyz: Float[Tensor, "n 3"]
    ):
        super().__init__()
        self.vertex_num = vertex_num
        t0 = time.time()
        self.vertex_idx = uniform_grid_sampling_optimized(xyz, self.vertex_num)
        t1 = time.time()
        self.vertex = xyz[self.vertex_idx]
        t2 = time.time()
        self.p2v = batched_keops_knn(xyz, self.vertex, 1)
        t3 = time.time()
        logging.info(f"Setup Adaptive_grids: {self.vertex_num}, total_time for fps: {t1-t0}, total time for knn: {t3-t2}")

    def reset_p2v(self, xyz: Float[Tensor, "n 3"]):
        t2 = time.time()
        self.p2v = batched_keops_knn(xyz, self.vertex, 1)
        t3 = time.time()
        logging.info(f"Reset p2v, Adaptive_grids: {self.vertex_num}, total time for knn: {t3-t2}")


@dataclass
class DeformConfig:
    quantile: float = 0.05
    lr: float = 0.0005
    num_stage1_steps: int = 100
    max_gs_per_grid: int = 5
    num_grid_levels: int = 3
    grid_level_ratio: int = 2
    momentum: Optional[float] = 0.6
    densify_interval: int = 40
    densify_grad_threshold: float = 1.5e-4
    opacity_threshold: float = 0.01
    grid_reset_interval: int = 4
    mask_build_step: int = 100
    fine_keep_ratio: float = 0.5
    hierarchy_mode: str = "legacy"
    refinement_mode: str = "gradhet"
    split_step: int = 50
    score_start_step: int = 31
    selection_seed: int = 0
    fine_overlap_ratio: float = 0.15
    gate_step_l2: int = 20
    gate_step_l3: int = 40
    gate_step_l4: int = 60
    gate_score_window: int = 10
    gate_keep_ratio: float = 0.5


class Deformation(nn.Module):
    config: DeformConfig
    grids: List[Grid]
    optimizer: Optional[torch.optim.Optimizer] = None

    def __init__(self, config: DeformConfig):
        super().__init__()
        self.config = config
        self.rotation_activation = torch.nn.functional.normalize
        if config.hierarchy_mode not in {"legacy", "nested"}:
            raise ValueError("hierarchy_mode must be legacy or nested")
        if config.hierarchy_mode == "nested":
            if config.num_grid_levels != 3 or config.grid_level_ratio < 1:
                raise ValueError("nested prototype requires three levels and a positive ratio")
            if config.refinement_mode not in {"gradhet", "random", "delayed_full", "full", "fixed2"}:
                raise ValueError("Unknown refinement_mode")
            if not 1 <= config.score_start_step <= config.split_step < config.num_stage1_steps:
                raise ValueError("Require 1 <= score_start_step <= split_step < num_stage1_steps")
            if not 0 <= config.fine_keep_ratio <= 1:
                raise ValueError("fine_keep_ratio must be in [0, 1]")
            if not 0 <= config.fine_overlap_ratio <= 1:
                raise ValueError("fine_overlap_ratio must be in [0, 1]")
        self.grids = []
        self.grids_num = 0
        # Legacy V0 uses nearest Middle anchors; nested V1 uses explicit membership.
        self.register_buffer("score_sum", None)
        self.register_buffer("score_after_sum", None)
        self.register_buffer("score_visible_sum", None)
        self.register_buffer("score_observations", None)
        self.score_steps = []
        self.register_buffer("middle_parent_of_gs", None)
        self.register_buffer("fine_to_mid", None)
        self.register_buffer("middle_active_mask", None)
        self.register_buffer("fine_active_mask", None)

    def clear_adaptive_masks(self):
        """Start each new frame with the full three-level hierarchy enabled."""
        self.middle_active_mask = None
        self.fine_active_mask = None

    @torch.no_grad()
    def setup_fine_to_mid(self):
        """Use explicit nested parents or nearest-anchor correspondence for V0."""
        if len(self.grids) < 2:
            raise RuntimeError("fine-to-middle mapping requires at least two grid levels")

        if self.config.hierarchy_mode == "nested":
            self.fine_to_mid = self.grids[0].parent
            self.middle_parent_of_gs = self.grids[1].p2v.flatten()
            return

        fine_xyz = self.grids[0].vertex
        middle_xyz = self.grids[1].vertex
        _, fine_to_mid = batched_knn(fine_xyz, middle_xyz, k=1)
        self.fine_to_mid = fine_to_mid.squeeze(-1)

        logging.info(
            "Built fine-to-middle parent map: %d fine anchors -> %d middle anchors",
            fine_xyz.shape[0],
            middle_xyz.shape[0],
        )

    @torch.no_grad()
    def compute_middle_heterogeneity(self) -> Dict[str, Tensor]:
        """Compute within-region fine-translation variance for every middle anchor.

        For middle region j, H_j is the mean squared L2 distance between each
        assigned fine-anchor translation and that region's mean translation.
        Rotation is deliberately excluded in this V0 score.
        """
        if self.fine_to_mid is None:
            raise RuntimeError("Build the fine-to-middle parent map before scoring regions")

        num_fine = self.grids[0].vertex_num
        num_middle = self.grids[1].vertex_num
        parent = self.fine_to_mid
        fine_translation = self.delta[:num_fine, :3].detach()

        count = torch.bincount(parent, minlength=num_middle)
        sum_translation = torch.zeros(
            num_middle,
            3,
            device=fine_translation.device,
            dtype=fine_translation.dtype,
        )
        sum_translation.index_add_(0, parent, fine_translation)
        mean_translation = sum_translation / count.clamp_min(1)[:, None]

        residual_sq = ((fine_translation - mean_translation[parent]) ** 2).sum(dim=-1)
        heterogeneity = torch.zeros(
            num_middle,
            device=fine_translation.device,
            dtype=fine_translation.dtype,
        )
        heterogeneity.index_add_(0, parent, residual_sq)
        heterogeneity = heterogeneity / count.clamp_min(1)

        return {
            "count": count,
            "mean_translation": mean_translation,
            "heterogeneity": heterogeneity,
        }

    @torch.no_grad()
    def select_random_middle_regions(self) -> Dict[str, Any]:
        """Keep the fine hierarchy in a random subset of nonempty middle regions."""
        keep_ratio = self.config.fine_keep_ratio
        if not 0.0 <= keep_ratio <= 1.0:
            raise ValueError(f"fine_keep_ratio must be in [0, 1], got {keep_ratio}")

        stats = self.compute_middle_heterogeneity()
        count = stats["count"]
        heterogeneity = stats["heterogeneity"]
        valid_idx = torch.where(count > 0)[0]
        num_selected = int(valid_idx.numel() * keep_ratio)

        middle_active = torch.zeros_like(count, dtype=torch.bool)
        if num_selected > 0:
            selected_relative = torch.randperm(
                valid_idx.numel(), device=valid_idx.device
            )[:num_selected]
            middle_active[valid_idx[selected_relative]] = True

        fine_active = middle_active[self.fine_to_mid]
        self.middle_active_mask = middle_active
        self.fine_active_mask = fine_active

        valid_h = heterogeneity[valid_idx]
        selected_h = heterogeneity[middle_active]
        inactive_h = heterogeneity[(count > 0) & ~middle_active]
        selected_h_mean = selected_h.mean().item() if selected_h.numel() else 0.0
        inactive_h_mean = inactive_h.mean().item() if inactive_h.numel() else 0.0

        logging.info(
            "Hierarchy statistics at mask construction:\n"
            "  num_fine = %d\n"
            "  num_middle = %d\n"
            "  selected_middle = %d / %d\n"
            "  active_fine = %d / %d = %.2f%%\n"
            "  H mean = %.6g | H median = %.6g | H max = %.6g\n"
            "  Random selected H mean = %.6g\n"
            "  Inactive H mean = %.6g",
            fine_active.numel(),
            middle_active.numel(),
            num_selected,
            valid_idx.numel(),
            fine_active.sum().item(),
            fine_active.numel(),
            100.0 * fine_active.float().mean().item(),
            valid_h.mean().item() if valid_h.numel() else 0.0,
            valid_h.median().item() if valid_h.numel() else 0.0,
            valid_h.max().item() if valid_h.numel() else 0.0,
            selected_h_mean,
            inactive_h_mean,
        )
        return {
            "selection_method": "random",
            "H": heterogeneity,
            "fine_to_mid": self.fine_to_mid,
            "mid_active": middle_active,
            "fine_active": fine_active,
            "fine_xyz": self.grids[0].vertex,
            "middle_xyz": self.grids[1].vertex,
            "count": count,
            "H_mean": valid_h.mean().item() if valid_h.numel() else 0.0,
            "H_median": valid_h.median().item() if valid_h.numel() else 0.0,
            "H_max": valid_h.max().item() if valid_h.numel() else 0.0,
            "selected_H_mean": selected_h_mean,
            "inactive_H_mean": inactive_h_mean,
            "selected_middle": num_selected,
            "valid_middle": valid_idx.numel(),
            "active_fine_ratio": fine_active.float().mean().item(),
        }
        
    @property
    def get_grid_num(self) -> int:
        return self.grids_num
    
    @torch.no_grad()
    def create_grids(self, xyz: Float[Tensor, "n 3"]):
        if self.config.hierarchy_mode == "nested":
            n_fine = math.ceil(xyz.shape[0] / self.config.max_gs_per_grid)
            n_middle = max(1, n_fine // self.config.grid_level_ratio)
            middle = Adaptive_Grid(n_middle, xyz).to(xyz.device)
            coarse = Adaptive_Grid(max(1, n_middle // self.config.grid_level_ratio), xyz).to(xyz.device)
            fine = NestedFineGrid(xyz, middle.p2v.flatten(), n_middle, self.config.grid_level_ratio,
                                  self.config.fine_overlap_ratio)
            grids = [fine, middle, coarse]
            return grids, sum(g.vertex_num for g in grids)
        q, max_gs_per_grid = self.config.quantile, self.config.max_gs_per_grid
        n = math.ceil((xyz.shape[0] / max_gs_per_grid))
        grids, level = [], self.config.num_grid_levels
        num_grids = 0
        while n > 0 and level > 0:
            num_grids += (n)
            grids.append(Adaptive_Grid(n, xyz).to(xyz.device))
            n, level = n // (self.config.grid_level_ratio), level - 1
        return grids, num_grids
    
    def reset_grid(self, xyz: Float[Tensor, "n 3"]):
        if self.config.hierarchy_mode == "nested":
            self.reset_nested(xyz, rebuild=True)
            return
        self.clear_adaptive_masks()
        old_grids = self.grids
        old_delta = self.delta
        self.grids, num_grids = self.create_grids(xyz)
        new_delta = torch.zeros(num_grids, 7, device=xyz.device)
        new_delta[:, 3] = 1
        offset = 0
        
        for level_idx in range(self.config.num_grid_levels):
            og = old_grids[level_idx]
            ng = self.grids[level_idx]
            dist, ng2og = batched_knn(ng.vertex, og.vertex,k=3)
            ng_corr_old_delta = old_delta[ng2og + offset]
            ng_delta3 = ng_corr_old_delta[:,:,:3].mean(dim=1)
            ng_delta4 = average_quaternions(ng_corr_old_delta[:, :, 3:7])
            new_delta[offset:offset + ng.vertex_num] = torch.cat([ng_delta3, ng_delta4], dim=1)
            offset+=ng.vertex_num
            
        self.delta.data.copy_(new_delta)
        
        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
            
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        self.setup_fine_to_mid()
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum
            
        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")

        self.reset_optimizer()

    def reset(self, xyz: Float[Tensor, "n 3"]):
        if self.config.hierarchy_mode == "nested":
            self.reset_nested(xyz, rebuild=False)
            return
        self.clear_adaptive_masks()
        for level_idx in range(self.config.num_grid_levels):
            self.grids[level_idx].reset_p2v(xyz)
            
        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
            
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        self.setup_fine_to_mid()
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum
            
        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")
    

    def setup(self, xyz: Float[Tensor, "n 3"], reset_grid: bool = False):
        self.clear_adaptive_masks()
        if not self.grids or reset_grid:
            self.grids, num_grids = self.create_grids(xyz)
            delta = torch.zeros(num_grids, 7, device=xyz.device)
            delta[:, 3] = 1  # 0 0 0 1 0 0 0 -> x y z q1 q2 q3 q4
            self.register_parameter("delta", nn.Parameter(delta))

        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        self.setup_fine_to_mid()
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum

        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")
        
        if self.config.hierarchy_mode == "nested":
            self.initialize_refinement()
        self.reset_optimizer()

    @torch.no_grad()
    def initialize_refinement(self):
        self.setup_fine_to_mid()
        num_middle = self.grids[1].vertex_num
        self.score_sum = self.delta.new_zeros(num_middle)
        self.score_after_sum = self.delta.new_zeros(num_middle)
        self.score_visible_sum = torch.zeros(num_middle, device=self.delta.device, dtype=torch.long)
        self.score_observations = torch.zeros(num_middle, device=self.delta.device, dtype=torch.long)
        self.score_steps = []
        self.middle_active_mask = torch.full((num_middle,), self.config.refinement_mode == "full",
                                              device=self.delta.device, dtype=torch.bool)
        self.fine_active_mask = self.middle_active_mask[self.fine_to_mid]
        self.delta[:self.grids[0].vertex_num] = 0
        self.delta[:self.grids[0].vertex_num, 3] = 1

    @torch.no_grad()
    def reset_nested(self, xyz, rebuild=False):
        old_grids, old_delta = list(self.grids), self.delta.detach()
        if rebuild:
            self.grids, num_grids = self.create_grids(xyz)
        else:
            self.grids[1].reset_p2v(xyz)
            self.grids[2].reset_p2v(xyz)
            self.grids[0] = NestedFineGrid(xyz, self.grids[1].p2v.flatten(),
                                          self.grids[1].vertex_num, self.config.grid_level_ratio,
                                          self.config.fine_overlap_ratio)
            num_grids = sum(g.vertex_num for g in self.grids)
        delta = xyz.new_zeros(num_grids, 7)
        delta[:, 3] = 1
        old_offset, new_offset = old_grids[0].vertex_num, self.grids[0].vertex_num
        for level in (1, 2):
            old, new = old_grids[level], self.grids[level]
            if rebuild:
                _, match = batched_knn(new.vertex, old.vertex, k=min(3, old.vertex_num))
                values = old_delta[match + old_offset]
                inherited = torch.cat([values[:, :, :3].mean(1), average_quaternions(values[:, :, 3:])], -1)
            else:
                inherited = old_delta[old_offset:old_offset + old.vertex_num]
            if self.config.momentum is not None:
                inherited = inherited * self.config.momentum
            delta[new_offset:new_offset + new.vertex_num] = inherited
            old_offset += old.vertex_num
            new_offset += new.vertex_num
        self.delta = nn.Parameter(delta)
        self.index = torch.stack([self.grids[0].p2v.flatten(),
                                  self.grids[1].p2v.flatten() + self.grids[0].vertex_num,
                                  self.grids[2].p2v.flatten() + self.grids[0].vertex_num + self.grids[1].vertex_num])
        self.grids_num = num_grids
        self.initialize_refinement()
        self.reset_optimizer()

    def collecting_gradients(self, step):
        return (self.config.hierarchy_mode == "nested" and
                self.config.refinement_mode in {"gradhet", "random", "delayed_full"} and
                self.config.score_start_step <= step <= self.config.split_step)

    @torch.no_grad()
    def accumulate_gradient_score(self, gradient, visible, step=None):
        if gradient is None:
            raise RuntimeError("Missing retained per-GS delta_xyz gradient")
        variance, after, count = weighted_gradient_diagnostics(
            gradient.detach(), self.middle_parent_of_gs, visible, self.grids[1].vertex_num,
            self.grids[0])
        observed = count >= 2
        self.score_sum[observed] += variance[observed]
        self.score_after_sum[observed] += after[observed]
        self.score_visible_sum[observed] += count[observed]
        self.score_observations += observed.long()
        if step is not None:
            self.score_steps.append(step)
        if observed.any():
            logging.info("Gradient diagnostic step %s: observed Middle %d, H_before %.6g, "
                         "H_after %.6g, H_reduction %.6g", step, observed.sum(),
                         variance[observed].mean(), after[observed].mean(),
                         (variance - after)[observed].mean())

    @torch.no_grad()
    def activate_refinement(self):
        score = self.score_sum / self.score_observations.clamp_min(1)
        after = self.score_after_sum / self.score_observations.clamp_min(1)
        count = torch.bincount(self.middle_parent_of_gs, minlength=score.numel())
        valid = torch.where(count > 0)[0]
        k = int(valid.numel() * self.config.fine_keep_ratio)
        child_count = torch.bincount(self.fine_to_mid, minlength=score.numel())
        eligible = (count > 0) & (child_count >= 2) & (self.score_observations > 0) & (score > 0)
        mode = self.config.refinement_mode
        self.middle_active_mask.zero_()
        if mode in {"full", "delayed_full"}:
            self.middle_active_mask[valid] = True
        elif mode in {"gradhet", "random"}:
            if mode == "gradhet":
                # A budget upper bound: missing/zero scores and unsplittable parents
                # never get activated just to fill the requested percentage.
                candidates = torch.where(eligible)[0]
                selected = candidates[torch.argsort(score[candidates], descending=True, stable=True)[:k]]
            else:
                generator = torch.Generator(device=score.device).manual_seed(self.config.selection_seed)
                selected = valid[torch.randperm(valid.numel(), device=score.device, generator=generator)[:k]]
            self.middle_active_mask[selected] = True
        self.fine_active_mask = self.middle_active_mask[self.fine_to_mid]
        overlap = self.grids[0].overlap_statistics()
        logging.info("Nested %s: active Middle %d/%d, Fine %d/%d, observed Middle %d",
                     mode, self.middle_active_mask.sum(), valid.numel(), self.fine_active_mask.sum(),
                     self.fine_active_mask.numel(), (self.score_observations > 0).sum())
        logging.info("Fine overlap: transition width %.1f%% of anchor separation, shared GS %d/%d (%.2f%%)",
                     100 * self.config.fine_overlap_ratio, overlap['shared_gs'], count.sum(),
                     100 * overlap['overlap_gs_ratio'])
        return dict(prototype_version=2, selection_method=mode, H=score,
                    H_before=score, H_after=after, H_reduction=score - after,
                    diagnostic_valid=self.score_observations > 0, eligible=eligible,
                    score_visible_sum=self.score_visible_sum, score_steps=self.score_steps,
                    overlap_ratio=self.config.fine_overlap_ratio, **overlap,
                    child_count=child_count, score_observations=self.score_observations,
                    count=count, fine_to_mid=self.fine_to_mid, middle_parent_of_gs=self.middle_parent_of_gs,
                    mid_active=self.middle_active_mask, fine_active=self.fine_active_mask,
                    fine_xyz=self.grids[0].vertex, middle_xyz=self.grids[1].vertex)

    def reg_loss(self):
        identity = torch.zeros_like(self.delta[:1, :])
        identity[:, 3] = 1
        if self.config.hierarchy_mode == "nested":
            # Inactive candidate Fine rows receive neither photo nor reg updates.
            active = torch.cat([self.fine_active_mask, torch.ones(
                self.delta.shape[0] - self.fine_active_mask.numel(),
                device=self.delta.device, dtype=torch.bool)])
            # A fixed denominator avoids changing the parent penalty merely
            # because zero-residual Fine candidates become active.
            return ((self.delta - identity).abs() * active[:, None]).mean()
        return (self.delta - identity).abs().mean()

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, eps=1e-15
        )

    def capture(self) -> Dict[str, Any]:
        if not self.grids:
            return {}
        state = dict(delta=self.delta, index=self.index)
        if self.config.hierarchy_mode == "nested":
            state.update(prototype_version=2, grids=self.grids, optimizer=self.optimizer.state_dict(),
                         score_steps=self.score_steps,
                         **{name: getattr(self, name) for name in
                            ("fine_to_mid", "middle_parent_of_gs", "middle_active_mask", "fine_active_mask",
                             "score_sum", "score_after_sum", "score_visible_sum", "score_observations")})
        return state

    def restore(self, ckpt: Dict[str, Any]):
        if "delta" in ckpt:
            logging.info("Restore deformation from checkpoint")
            self.register_buffer("index", ckpt["index"])
            self.register_parameter("delta", ckpt["delta"])
            if self.config.hierarchy_mode == "nested":
                if "grids" not in ckpt:
                    raise RuntimeError("Nested resume needs a nested checkpoint; a shared first-frame checkpoint is supported")
                if ckpt.get("prototype_version") != 2:
                    raise RuntimeError("This prototype needs a v2 overlap/residual checkpoint or the shared first-frame checkpoint")
                self.grids = ckpt["grids"]
                self.score_steps = ckpt["score_steps"]
                self.grids_num = sum(g.vertex_num for g in self.grids)
                for name in ("fine_to_mid", "middle_parent_of_gs", "middle_active_mask", "fine_active_mask",
                             "score_sum", "score_after_sum", "score_visible_sum", "score_observations"):
                    setattr(self, name, ckpt[name])
            self.reset_optimizer()
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])

    def forward(self, xyz: Float[Tensor, "n 3"], normalized: bool = False):
        if self.config.hierarchy_mode == "nested":
            fine = self.grids[0]
            identity = self.delta.new_zeros(7)
            identity[3] = 1
            residual = self.delta[fine.blend_indices] - identity
            fine_delta = (residual * fine.blend_weights[:, :, None]).sum(1)
            gate = self.middle_active_mask[self.middle_parent_of_gs].to(self.delta.dtype)[:, None]
            delta = self.delta[self.index[1:]].sum(0) + fine_delta * gate
            return delta[:, :3].contiguous(), self.rotation_activation(delta[:, 3:].contiguous())
        delta_levels = self.delta[self.index]
        fine_delta = delta_levels[0]

        if self.fine_active_mask is not None:
            fine_anchor_idx = self.index[0]
            gate = self.fine_active_mask[fine_anchor_idx].to(
                dtype=delta_levels.dtype
            ).unsqueeze(-1)
            fine_delta = fine_delta * gate

        delta = fine_delta + delta_levels[1:].sum(dim=0)
        delta_xyz = delta[:, :3].contiguous()
        delta_rot = delta[:, 3:].contiguous()
        return delta_xyz, self.rotation_activation(delta_rot)

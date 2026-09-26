"""Bounded S²GS voxel densification for the V2 third training stage.

New points stay in S²GS's GaussianModel.  In particular, residual-decoder
baselines, root gates and accumulated motion are extended with the point set.
"""

import torch

from scene.decoders import DecoderIdentity, LatentDecoderRes


def eligible_birth_mask(scores, observations, target_levels, opt_lod, fork):
    """S²GS-style evidence threshold for finer-level dynamic births."""
    min_observations = (opt_lod.update_interval
                        * opt_lod.success_threshold * 0.5)
    level_threshold = (opt_lod.densify_grad_threshold
                       * (float(fork) **
                          (opt_lod.update_ratio * target_levels.float())))
    return ((observations > min_observations)
            & torch.isfinite(scores) & (scores >= level_threshold))


@torch.no_grad()
def add_dynamic_gaussians(gaussians, frame_idx, opt_lod,
                          max_new_points=128):
    """Grow high-gradient Gaussians on the S²GS voxel grid; return old count.

    The current Flame Steak configuration has one offset per anchor.  A hard
    per-frame cap prevents uncontrolled growth across the 299 dynamic frames.
    """
    if gaussians.n_offsets != 1:
        raise ValueError("V2 growth currently requires n_offsets=1")
    if gaussians.shared_motion is not None:
        raise RuntimeError("Commit shared motion before adding Gaussians")

    old_count = gaussians.get_anchor.shape[0]
    if max_new_points <= 0:
        return old_count, 0
    scores = (gaussians.offset_gradient_accum
              / gaussians.offset_denom.clamp_min(1)).reshape(-1)
    target_levels = torch.clamp(
        gaussians.get_level.reshape(-1).long() + 1,
        max=gaussians.levels - 1)
    # S²GS run_densify requires enough observations and anchor_growing uses
    # a level-dependent gradient threshold.  A top-K alone forces births even
    # when every candidate has negligible evidence.
    eligible = eligible_birth_mask(
        scores, gaussians.offset_denom.reshape(-1),
        target_levels, opt_lod, gaussians.fork)
    count = min(int(eligible.sum().item()), int(max_new_points))
    if count == 0:
        return old_count, 0
    parent = torch.topk(scores.masked_fill(~eligible, -torch.inf), count).indices

    decoded = gaussians.get_decoded_atts
    local_center = (gaussians.get_anchor[parent]
                    + gaussians.get_offset[parent]
                    * gaussians.get_scaling[parent, :3])
    # Reuse S²GS's finer voxel placement and its camera-visibility weeding.
    new_level = target_levels[parent]
    cell_size = gaussians.voxel_size / (
        float(gaussians.fork) ** new_level.float())
    voxel = torch.floor(
        (local_center - gaussians.init_pos) / cell_size.unsqueeze(1)
        - gaussians.padding).int()
    # Match S²GS anchor_growing's voxel uniqueness and overlap check.  Without
    # this, the same high-gradient voxel can receive another point each frame.
    key = torch.cat((new_level.int().unsqueeze(1), voxel), dim=1)
    unique_key, inverse = torch.unique(key, dim=0, return_inverse=True)
    first = torch.full((unique_key.shape[0],), count, device=parent.device,
                       dtype=torch.long)
    first.scatter_reduce_(0, inverse,
                          torch.arange(count, device=parent.device),
                          reduce="amin")
    parent, new_level, cell_size, voxel = (
        value[first] for value in (parent, new_level, cell_size, voxel))
    keep = torch.ones(parent.numel(), dtype=torch.bool, device=parent.device)
    existing_levels = gaussians.get_level.reshape(-1).long()
    existing_anchors = gaussians.get_anchor
    for level in torch.unique(new_level).tolist():
        candidate_mask = new_level == level
        existing_size = gaussians.voxel_size / (float(gaussians.fork) ** level)
        existing_grid = torch.floor(
            (existing_anchors[existing_levels == level] - gaussians.init_pos)
            / existing_size - gaussians.padding).int()
        keep[candidate_mask] = ~gaussians.get_remove_duplicates(
            existing_grid, voxel[candidate_mask])
    parent, new_level, cell_size, voxel = (
        value[keep] for value in (parent, new_level, cell_size, voxel))
    if parent.numel() == 0:
        return old_count, 0
    new_anchor = (voxel * cell_size.unsqueeze(1)
                  + gaussians.init_pos
                  + gaussians.padding * cell_size.unsqueeze(1))
    new_anchor, new_level, _, keep, _ = gaussians.weed_out(
        new_anchor, new_level.float(), gaussians.root_indices[parent])
    parent = parent[keep]
    count = parent.numel()
    if count == 0:
        return old_count, 0

    baseline = {
        "anchor": new_anchor,
        "offset": torch.zeros_like(new_anchor),
        "f_dc": decoded["f_dc"][parent].reshape(count, -1).clone(),
        "f_rest": decoded["f_rest"][parent].reshape(count, -1).clone(),
        "sc": decoded["sc"][parent].clone(),
        "rot": decoded["rot"][parent].clone(),
        "op": torch.logit(torch.full_like(decoded["op"][parent], 0.05)),
    }
    extensions = {}
    for name, value in baseline.items():
        decoder = gaussians.latent_decoders[name]
        if isinstance(decoder, LatentDecoderRes):
            extensions[name] = torch.zeros(
                (count, gaussians._latents[name].shape[1]),
                device=value.device, dtype=gaussians._latents[name].dtype)
            decoder.decoded_att = torch.cat(
                (decoder.decoded_att, value.reshape(count, -1)), dim=0)
            if decoder.gates is not None:
                decoder.gates = torch.cat(
                    (decoder.gates, torch.ones_like(decoder.gates[:count])), dim=0)
        elif isinstance(decoder, DecoderIdentity):
            extensions[name] = value.reshape(count, *gaussians._latents[name].shape[1:])
        else:
            # A non-residual quantized decoder (f_rest in this config) is
            # initialized from its parent's latent code.
            extensions[name] = gaussians._latents[name][parent].clone()

    updated = gaussians.cat_tensors_to_optimizer(extensions)
    for name, parameter in updated.items():
        gaussians._latents[name] = parameter

    for name in gaussians.param_names:
        for history in (gaussians.prev_atts, gaussians.prev_atts_initial):
            if history[name] is not None:
                shape = (count, *history[name].shape[1:])
                history[name] = torch.cat(
                    (history[name], baseline[name].reshape(shape)), dim=0)
        if gaussians.prev_latents[name] is not None:
            history_shape = (count, *gaussians.prev_latents[name].shape[1:])
            gaussians.prev_latents[name] = torch.cat(
                (gaussians.prev_latents[name],
                 gaussians.prev_latents[name].new_zeros(history_shape)), dim=0)

    previous_offset_count = gaussians.offset_before.shape[0]
    gaussians.offset_before = torch.cat(
        (gaussians.offset_before, baseline["offset"]), dim=0)
    gaussians.mapping = torch.cat((
        gaussians.mapping,
        torch.arange(previous_offset_count, previous_offset_count + count,
                     device=parent.device)), dim=0)

    gaussians._level = torch.cat((
        gaussians._level, new_level.reshape(count, 1)), dim=0)
    gaussians._extra_level = torch.cat(
        (gaussians._extra_level, gaussians._extra_level[parent]), dim=0)
    gaussians.root_indices = torch.cat(
        (gaussians.root_indices, gaussians.root_indices[parent]), dim=0)
    for name in ("offset_gradient_accum", "offset_denom", "opacity_accum",
                 "anchor_demon", "max_radii2D", "infl_accum", "infl_denom"):
        value = getattr(gaussians, name)
        if value.shape[0] == old_count:
            setattr(gaussians, name, torch.cat(
                (value, value.new_zeros((count, *value.shape[1:]))), dim=0))

    if gaussians.motion_base_translation is not None:
        gaussians.motion_base_translation = torch.cat((
            gaussians.motion_base_translation,
            gaussians.motion_base_translation[parent]), dim=0)
        gaussians.motion_base_quaternion = torch.cat((
            gaussians.motion_base_quaternion,
            gaussians.motion_base_quaternion[parent]), dim=0)

    if gaussians.gate_atts is not None:
        roots, inverse = torch.unique(
            gaussians.root_indices, sorted=True, return_inverse=True)
        if roots.numel() != gaussians.gate_atts.num_gates:
            raise RuntimeError("V2 growth changed the number of root gates")
        gaussians.gate_atts.inverse_indices = inverse

    # Physical birth attributes plus the root/level reference, float32-based
    # estimate.  The raw .pt payload has additional serialization overhead.
    gaussians.current_frame_birth_bits = count * (44 * 32 + 64)
    return old_count, count


def freeze_existing_gaussians(gaussians):
    """Train only new latent rows during stage 3, not shared decoders/gates."""
    for decoder in gaussians.latent_decoders.values():
        decoder.requires_grad_(False)
    if gaussians.gate_atts is not None:
        gaussians.gate_atts.log_alphas.requires_grad_(False)
        gaussians.gate_atts.eval()


def restore_shared_decoders(gaussians):
    """Undo stage-3 freezing before the next frame applies its own masks."""
    for decoder in gaussians.latent_decoders.values():
        decoder.requires_grad_(True)


@torch.no_grad()
def mask_existing_gradients(gaussians, old_count):
    """Clear both gradients and Adam momentum on pre-existing latent rows."""
    for parameter in gaussians._latents.values():
        if parameter.grad is None:
            continue
        parameter.grad[:old_count] = 0
        state = gaussians.optimizer.state.get(parameter)
        if state is not None:
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][:old_count] = 0


@torch.no_grad()
def frame_birth_payload(gaussians, old_count):
    """Record the optimized new Gaussians for this frame's experiment output."""
    attributes = gaussians.get_decoded_atts
    return {
        "old_count": old_count,
        "new_count": gaussians.get_anchor.shape[0] - old_count,
        "root_indices": gaussians.root_indices[old_count:].cpu(),
        "level": gaussians._level[old_count:].cpu(),
        "attributes": {
            name: value[old_count:].cpu()
            for name, value in attributes.items()
        },
        "motion_translation": (
            None if gaussians.motion_base_translation is None else
            gaussians.motion_base_translation[old_count:].cpu()),
        "motion_quaternion": (
            None if gaussians.motion_base_quaternion is None else
            gaussians.motion_base_quaternion[old_count:].cpu()),
        "estimated_bits": gaussians.current_frame_birth_bits,
    }

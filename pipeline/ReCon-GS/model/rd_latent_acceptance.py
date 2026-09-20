"""Stateless, bounded backtracking for SGD envelope-latent proposals.

Only the evaluated batch's rendered distortion plus actual payload bits is
checked. No cross-batch or global guarantee, and no Adam state is supported.
"""
from dataclasses import dataclass
import math
import numpy as np

from .rd_tree_projection import encode_projection


@dataclass
class LatentAcceptance:
    latent: np.ndarray
    projection: object
    accepted: bool
    scale: float | None
    objective: float
    distortion: float
    distortion_evaluations: int
    projection_calls: int
    trials: list


class ObservationLossCache:
    """Single immutable-packet/ordered-observation binding, never a global cache."""
    def __init__(self):
        self.key = None
        self.distortion = None

    def get(self, packet, observations):
        return self.distortion if self.key == (bytes(packet), tuple(observations)) else None

    def put(self, packet, observations, distortion):
        if not math.isfinite(distortion):
            raise ValueError('Cannot cache a nonfinite distortion')
        self.key = (bytes(packet), tuple(observations))
        self.distortion = float(distortion)


def accept_latent_step(latent, direction, state, tree, rate_weight, scales,
                       project_fn, distortion_fn, current_distortion):
    """Return first non-worsening proposal; never mutate caller's latent.

    project_fn maps an encoder latent to its exact quantized tree projection.
    distortion_fn must evaluate the SAME observations as current_distortion.
    Exact packet repeats reuse the current distortion. Ties (including same
    packet) accept latent progress; rejected proposals leave latent unchanged.
    All new projection and distortion calls are counted, even rejected ones.
    """
    latent = np.asarray(latent, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    scales = tuple(float(s) for s in scales)
    if latent.shape != direction.shape or not np.isfinite(latent).all() or not np.isfinite(direction).all():
        raise ValueError('Expected finite, matching latent and direction')
    if not scales or any(not math.isfinite(s) or not 0 < s <= 1 for s in scales):
        raise ValueError('Expected bounded positive backtracking scales')
    if not math.isfinite(rate_weight) or rate_weight < 0 or not math.isfinite(current_distortion):
        raise ValueError('Invalid rate or current distortion')
    base_objective = current_distortion + rate_weight * state.payload_bits
    cache = {encode_projection(state, tree): current_distortion}
    trials, evaluations = [], 0
    for scale in scales:
        candidate_latent = latent + scale * direction
        candidate = project_fn(candidate_latent)
        packet = encode_projection(candidate, tree)
        cached = packet in cache
        if not cached:
            cache[packet] = float(distortion_fn(candidate.field))
            evaluations += 1
        distortion = cache[packet]
        objective = distortion + rate_weight * candidate.payload_bits
        accepted = math.isfinite(objective) and objective <= base_objective
        trials.append(dict(scale=scale, payload_bits=candidate.payload_bits,
                           objective=objective if math.isfinite(objective) else None,
                           cached_distortion=cached, accepted=accepted))
        if accepted:
            return LatentAcceptance(candidate_latent, candidate, True, scale, objective, distortion,
                                    evaluations, len(trials), trials)
    return LatentAcceptance(latent.copy(), state, False, None, base_objective, current_distortion,
                            evaluations, len(trials), trials)

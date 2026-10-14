"""Unactivated CPU translation initial values from two decoded transitions.

This is an optimizer initializer, not evidence that the current frame moves.
It neither writes history/observation state nor supplies regional or rotational
motion. Pending IDs must be checked against actual current observations before
the future integration may retain their initialized motion.
"""

from dataclasses import dataclass
import math

import numpy as np


MIN_DIRECTION_COSINE = .5
MAX_AMPLITUDE_RATIO = 2.


@dataclass
class CausalTranslationInitialization:
    translation: np.ndarray
    predicted_mask: np.ndarray
    pending_predicted_ids: np.ndarray
    metrics: dict


def _ids(value):
    value = np.asarray(value)
    if (value.ndim != 1 or value.dtype.kind not in 'iu' or np.any(value < 0) or
            np.any(value > np.iinfo(np.int64).max) or len(np.unique(value)) != len(value)):
        raise ValueError('stable GS IDs must be unique nonnegative int64-range integers')
    return value.astype(np.int64, copy=False)


def _align(record, current_ids):
    stored = _ids(record['ids'])
    source, target = (np.asarray(record[key]) for key in ('source', 'target'))
    observed = np.asarray(record['observed'])
    if (source.shape != (len(stored), 3) or target.shape != source.shape or
            observed.shape != stored.shape or observed.dtype.kind != 'b' or
            not np.isfinite(source[observed]).all() or not np.isfinite(target[observed]).all()):
        raise ValueError('decoded history needs aligned finite observed positions and boolean masks')
    order = np.argsort(stored, kind='stable')
    locations = np.searchsorted(stored[order], current_ids)
    present = locations < len(stored)
    present[present] &= stored[order[locations[present]]] == current_ids[present]
    matched = np.flatnonzero(present)
    rows = order[locations[present]]
    valid = np.zeros(len(current_ids), bool)
    valid[matched] = observed[rows]
    displacement = np.zeros((len(current_ids), 3), np.float64)
    # Do not read masked values, including a missing birth's placeholder NaNs.
    observed_rows = rows[observed[rows]]
    displacement[matched[observed[rows]]] = (target[observed_rows].astype(np.float64) -
                                           source[observed_rows].astype(np.float64))
    return present, valid, displacement


def propose_causal_translation_initialization(history, current_ids, current_frame,
                                              translation_step, extent,
                                              clustering_config, *, gs_xyz_step_extent):
    """Use the latest complete WORLD displacement after conservative past gates.

    The two records must end at current_frame-2 and current_frame-1. Both must
    observe this exact GS ID. Integer GS displacement norm must exceed the
    configured static_translation_extent / gs_xyz_step_extent ratio. Actual
    float32 GS steps round the displacement, without redefining this threshold.
    Direction and symmetric amplitude ratio use the real
    decoded WORLD vectors, not locally encoded residuals or future targets.
    """
    ids = _ids(current_ids)
    if (isinstance(current_frame, bool) or not isinstance(current_frame, (int, np.integer)) or current_frame < 1):
        raise ValueError('current_frame must be a positive integer')
    step = np.asarray(translation_step, np.float64)
    threshold_extent = clustering_config.static_translation_extent
    if (step.shape != (3,) or not np.isfinite(step).all() or np.any(step <= 0) or
            not math.isfinite(extent) or extent <= 0 or
            isinstance(threshold_extent, bool) or not math.isfinite(threshold_extent) or threshold_extent <= 0 or
            isinstance(gs_xyz_step_extent, bool) or not math.isfinite(gs_xyz_step_extent) or gs_xyz_step_extent <= 0):
        raise ValueError('prediction needs positive GS translation steps, extent and existing motion threshold')
    if (not np.all(step == step[0]) or not np.allclose(step, gs_xyz_step_extent * extent,
            rtol=8 * np.finfo(np.float32).eps, atol=0)):
        raise ValueError('prediction requires isotropic actual GS steps matching the configured step within float32 precision')
    translation = np.zeros((len(ids), 3), np.float32)
    predicted = np.zeros(len(ids), bool)
    threshold = float(threshold_extent * extent)
    threshold_units = float(threshold_extent / gs_xyz_step_extent)
    counts = dict(no_latest_transition_id=0, no_older_transition_id=0,
        latest_unobserved=0, older_unobserved=0, latest_not_moving=0,
        older_not_moving=0, direction_inconsistent=0, amplitude_inconsistent=0,
        GS_quantization_overflow=0, unavailable_history=0)
    metrics = dict(current_frame=int(current_frame), current_gs=len(ids), predicted_gs=0,
        decoded_history_frames=[], source_latest_displacement_frame=None,
        motion_threshold_world=threshold, gs_translation_steps=step.tolist(),
        configured_gs_xyz_step_extent=float(gs_xyz_step_extent), moving_threshold_GS_units=threshold_units,
        direction_cosine_min=MIN_DIRECTION_COSINE, symmetric_amplitude_ratio_max=MAX_AMPLITUDE_RATIO,
        motion_evidence='two consecutive observed decoded transitions; full WORLD displacement rounded by actual GS steps, integer norm compared with configuration ratio',
        observation_scope='saved renderer visibility/radii proxy; not guaranteed unoccluded pixel contribution',
        prediction_scope='latest complete WORLD target-source translation in current GS row order',
        initializer_only=True, current_evidence=False, observations_written=False, history_written=False,
        rotation_or_region_prediction=False, rejection_counts=counts, rejection_reason=None,
        pending_id_order='sorted stable IDs; future current-observation fallback must align IDs, not assume rows',
        future_fallback='before structure: pending IDs without actual current observations -> zero GS translation and clear Adam row moments')

    def result(reason=None):
        metrics['rejection_reason'] = reason
        metrics['predicted_gs'] = int(predicted.sum())
        return CausalTranslationInitialization(translation, predicted,
            np.sort(ids[predicted]).copy(), metrics)

    records = history.records
    frames = [record['frame_index'] for record in records]
    if any(isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or frame < 1 for frame in frames):
        raise ValueError('decoded history frames must be positive integer target frames')
    if list(frames) != sorted(set(frames)):
        raise ValueError('decoded history target frames must be strictly chronological')
    metrics['decoded_history_frames'] = [int(frame) for frame in frames[-2:]]
    if current_frame == 1:
        counts['unavailable_history'] = len(ids)
        return result('first_dynamic_frame')
    if any(frame >= current_frame for frame in frames):
        counts['unavailable_history'] = len(ids)
        return result('history_not_strictly_past')
    if len(records) < 2:
        counts['unavailable_history'] = len(ids)
        return result('fewer_than_two_decoded_transitions')
    if frames[-2:] != [current_frame - 2, current_frame - 1]:
        counts['unavailable_history'] = len(ids)
        return result('history_not_latest_two_consecutive_transitions')
    metrics['source_latest_displacement_frame'] = int(frames[-1])
    older_present, older_seen, older = _align(records[-2], ids)
    latest_present, latest_seen, latest = _align(records[-1], ids)
    remaining = np.ones(len(ids), bool)

    def reject(reason, rejected):
        selected = remaining & rejected
        counts[reason] = int(selected.sum())
        remaining[selected] = False

    reject('no_latest_transition_id', ~latest_present)
    reject('no_older_transition_id', ~older_present)
    reject('latest_unobserved', ~latest_seen)
    reject('older_unobserved', ~older_seen)
    older_norm, latest_norm = np.linalg.norm(older, axis=1), np.linalg.norm(latest, axis=1)
    older_integer_norm = np.linalg.norm(np.rint(older / step), axis=1)
    latest_integer_norm = np.linalg.norm(np.rint(latest / step), axis=1)
    reject('latest_not_moving', latest_integer_norm <= threshold_units)
    reject('older_not_moving', older_integer_norm <= threshold_units)
    denominator = older_norm * latest_norm
    cosine = np.divide(np.einsum('ij,ij->i', older, latest), denominator,
                       out=np.zeros(len(ids)), where=denominator > 0)
    # Roundoff guards only: exact mathematical boundaries .5 and 2 remain
    # inclusive without adding a tunable directional/magnitude tolerance.
    epsilon = 16 * np.finfo(np.float64).eps
    reject('direction_inconsistent', cosine < MIN_DIRECTION_COSINE - epsilon)
    small, large = np.minimum(older_norm, latest_norm), np.maximum(older_norm, latest_norm)
    ratio = np.divide(large, small, out=np.full(len(ids), np.inf), where=small > 0)
    reject('amplitude_inconsistent', ratio > MAX_AMPLITUDE_RATIO * (1 + epsilon))
    symbols = np.rint(latest.astype(np.float32).astype(np.float64) / step)
    reject('GS_quantization_overflow', np.any(np.abs(symbols) > 32767, axis=1))
    predicted[:] = remaining
    translation[predicted] = latest[predicted].astype(np.float32)
    if predicted.any():
        metrics['predicted_displacement_p50_world'] = float(np.quantile(latest_norm[predicted], .5))
        metrics['predicted_displacement_p90_world'] = float(np.quantile(latest_norm[predicted], .9))
        metrics['supported_cosine_min'] = float(cosine[predicted].min())
        metrics['supported_symmetric_amplitude_ratio_max'] = float(ratio[predicted].max())
    return result()

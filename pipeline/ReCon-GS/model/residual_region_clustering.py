"""Standard sklearn HDBSCAN on per-GS motion in one reference frame."""

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .shared_motion_clustering import motion_features


@dataclass
class ResidualHdbscanConfig:
    feature_mode: str = 'se3_affine'
    min_cluster_size: int = 128
    min_samples: int = 16
    min_observations: int = 1
    translation_scale_extent: float = 4e-4
    rotation_scale_rad: float = 0.004
    static_translation_extent: float = 4e-4
    static_rotation_rad: float = 0.004
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = 'eom'
    allow_single_cluster: bool = True
    leaf_size: int = 40
    n_jobs: int = 1

    def validate(self):
        if self.feature_mode not in ('se3_affine', 'point_displacement'):
            raise ValueError('hdbscan.feature_mode must be se3_affine or point_displacement')
        for name in ('min_cluster_size', 'min_samples', 'min_observations', 'leaf_size', 'n_jobs'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'hdbscan.{name} must be a positive integer')
        if self.min_cluster_size < 2:
            raise ValueError('HDBSCAN min_cluster_size must be at least two')
        for name in ('translation_scale_extent', 'rotation_scale_rad',
                     'static_translation_extent', 'static_rotation_rad'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'hdbscan.{name} must be finite and positive')
        if (not math.isfinite(self.cluster_selection_epsilon) or
                self.cluster_selection_epsilon < 0 or
                self.cluster_selection_method not in ('eom', 'leaf') or
                not isinstance(self.allow_single_cluster, bool)):
            raise ValueError('Invalid HDBSCAN cluster selection settings')


def _motion_region_inputs(source, target, relative_quaternion, observations, extent, config):
    """Return standard cluster labels and separate static/unknown categories.

    Affine mode assumes GS orientation updates are physical rigid rotations.
    Displacement mode uses center motion only, since covariance orientation is
    not a reliable object-rotation observation. Candidate groups still undergo
    shared SE(3) fitting and sparse-support checks in the tree builder. Neither
    mode includes XYZ as a spatial clustering feature.
    Labels: -2 near-identity candidates in the selected feature space,
    -3 unobserved, -1 HDBSCAN noise,
    >=0 motion clusters. Noise is NOT a stationary-motion assertion.
    """
    config.validate()
    source, target, q = (np.asarray(value, dtype=np.float64)
                         for value in (source, target, relative_quaternion))
    observations = np.asarray(observations)
    if (source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape or
            q.shape != (len(source), 4) or observations.shape != (len(source),) or
            observations.dtype.kind not in 'iu' or np.any(observations < 0) or
            not math.isfinite(extent) or extent <= 0 or
            not all(np.isfinite(value).all() for value in (source, target, q)) or
            np.any(np.linalg.norm(q, axis=1) == 0)):
        raise ValueError('Invalid residual-motion clustering input')
    if config.feature_mode == 'point_displacement':
        features = (target - source) / (config.translation_scale_extent * extent)
    else:
        features, _ = motion_features(source, target, q,
            config.translation_scale_extent * extent, config.rotation_scale_rad)
    angle = Rotation.from_quat(q[:, [1, 2, 3, 0]]).magnitude()
    stationary = (np.linalg.norm(target - source, axis=1) <=
                  config.static_translation_extent * extent)
    if config.feature_mode == 'se3_affine':
        stationary &= angle <= config.static_rotation_rad
    observed = observations >= config.min_observations
    labels = np.full(len(source), -3, dtype=np.int64)
    labels[observed & stationary] = -2
    selected = np.flatnonzero(observed & ~stationary)
    labels[selected] = -1
    return features, selected, labels


def hdbscan_motion_regions(source, target, relative_quaternion, observations, extent, config):
    features, selected, labels = _motion_region_inputs(
        source, target, relative_quaternion, observations, extent, config)
    if len(selected) >= max(config.min_cluster_size, config.min_samples):
        from sklearn.cluster import HDBSCAN

        estimator = HDBSCAN(min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
            cluster_selection_method=config.cluster_selection_method,
            allow_single_cluster=config.allow_single_cluster,
            algorithm='kd_tree', leaf_size=config.leaf_size, n_jobs=config.n_jobs,
            copy=True)
        labels[selected] = estimator.fit_predict(features[selected])
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels[labels >= 0])]
    groups.sort(key=lambda rows: int(rows[0]))
    return groups, labels


@dataclass
class ResidualGmmConfig:
    components: list = field(default_factory=lambda: [4, 8, 16, 32])
    min_cluster_size: int = 128
    max_fit_points: int = 30000
    max_iter: int = 100
    n_init: int = 1
    random_seed: int = 0
    tol: float = 0.001
    reg_covar: float = 0.0001

    def validate(self):
        if not self.components or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                                      for k in self.components):
            raise ValueError('gmm.components needs positive integer BIC candidates')
        if list(self.components) != sorted(set(self.components)):
            raise ValueError('gmm.components must be distinct and increasing')
        for name in ('min_cluster_size', 'max_fit_points', 'max_iter', 'n_init'):
            value = getattr(self, name)
            minimum = 2 if name in ('min_cluster_size', 'max_fit_points') else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'gmm.{name} must be a positive supported integer')
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError('gmm.random_seed must be nonnegative')
        for name in ('tol', 'reg_covar'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'gmm.{name} must be finite and positive')


def gmm_motion_regions(source, target, relative_quaternion, observations, extent,
                       feature_config, config):
    """Standard full-covariance GMM; BIC selects K on one deterministic sample.

    Component search is part of the clustering algorithm, not separate scene
    training. Small components keep their individual motion as unsupported
    groups; useful shared SE(3) is checked by the existing tree builder.
    """
    from sklearn.mixture import GaussianMixture

    config.validate()
    features, selected, labels = _motion_region_inputs(
        source, target, relative_quaternion, observations, extent, feature_config)
    metrics = dict(gmm_selected_components=0, gmm_fit_points=0, gmm_converged=1)
    if len(selected) < config.min_cluster_size:
        return [], labels, metrics
    values = features[selected]
    random = np.random.default_rng(config.random_seed)
    fit = (values[np.sort(random.choice(len(values), config.max_fit_points, replace=False))]
           if len(values) > config.max_fit_points else values)
    best, best_bic = None, math.inf
    for components in config.components:
        if components > len(fit):
            continue
        model = GaussianMixture(n_components=components, covariance_type='full',
            max_iter=config.max_iter, n_init=config.n_init, random_state=config.random_seed,
            tol=config.tol, reg_covar=config.reg_covar).fit(fit)
        bic = model.bic(fit)
        if math.isfinite(bic) and bic < best_bic:
            best, best_bic = model, bic
    if best is None:
        raise ValueError('No valid GMM component candidate for the observed motion')
    predicted = best.predict(values)
    counts = np.bincount(predicted, minlength=best.n_components)
    labels[selected] = np.where(counts[predicted] >= config.min_cluster_size, predicted, -1)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels[labels >= 0])]
    groups.sort(key=lambda rows: int(rows[0]))
    metrics.update(gmm_selected_components=best.n_components, gmm_bic=float(best_bic),
        gmm_fit_points=len(fit), gmm_converged=int(best.converged_), gmm_iterations=best.n_iter_)
    return groups, labels, metrics

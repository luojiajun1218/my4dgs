"""Conditional cross-calibration of a frozen population's colour endpoints.

The default endpoints are prior, DC-only, SH-only and full. Callers may label
the same three simplex directions as common-only, detail-only and full instead.
Geometry, opacity,
population, old background and surviving codec predictors must stay identical.
The three weights blend *attributes* with a remaining prior weight. Rendered
endpoint images define only a linear surrogate: native SH/output clamps and
the actual codec are checked afterwards by the caller's validation callback.
This is conditional prediction on the opposite training-camera group, not an
untouched image holdout (source selection/initialisation may have used all20).
"""

from itertools import combinations
from pathlib import Path

import torch


ENDPOINTS = ('prior', 'dc', 'sh', 'full')
MAX_BACKTRACKS = 12


def _finite_double(value, shape, name):
    value = torch.as_tensor(value).detach().to(device='cpu', dtype=torch.float64)
    if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return value


def solve_predictive_simplex(gram, rhs):
    """Solve min w'G w-2 b'w, w>=0, sum(w)<=1 by active faces.

    All 3-variable simplex faces are enumerated analytically. Singular faces
    use a Moore-Penrose solve, not a ridge parameter or a weight grid. Feasible
    face stationary points and vertices include a global minimizer of this
    convex quadratic. The zero/prior solution wins numerical ties.
    """
    gram = _finite_double(gram, (3, 3), 'Gram')
    rhs = _finite_double(rhs, (3,), 'RHS')
    scale = max(float(gram.abs().max()), float(rhs.abs().max()), torch.finfo(torch.float64).tiny)
    numeric = 256 * torch.finfo(torch.float64).eps
    if float((gram-gram.T).abs().max()) > numeric*scale:
        raise ValueError('Gram must be symmetric')
    gram = (gram+gram.T)*.5
    eigenvalues = torch.linalg.eigvalsh(gram)
    if float(eigenvalues.min()) < -numeric*scale:
        raise ValueError('Gram must be positive semidefinite')
    # Scaling makes KKT's dimensionless equality row comparable to image terms.
    g, b = gram/scale, rhs/scale
    candidates = [(torch.zeros(3, dtype=torch.float64), 'prior')]
    for size in range(1, 4):
        for support in combinations(range(3), size):
            ids = torch.tensor(support, dtype=torch.long)
            block, target = g[ids][:, ids], b[ids]
            solutions = [(torch.linalg.pinv(block, hermitian=True)@target, False)]
            kkt = torch.zeros(size+1, size+1, dtype=torch.float64)
            kkt[:size, :size] = block
            kkt[:size, size] = kkt[size, :size] = 1
            answer = torch.linalg.pinv(kkt, hermitian=True)@torch.cat((target, target.new_ones(1)))
            solutions.append((answer[:size], True))
            for local, capped in solutions:
                if not bool(torch.isfinite(local).all()) or float(local.min()) < -numeric:
                    continue
                if float(local.sum()) > 1+numeric or (capped and abs(float(local.sum())-1) > numeric):
                    continue
                # A singular/inconsistent pseudoinverse is not stationary.
                gradient = block@local-target
                residual = gradient-gradient.mean() if capped else gradient
                if float(residual.abs().max()) > numeric*max(1., float(target.abs().max())):
                    continue
                local = local.clamp_min(0)
                if float(local.sum()) > 1:
                    local = local/local.sum()  # roundoff only, not a projection heuristic
                weights = torch.zeros(3, dtype=torch.float64).index_copy(0, ids, local)
                candidates.append((weights, f'{support}:sum={1 if capped else "free"}'))
    best, face = candidates[0]
    value = 0.
    tolerance = numeric*scale
    for weights, name in candidates[1:]:
        objective = float(weights@gram@weights-2*rhs@weights)
        if objective < value-tolerance:
            best, face, value = weights, name, objective
    return best, dict(active_face=face, candidate_faces=len(candidates),
                      objective_change=value, gram_eigenvalues=eigenvalues.tolist(),
                      gram_rank=int(torch.linalg.matrix_rank(gram, hermitian=True)))


class PredictiveColourMoments:
    """One whole-image Gram per view, retaining all overlapping responses."""

    def __init__(self):
        self.gram = torch.zeros(3, 3, dtype=torch.float64)
        self.rhs = torch.zeros(3, dtype=torch.float64)
        self.prior_mse_sum = 0.
        self.views = 0
        self.elements = 0

    @torch.no_grad()
    def add(self, prior, dc, sh, full, target):
        images = [torch.as_tensor(x).detach() for x in (prior, dc, sh, full, target)]
        if not images[0].numel() or any(x.shape != images[0].shape for x in images):
            raise ValueError('Endpoint images and target must have the same nonempty shape')
        if any(not bool(torch.isfinite(x).all()) for x in images):
            raise ValueError('Endpoint images and target must be finite')
        # Images are output-clamped for the surrogate loss, as in actual image
        # quality evaluation. Attribute interpolation is checked separately.
        prior, dc, sh, full, target = [x.to(device=images[0].device).clamp(0, 1) for x in images]
        basis = torch.stack((dc-prior, sh-prior, full-prior)).flatten(1).double()
        residual = (target-prior).flatten().double()
        count = residual.numel()
        self.gram += (basis@basis.T/count).cpu()
        self.rhs += (basis@residual/count).cpu()
        self.prior_mse_sum += float(residual.square().mean())
        self.views += 1
        self.elements += count

    def solve(self):
        if not self.views:
            raise ValueError('Predictive calibration needs at least one view')
        g, b = self.gram/self.views, self.rhs/self.views
        weights, report = solve_predictive_simplex(g, b)
        prior = self.prior_mse_sum/self.views
        report.update(gram=g.tolist(), rhs=b.tolist(), view_count=self.views,
                      image_elements=self.elements, prior_surrogate_mse=prior,
                      fitted_surrogate_mse=prior+float(weights@g@weights-2*b@weights),
                      view_weighting='equal mean-image-MSE per view; full RGB cross terms')
        return weights, report


def _validation(value):
    value = dict(value) if isinstance(value, dict) else dict(mse=float(value))
    if 'mse' not in value:
        raise ValueError('Native/codec validation must return mse or a scalar MSE')
    mse = float(value['mse'])
    # A failed codec trial may return inf with eligible=False. It can never win.
    eligible = bool(value.get('eligible', True)) and bool(torch.isfinite(torch.tensor(mse)))
    if mse < 0:
        raise ValueError('Native/codec MSE cannot be negative')
    return dict(value, mse=mse, eligible=eligible)


@torch.no_grad()
def calibrate_predictive_colour(views, render_endpoint, apply_weights,
                                validate_weights, target=None, *,
                                endpoint_kinds=ENDPOINTS):
    """Fit one three-direction simplex and apply validated weights once.

    ``render_endpoint(view, kind)`` returns a whole-scene RGB Tensor or a dict
    containing ``render``. The caller freezes model endpoints before calling.
    ``endpoint_kinds`` is a tuple of four unique names, starting with ``prior``;
    its remaining three names specify the weight order passed to the model.
    Defaults preserve DC/SH/full behaviour. V11 fine populations use
    ``('prior', 'common', 'detail', 'full')``.
    ``validate_weights(w)`` must temporarily blend frozen raw endpoints, run
    actual native/codec validation in the *same fixed population and predictor
    context*, and restore its temporary state. It returns mean image MSE, or
    ``{'mse': ..., 'eligible': ..., ...}``. No training follows calibration.
    ``eligible`` means only codec/physical validity, never improvement against
    the incumbent: a poor but legal prior can still have a useful nonzero fit.
    ``apply_weights(w)`` is invoked exactly once with the selected weights;
    B's endpoint snapshot prevents ratcheting on repeated temporary blends.

    Structure/deletion choice and the two-direction prediction/incumbent
    acceptance are deliberately the controller's responsibility. No comparison
    to a different population's prior is valid inside this helper.
    """
    if validate_weights is None:
        raise ValueError('Actual native/codec validation callback is required')
    kinds = tuple(endpoint_kinds)
    if (len(kinds) != 4 or any(not isinstance(kind, str) or not kind for kind in kinds)
            or kinds[0] != 'prior' or len(set(kinds)) != 4):
        raise ValueError('endpoint_kinds must have four unique names starting with prior')
    named = list(views.items()) if isinstance(views, dict) else [(str(i), v) for i, v in enumerate(views)]
    if not named:
        raise ValueError('Predictive calibration needs opposite training views')
    for name, view in named:
        camera = Path(str(getattr(view, 'path', name))).name
        if Path(str(name)).name in ('cam00', 'cam00.png') or camera in ('cam00', 'cam00.png'):
            raise ValueError('External cam00 cannot enter predictive colour calibration')
    moments = PredictiveColourMoments()
    for _, view in named:
        rendered = []
        for kind in kinds:
            image = render_endpoint(view, kind)
            rendered.append(image['render'] if isinstance(image, dict) else image)
        truth = target(view) if target is not None else view.image
        moments.add(*rendered, truth)
    fitted, report = moments.solve()
    zero = torch.zeros_like(fitted)
    baseline = _validation(validate_weights(zero.clone()))
    attempts = []
    chosen, chosen_validation = zero, baseline
    if baseline['eligible'] and bool(fitted.any()):
        for exponent in range(MAX_BACKTRACKS+1):
            scale = 2.**(-exponent)
            weights = fitted*scale
            result = _validation(validate_weights(weights.clone()))
            attempts.append(dict(scale=scale, weights=weights.tolist(), validation=result))
            # No quality threshold: strict actual-MSE decrease, otherwise prior.
            if result['eligible'] and result['mse'] < baseline['mse']:
                chosen, chosen_validation = weights, result
                break
    apply_weights(chosen.clone())
    report.update(weights=chosen.tolist(), prior_weight=1-float(chosen.sum()),
                  analytic_weights=fitted.tolist(), validated=bool(baseline['eligible']),
                  nonprior_selected=bool(chosen.any()), prior_validation=baseline,
                  selected_validation=chosen_validation, native_backtracking=attempts,
                  endpoint_order=list(kinds), weight_labels=list(kinds[1:]),
                  weights_by_endpoint=dict(zip(kinds, [1-float(chosen.sum()), *chosen.tolist()])),
                  scope='conditional opposite-training-view prediction; frozen population/geometry/alpha/predictor')
    return report

"""Observation-driven birth proposals and reversible birth/death transactions.

Depth is estimated from synchronized RGB cameras, not copied from an old GS.
Candidates remain provisional until all standard training cameras accept their
aggregate *quantized* distortion and byte cost. External test cameras never
enter this module. The selected complete population is then refitted unchanged.
"""

from dataclasses import dataclass, asdict, field
import math

import torch
from torch.nn import functional as F

from .birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES


@dataclass
class ResidualBirthConfig:
    proposal_views: int = 8
    validation_views: int = 4  # Legacy config compatibility; v5 uses every training view.
    max_candidates: int = 512
    repair_candidates: int = 128
    depth_planes: int = 64
    max_groups: int = 16
    birth_budget_bytes: int = 16000
    target_frame_bytes: int = 50000
    max_gaussians: int = 150000
    residual_floor: float = .03
    match_cost_limit: float = .25
    voxel_extent: float = .002
    validation_tolerance: float = 1.e-4
    max_cohorts: int = 2

    def validate(self):
        for name in ('proposal_views', 'validation_views', 'max_candidates',
                     'depth_planes', 'max_groups', 'birth_budget_bytes',
                     'target_frame_bytes', 'max_gaussians', 'max_cohorts'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'birth.residual.{name} must be a positive integer')
        if not 0 <= self.repair_candidates < self.max_candidates:
            raise ValueError('Repair quota must leave room for new spatial support')
        if self.depth_planes < 8:
            raise ValueError('RGB depth search needs at least eight planes')
        for name in ('residual_floor', 'match_cost_limit', 'voxel_extent'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'birth.residual.{name} must be positive')
        if not 0 <= self.validation_tolerance < .01:
            raise ValueError('Invalid independent-camera tolerance')


def project(view, xyz):
    """COLMAP/3DGS row-vector convention; grid_sample align_corners=False."""
    matrix = view.world_view_transform.to(xyz)
    camera = xyz @ matrix[:3, :3] + matrix[3, :3]
    z = camera[..., 2]
    grid = torch.stack((camera[..., 0] / z.clamp_min(1.e-8) /
                        math.tan(float(view.fovx) * .5),
                        camera[..., 1] / z.clamp_min(1.e-8) /
                        math.tan(float(view.fovy) * .5)), -1)
    return grid, z


def unproject(view, pixels, depth):
    height, width = view.image.shape[-2:]
    grid = (2 * pixels + 1) / pixels.new_tensor([width, height]) - 1
    camera = torch.stack((grid[..., 0] * math.tan(float(view.fovx) * .5) * depth,
                          grid[..., 1] * math.tan(float(view.fovy) * .5) * depth,
                          depth), -1)
    inverse = torch.linalg.inv(view.world_view_transform.to(camera))
    return camera @ inverse[:3, :3] + inverse[3, :3]


def sample_image(image, grid):
    shape = grid.shape[:-1]
    samples = F.grid_sample(image[None], grid.reshape(1, -1, 1, 2),
                            mode='bilinear', padding_mode='zeros',
                            align_corners=False)[0, :, :, 0].T
    return samples.reshape(*shape, image.shape[0])


def matrix_quaternion(rotation):
    """Stable scalar-first quaternion, including 180-degree camera rotations."""
    r = rotation
    roots = torch.stack((1+r[0, 0]+r[1, 1]+r[2, 2],
                         1+r[0, 0]-r[1, 1]-r[2, 2],
                         1-r[0, 0]+r[1, 1]-r[2, 2],
                         1-r[0, 0]-r[1, 1]+r[2, 2])).clamp_min(0).sqrt()
    candidates = torch.stack((
        torch.stack((roots[0]**2, r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1])),
        torch.stack((r[2, 1]-r[1, 2], roots[1]**2, r[1, 0]+r[0, 1], r[0, 2]+r[2, 0])),
        torch.stack((r[0, 2]-r[2, 0], r[1, 0]+r[0, 1], roots[2]**2, r[2, 1]+r[1, 2])),
        torch.stack((r[1, 0]-r[0, 1], r[2, 0]+r[0, 2], r[2, 1]+r[1, 2], roots[3]**2))))
    index = roots.argmax()
    return F.normalize(candidates[index] / (2 * roots[index].clamp_min(1.e-8)), dim=0)


@torch.no_grad()
def match_depth(reference, peers, pixels, near, far, planes=64):
    """Coarse-to-fine patch plane sweep, with support from two peer cameras.

    The search spans the scene, so a surface in front of/behind an incumbent
    can be proposed. Rendered depth is deliberately not the selected depth.
    """
    count = len(pixels)
    offsets = pixels.new_tensor([[x, y] for y in (-3., 0., 3.) for x in (-3., 0., 3.)])
    patch_pixels = pixels[:, None] + offsets[None]
    h, w = reference.image.shape[-2:]
    ref_grid = (2*patch_pixels+1) / pixels.new_tensor([w, h]) - 1
    ref_patch = sample_image(reference.image, ref_grid)
    # Spatial texture, not variance between RGB channels: a flat red wall
    # contains no triangulation evidence even though R differs from G/B.
    centered = ref_patch - ref_patch.mean(1, keepdim=True)
    texture = centered.square().mean((1, 2)).sqrt()
    ref_unit = centered / centered.square().sum((1, 2), keepdim=True).sqrt().clamp_min(1.e-6)

    def costs(depths):
        size = depths.shape[1]
        xyz = unproject(reference, patch_pixels[:, None].expand(-1, size, -1, -1),
                        depths[:, :, None].expand(-1, -1, len(offsets)))
        values = []
        for peer in peers:
            grid, z = project(peer, xyz)
            valid = ((grid.abs() < 1).all(-1) & (z > 0)).float().mean(-1) > .8
            patch = sample_image(peer.image, grid)
            patch = patch - patch.mean(2, keepdim=True)
            patch = patch / patch.square().sum((2, 3), keepdim=True).sqrt().clamp_min(1.e-6)
            cost = (1 - (patch * ref_unit[:, None]).sum((2, 3))) * .5
            values.append(cost.masked_fill(~valid, 2.))
        all_cost = torch.stack(values)
        # Two agreeing views, rather than a single accidental match.
        best_two = all_cost.topk(2, dim=0, largest=False).values
        return best_two.mean(0), (all_cost < .3).sum(0)

    depths = torch.linspace(float(near), float(far), planes, device=pixels.device,
                            dtype=pixels.dtype)[None].expand(count, -1)
    spacing = (float(far)-float(near)) / (planes-1)
    unique_depth = torch.ones(count, dtype=torch.bool, device=pixels.device)
    for refinement in range(3):
        cost, support = costs(depths)
        if refinement == 0:
            plausible = cost <= cost.min(-1, keepdim=True).values+.03
            modes = plausible[:, 0].long() + (plausible[:, 1:] & ~plausible[:, :-1]).sum(-1)
            unique_depth = modes <= 1
        index = cost.argmin(-1)
        best = depths.gather(1, index[:, None])[:, 0]
        if refinement < 2:
            offsets_d = torch.linspace(-spacing, spacing, 16, device=pixels.device)
            depths = (best[:, None] + offsets_d).clamp(float(near), float(far))
            spacing = 2*spacing / 15
    best_cost = cost.gather(1, index[:, None])[:, 0]
    supporters = support.gather(1, index[:, None])[:, 0]
    weights = torch.softmax(-(cost-cost.min(-1, keepdim=True).values)/.03, -1)
    uncertainty = ((depths-best[:, None]).square()*weights).sum(-1).sqrt()
    xyz = unproject(reference, pixels, best)
    colors = [sample_image(reference.image, ref_grid[:, 4])]
    for peer in peers:
        grid, _ = project(peer, xyz)
        colors.append(sample_image(peer.image, grid))
    color = torch.stack(colors).median(0).values.clamp(0, 1)
    valid = (supporters >= 2) & (texture > .01) & torch.isfinite(best_cost) & unique_depth
    return xyz, color, best, uncertainty, best_cost, valid


@torch.no_grad()
def nearest_sources(xyz, reference):
    if not len(xyz):
        return torch.empty(0, dtype=torch.long, device=reference.device)
    return torch.cat([torch.cdist(part, reference).argmin(-1)
                      for part in xyz.split(64)])


def spatial_groups(xyz, limit):
    """Balanced spatial groups; validation evaluates whole interacting groups."""
    if not len(xyz):
        return []
    groups = [torch.arange(len(xyz), device=xyz.device)]
    while len(groups) < limit:
        sizes = [len(group) for group in groups]
        index = max(range(len(groups)), key=sizes.__getitem__)
        ids = groups[index]
        if len(ids) < 2:
            break
        axis = xyz[ids].amax(0).sub(xyz[ids].amin(0)).argmax()
        ordered = ids[xyz[ids, axis].argsort(stable=True)]
        middle = len(ordered)//2
        groups[index:index+1] = [ordered[:middle], ordered[middle:]]
    return groups


def improves(candidate, baseline, tolerance):
    """Optimize the same aggregate distortion used for all training views."""
    return bool(candidate.mean() < baseline.mean()-1.e-10)


class SubsetView:
    def __init__(self, model, rows):
        self.model, self.rows = model, rows

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def get_xyz(self): return self.model.get_xyz[self.rows]
    @property
    def get_rotation(self): return self.model.get_rotation[self.rows]
    @property
    def get_scaling(self): return self.model.get_scaling[self.rows]
    @property
    def get_opacity(self): return self.model.get_opacity[self.rows]
    @property
    def get_features(self): return self.model.get_features[self.rows]

    def get_covariance(self, scaling_modifier=1.):
        return self.model.get_covariance(scaling_modifier)[self.rows]


@dataclass
class BirthTransaction:
    raw: dict
    sources: torch.Tensor
    replacements: torch.Tensor
    scores: torch.Tensor
    seed_xyz: torch.Tensor
    seed_rotation: torch.Tensor
    seed_scale: torch.Tensor
    observations: torch.Tensor
    fit_names: tuple
    validation_names: tuple
    proposed_spatial: int = 0
    proposed_repairs: int = 0
    group_ids: torch.Tensor = None
    region_diagnostics: dict = field(default_factory=dict)
    support_domains: list = field(default_factory=list)
    provisional_deleted: torch.Tensor = None
    fixed_keep: torch.Tensor = None
    fixed_reference: dict = None
    family_state: dict = None
    patch_state: dict = None
    patch_specs: list = field(default_factory=list)
    old_base_symbols: torch.Tensor = None
    old_base_opacity: torch.Tensor = None
    fixed_old_symbols: torch.Tensor = None
    old_render_reference: dict = None
    reincarnation_state: dict = None
    reincarnation_specs: list = field(default_factory=list)
    crossfit_state: dict = None
    predictive_model: bool = False
    persistent_state: dict = None

    @staticmethod
    def tensor_tree(value, device):
        if torch.is_tensor(value):
            return value.detach().to(device).clone()
        if isinstance(value, dict):
            return {key: BirthTransaction.tensor_tree(item, device) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(BirthTransaction.tensor_tree(item, device) for item in value)
        return value

    def capture(self):
        return self.tensor_tree(self.__dict__, 'cpu')

    @classmethod
    def restore(cls, state, device):
        return cls(**cls.tensor_tree(state, device))


@torch.no_grad()
def residual_regions(residual, image, limit, floor):
    """Partition connected residual support by integrated error, not peaks.

    Cells retain their active-pixel area and error mass. Appearance boundaries
    prevent a region from borrowing support across an obvious image edge.
    Weighted spatial bisection spends the finite proposal budget on large error
    regions; isolated high peaks cannot consume an entire region's allocation.
    """
    height, width = residual.shape
    active = residual > floor
    active[:5] = False; active[-5:] = False
    active[:, :5] = False; active[:, -5:] = False
    mass = residual.square()*active
    area = F.avg_pool2d(active.float()[None, None], 8, 8)[0, 0]*64
    pooled = F.avg_pool2d(mass[None, None], 8, 8)[0, 0]*64
    # A cell's representative is its error-weighted pixel centroid, so patches
    # follow the residual support rather than arbitrary grid-center rays.
    yy, xx = torch.meshgrid(torch.arange(height, device=residual.device),
                            torch.arange(width, device=residual.device), indexing='ij')
    cx = F.avg_pool2d((mass*xx)[None, None], 8, 8)[0, 0]*64/pooled.clamp_min(1.e-12)
    cy = F.avg_pool2d((mass*yy)[None, None], 8, 8)[0, 0]*64/pooled.clamp_min(1.e-12)
    color = F.avg_pool2d(image[None], 8, 8)[0].permute(1, 2, 0)
    locations = (pooled > 0).nonzero(as_tuple=False)
    if not len(locations):
        return []
    cells = locations.cpu().tolist()
    colors = color[locations[:, 0], locations[:, 1]].cpu().tolist()
    lookup = {tuple(cell): index for index, cell in enumerate(cells)}
    remaining = set(range(len(cells)))
    components = []
    while remaining:
        first = min(remaining); remaining.remove(first)
        component, pending = [first], [first]
        while pending:
            index = pending.pop(); y, x = cells[index]
            for key in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                peer = lookup.get(key)
                if peer not in remaining:
                    continue
                if sum((a-b)**2 for a, b in zip(colors[index], colors[peer])) > .15**2:
                    continue
                remaining.remove(peer); component.append(peer); pending.append(peer)
        components.append(torch.tensor(component, device=residual.device))
    rows, cols = locations.unbind(-1)
    points = torch.stack((cx[rows, cols], cy[rows, cols]), -1)
    masses, areas = pooled[rows, cols], area[rows, cols]
    # At least four cells/anchors are needed to test a surface's planarity.
    regions = [ids for ids in components if len(ids) >= 4]
    regions.sort(key=lambda ids: float(masses[ids].sum()), reverse=True)
    regions = regions[:limit]
    while len(regions) < limit:
        splittable = [i for i, ids in enumerate(regions) if len(ids) >= 8]
        if not splittable:
            break
        index = max(splittable, key=lambda i: float(masses[regions[i]].sum()))
        ids = regions.pop(index)
        centered = points[ids]-points[ids].mean(0)
        axis = (centered.square()*masses[ids, None]).sum(0).argmax()
        ordered = ids[points[ids, axis].argsort(stable=True)]
        cumulative = masses[ordered].cumsum(0)
        cut = int(torch.searchsorted(cumulative, cumulative[-1]*.5))+1
        cut = max(4, min(len(ordered)-4, cut))
        regions.extend((ordered[:cut], ordered[cut:]))
    result = []
    for ids in regions:
        # Farthest-point anchor selection covers the region's extent. Anchor
        # weights include assigned cells, preserving integrated residual mass.
        chosen = [int(masses[ids].argmax())]
        distance = (points[ids]-points[ids[chosen[0]]]).square().sum(-1)
        while len(chosen) < min(9, len(ids)):
            distance[chosen] = -1
            index = int(distance.argmax()); chosen.append(index)
            distance = torch.minimum(distance, (points[ids]-points[ids[index]]).square().sum(-1))
        anchors = ids[torch.tensor(chosen, device=ids.device)]
        assignments = torch.cdist(points[ids], points[anchors]).argmin(-1)
        anchor_mass = masses.new_zeros(len(anchors)).scatter_add_(0, assignments, masses[ids])
        anchor_area = areas.new_zeros(len(anchors)).scatter_add_(0, assignments, areas[ids])
        result.append(dict(pixels=points[anchors], mass=anchor_mass, area=anchor_area,
                           total_mass=float(masses[ids].sum()), total_area=float(areas[ids].sum())))
    return result


@torch.no_grad()
def region_surface_patches(reference, region, xyz, color, depth, uncertainty, cost, valid):
    """Explain a residual region with coherent multi-view surface patches.

    Covariance is measured from independently triangulated anchor positions.
    No area is extrapolated from unmatched image pixels or a preset radius.
    Disconnected depths/colors form separate components; non-planar components
    are rejected rather than initialized as a thick foreground blob.
    """
    ids = valid.nonzero(as_tuple=True)[0]
    if len(ids) < 4:
        return []
    pixels = region['pixels'][ids]
    points, rgb, z, noise = xyz[ids], color[ids], depth[ids], uncertainty[ids]
    focal = reference.image.shape[-1]/(2*math.tan(float(reference.fovx)*.5))
    pixel_world = z/focal
    pixel_distance = torch.cdist(pixels, pixels)
    world_distance = torch.cdist(points, points)
    color_distance = torch.cdist(rgb, rgb)
    adjacent = torch.zeros_like(pixel_distance, dtype=torch.bool)
    nearest = pixel_distance.topk(min(4, len(ids)), largest=False).indices
    adjacent.scatter_(1, nearest, True); adjacent |= adjacent.T.clone()
    lateral = pixel_distance*(pixel_world[:, None]+pixel_world[None])*.5
    adjacent &= (world_distance <= 3*lateral+3*(noise[:, None]+noise[None]))
    adjacent &= color_distance < .2
    remaining = set(range(len(ids))); components = []
    while remaining:
        first = min(remaining); remaining.remove(first)
        part, pending = [first], [first]
        while pending:
            index = pending.pop()
            for peer in adjacent[index].nonzero(as_tuple=True)[0].tolist():
                if peer in remaining:
                    remaining.remove(peer); part.append(peer); pending.append(peer)
        if len(part) >= 4:
            components.append(torch.tensor(part, device=xyz.device))
    patches = []
    for part in components:
        anchors = ids[part]
        values = xyz[anchors]; weights = region['mass'][anchors]
        normalized = weights/weights.sum().clamp_min(1.e-12)
        center = (values*normalized[:, None]).sum(0)
        delta = values-center
        covariance = (delta*normalized[:, None]).T@delta
        eigenvalues, basis = torch.linalg.eigh(covariance)
        resolution = (depth[anchors]/focal).median()
        tolerance = 3*uncertainty[anchors].median()+resolution
        plane_error = (delta@basis[:, 0]).abs()
        # Require support in two tangent directions and a thin explainable
        # plane. Merely collecting several matches is not surface evidence.
        if eigenvalues[1] < resolution.square()/12 or bool((plane_error > tolerance).any()):
            continue
        if color[anchors].sub(color[anchors].median(0).values).norm(dim=-1).max() > .2:
            continue
        variance = eigenvalues.clamp_min(resolution.square()/12)
        variance[0] = torch.maximum(variance[0], uncertainty[anchors].median().square())
        # Eigenvectors are columns; choose a proper (right-handed) rotation.
        if torch.det(basis) < 0:
            basis[:, 0] *= -1
        world_covariance = basis@torch.diag(variance)@basis.T
        transform = reference.world_view_transform.to(center)
        camera_center = center@transform[:3, :3]+transform[3, :3]
        cz = camera_center[2].clamp_min(1.e-8)
        fy = reference.image.shape[-2]/(2*math.tan(float(reference.fovy)*.5))
        jacobian = center.new_tensor([[focal, 0., 0.], [0., fy, 0.]])/cz
        jacobian[:, 2] = -center.new_tensor([focal, fy])*camera_center[:2]/cz.square()
        camera_covariance = transform[:3, :3].T@world_covariance@transform[:3, :3]
        screen_covariance = jacobian@camera_covariance@jacobian.T
        support_area = region['area'][anchors].sum()
        score = region['mass'][anchors].sum()*torch.exp(-cost[anchors].mean()/.1)
        patches.append(dict(xyz=center, color=(color[anchors]*normalized[:, None]).sum(0),
            scale=variance.sqrt(), rotation=matrix_quaternion(basis), score=score,
            support=dict(kind='mvs', xyz=values.clone(), color=color[anchors].clone(),
                         mass=region['mass'][anchors].clone()),
            diagnostic=dict(error_mass=float(region['mass'][anchors].sum()),
                support_pixels=float(support_area), matched_anchors=len(anchors),
                plane_error_max=float(plane_error.max()),
                projected_integral_area=float(2*math.pi*screen_covariance.det().clamp_min(0).sqrt()))))
    return patches


def typed_novel_residual(residual, current_gt, previous_gt, uncovered):
    """Separate temporal/uncovered support from already covered static error."""
    change = (current_gt-previous_gt).square().mean(0).sqrt()
    novel = torch.minimum(residual, torch.maximum(change, residual*uncovered))
    return novel, dict(current_error_mass=float(residual.square().sum()),
        temporal_change_mass=float(change.square().sum()),
        typed_novel_mass=float(novel.square().sum()),
        suppressed_persistent_mass=float((residual.square()-novel.square()).clamp_min(0).sum()))


def previous_training_views(module, names):
    """Read t-1 standard train images without changing the active frame."""
    from pathlib import Path
    data = module.datamodule
    current = data.current_frame
    if current < 1:
        raise ValueError('Typed births require a preceding observed frame')
    try:
        data.current_frame = current-1
        dataset = data.get_train_dataset()
        lookup = {Path(item.path).name: index for index, item in enumerate(dataset.items)}
        if not set(names).issubset(lookup):
            raise ValueError('Previous training frame lost a proposal camera')
        return {name: dataset[lookup[name]].to(module.device) for name in names}
    finally:
        data.current_frame = current


@torch.no_grad()
def cache_surface_patch_evidence(module, views):
    """Native compositing moments from standard training cameras, read-only."""
    from pathlib import Path
    from .birth_quantization import model_raw
    from .rigid_residual_tree import quaternion_rotate
    views = list(views.values()) if isinstance(views, dict) else list(views)
    names = [Path(view.path).name for view in views]
    if 'cam00.png' in names:
        raise ValueError('Surface evidence cannot use external test cam00')
    previous = previous_training_views(module, names)
    base = model_raw(module.gaussians)
    q = F.normalize(base['rotation'], dim=-1)
    normal_axis = base['logscale'].argmin(-1)
    local = F.one_hot(normal_axis, 3).to(base['xyz'])
    normals = quaternion_rotate(q, local)
    renderer = module.renderer
    old_model, old_collect = renderer.gaussians, getattr(renderer, 'collect_contributions', False)
    renderer.gaussians = module.gaussians
    result = {}
    try:
        for name, view in zip(names, views):
            renderer.collect_contributions = True
            package = renderer(view, module.background)
            observed = package.get('motion_contribution_filter')
            if observed is None:
                raise ValueError('Surface proposals require actual native old contribution evidence')
            renderer.collect_contributions = False
            black = torch.zeros_like(module.background)
            white = base['xyz'].new_ones((len(base['xyz']), 3))
            coverage = renderer(view, black, override_color=white)['render'][0]
            matrix = view.world_view_transform.to(base['xyz'])
            z = (base['xyz']@matrix[:3, :3]+matrix[3, :3])[:, 2]
            depth_moments = renderer(view, black, override_color=torch.stack((z, z.square(), torch.ones_like(z)), -1))['render']
            mass = coverage.clamp_min(1.e-8)
            depth = depth_moments[0]/mass
            variance = (depth_moments[1]/mass-depth.square()).clamp_min(0)
            facing = ((view.camera_center.to(normals)-base['xyz'])*normals).sum(-1)
            n = normals*torch.where(facing >= 0, 1., -1.)[:, None]
            normal = renderer(view, black, override_color=n)['render']/mass[None]
            payload = dict(rendered=package['render'], residual_rgb=package['render']-view.image,
                current_gt=view.image, previous_image=previous[name].image,
                coverage=coverage, depth=depth, depth_variance=variance,
                normal=normal, observed_old=observed)
            result[name] = dict(view=view, **{key:value.detach().cpu() for key,value in payload.items()})
    finally:
        renderer.gaussians = old_model
        renderer.collect_contributions = old_collect
    return result


@torch.no_grad()
def rank_repair_contrast(module, views, candidates, base, cached):
    """Allocate detail capacity to coherent spatial residual, not DC bias."""
    from pathlib import Path
    if not len(candidates):
        return candidates, dict(prefilter_count=0, positive_contrast_count=0)
    domains = [covariance_support(base, int(row)) for row in candidates]
    points = torch.stack([domain['xyz'] for domain in domains])
    counts = points.new_zeros(points.shape[:-1])
    sums = points.new_zeros((*points.shape[:-1], 3)); squares = torch.zeros_like(sums)
    area_sum = points.new_zeros(len(candidates)); area_count = torch.zeros_like(area_sum)
    residual_scale = torch.zeros_like(area_sum)
    kernel = points.new_tensor([math.exp(-.5*(x*x+y*y)) for y in (-1., 0., 1.) for x in (-1., 0., 1.)])
    for view in views:
        name = Path(view.path).name
        if name in cached:
            residual, observed = cached[name]
        else:
            with torch.enable_grad():
                probe = module.gaussians._features_dc.detach().requires_grad_(True)
                module.gaussians.rigid_sh_dc = probe
                try:
                    image = module.renderer(view, module.background)['render']
                    loss = (image-view.image).square().mean()
                    gradient = torch.autograd.grad(loss, probe)[0]
                    observed = gradient[:, 0].norm(dim=-1) > 0
                    residual = (image.detach()-view.image)
                finally:
                    module.gaussians.rigid_sh_dc = None
        grid, depth = project(view, points)
        valid = (grid.abs() < 1).all(-1) & (depth > 0) & observed[candidates, None]
        sampled = sample_image(residual, grid)
        residual_scale = torch.maximum(residual_scale, sampled.abs().amax((1, 2)))
        weight = valid*kernel[None]
        mean = (sampled*weight[..., None]).sum(1)/weight.sum(1).clamp_min(1.e-12)[:, None]
        centered = sampled-mean[:, None]
        counts += valid
        sums += centered*valid[..., None]; squares += centered.square()*valid[..., None]
        h, w = view.image.shape[-2:]
        pixels = grid*grid.new_tensor([w*.5, h*.5])
        a, b = (pixels[:, 5]-pixels[:, 3])*.5, (pixels[:, 7]-pixels[:, 1])*.5
        area = (a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]).abs()*2*math.pi
        seen = valid.sum(-1) >= 2
        area_sum += area*seen; area_count += seen
    mean = sums/counts.clamp_min(1)[..., None]
    variance = (squares/counts.clamp_min(1)[..., None]-mean.square()).clamp_min(0)
    coherent = (mean.square()-variance/counts.clamp_min(1)[..., None]).clamp_min(0)
    coherent *= (counts >= 2)[..., None]
    energy = (coherent.mean(-1)*kernel[None]).sum(-1)/kernel.sum()
    # A constant residual should be exactly zero detail, including bilinear
    # sampling/weighted-mean roundoff. This is an arithmetic precision bound,
    # not an image-error or scene-specific allocation threshold.
    roundoff = torch.finfo(points.dtype).eps*(len(views)+len(kernel)+4)*residual_scale
    energy = (energy-roundoff.square()).clamp_min(0)
    scores = energy*area_sum/area_count.clamp_min(1)
    positive = scores > 0
    ordered = scores.argsort(descending=True, stable=True)
    ordered = ordered[positive[ordered]]
    return candidates[ordered], dict(prefilter_count=len(candidates), positive_contrast_count=int(positive.sum()),
        positive_contrast_fraction=float(positive.float().mean()),
        prefilter_parent_rows=candidates.tolist(), ranked_parent_rows=candidates[ordered].tolist(),
        ranked_contrast_scores=scores[ordered].tolist(),
        semantics='per-source tangent 9 probes; per-view mean removed; coherent mean-pattern energy minus variance/count; weighted by visible projected tangent area; camera visibility from nonzero actual image-DC gradient')


@torch.no_grad()
def _propose_existing(module, config, views, fit_names, validation_names):
    """Generate independent spatial births plus reversible source repairs."""
    from pathlib import Path
    from .birth_quantization import model_raw
    device = module.device
    fit = [view for view in views if Path(view.path).name in fit_names]
    chosen = [fit[index] for index in torch.linspace(0, len(fit)-1,
              min(config.proposal_views, len(fit))).round().long().tolist()]
    conservative = getattr(getattr(getattr(module, 'config', None), 'birth', None),
                           'radiance_mode', None) in ('conservative_detail', 'surface_patch', 'support_reincarnation', 'dual_predictive_reincarnation', 'source_residual')
    previous = previous_training_views(module, [Path(view.path).name for view in chosen]) if conservative else {}
    proposal_cache, typed_reports = {}, []
    extent = module.datamodule.cameras_extent
    xyz_parts, color_parts, scale_parts, rotation_parts, score_parts, replacement_parts = [], [], [], [], [], []
    group_parts, region_reports, support_domains = [], [], []
    next_group = 0
    proposed_region_mass = proposed_region_area = 0.
    repair_gradient = module.gaussians._features_dc.new_zeros(len(module.gaussians._xyz), 3)
    quota = max(1, math.ceil((config.max_candidates-config.repair_candidates)/len(chosen)))
    base = model_raw(module.gaussians)
    for reference in chosen:
        # Proposal probes use training views only. Selection later aggregates
        # the full standard training set; external test views remain excluded.
        with torch.enable_grad():
            probe = module.gaussians._features_dc.detach().requires_grad_(True)
            module.gaussians.rigid_sh_dc = probe
            try:
                rendered = module.renderer(reference, module.background)['render']
                error = (rendered-reference.image).square().mean()
                gradient = torch.autograd.grad(error, probe)[0]
                repair_gradient.add_(gradient[:, 0].detach())
                residual = (rendered.detach()-reference.image).square().mean(0).sqrt()
                proposal_cache[Path(reference.path).name] = (
                    rendered.detach()-reference.image, gradient[:, 0].detach().norm(dim=-1) > 0)
            finally:
                module.gaussians.rigid_sh_dc = None
        if conservative:
            coverage = module.renderer(reference, torch.zeros_like(module.background),
                override_color=base['xyz'].new_ones((len(base['xyz']), 3)))['render'][0]
            residual, typed = typed_novel_residual(residual, reference.image,
                previous[Path(reference.path).name].image, (1-coverage).clamp(0, 1))
            typed_reports.append(dict(camera=Path(reference.path).name, **typed))
        regions = residual_regions(residual, reference.image, quota, config.residual_floor)
        if not regions:
            continue
        pixels = torch.cat([region['pixels'] for region in regions])
        distances = [(float((view.camera_center-reference.camera_center).square().sum()), view)
                     for view in fit if view is not reference]
        peers = [item[1] for item in sorted(distances, key=lambda item: item[0])[:3]]
        _, z = project(reference, base['xyz'])
        positive = z[z > 0]
        if not len(positive):
            continue
        near = max(float(positive.quantile(.01))*.5, extent*.05)
        far = min(float(positive.quantile(.99))*1.5, near+extent*8)
        if far <= near or len(peers) < 2:
            continue
        matched = match_depth(reference, peers, pixels, near, far, config.depth_planes)
        offset = 0
        for region in regions:
            count = len(region['pixels'])
            xyz, color, depth, uncertainty, cost, valid = [value[offset:offset+count] for value in matched]
            offset += count
            valid = valid & (cost < config.match_cost_limit)
            proposed_region_mass += region['total_mass']
            proposed_region_area += region['total_area']
            patches = region_surface_patches(reference, region, xyz, color, depth, uncertainty, cost, valid)
            for patch in patches:
                xyz_parts.append(patch['xyz'][None]); color_parts.append(patch['color'][None])
                scale_parts.append(patch['scale'][None]); rotation_parts.append(patch['rotation'][None])
                score_parts.append(patch['score'][None])
                replacement_parts.append(torch.full((1,), -1, device=device, dtype=torch.long))
                group_parts.append(torch.full((1,), next_group, device=device, dtype=torch.long))
                region_reports.append(dict(group=next_group, camera=Path(reference.path).name,
                                           **patch['diagnostic']))
                support_domains.append(patch['support'])
            next_group += 1
    raw = {name: value[:0].clone() for name, value in base.items()}
    scores = base['xyz'].new_empty(0)
    replacements = torch.empty(0, dtype=torch.long, device=device)
    group_ids = torch.empty(0, dtype=torch.long, device=device)
    if xyz_parts:
        xyz = torch.cat(xyz_parts); scores = torch.cat(score_parts)
        all_groups = torch.cat(group_parts)
        labels = all_groups.unique(sorted=True)
        members = [(all_groups == label).nonzero(as_tuple=True)[0] for label in labels]
        order = sorted(range(len(members)), key=lambda i: float(scores[members[i]].sum()), reverse=True)
        cells = (xyz/(config.voxel_extent*extent)).floor().long().tolist()
        seen, chosen_ids = set(), []
        for index in order:
            group = members[index].tolist()
            keys = {tuple(cells[row]) for row in group}
            if keys.intersection(seen) or len(chosen_ids)+len(group) > config.max_candidates-config.repair_candidates:
                continue
            # Deduplication and count budgets must preserve the same patch
            # lineage that later complete-scene acceptance will evaluate.
            seen.update(keys); chosen_ids.extend(group)
        ids = torch.tensor(chosen_ids, device=device, dtype=torch.long)
        raw['xyz'] = xyz[ids]
        raw['rotation'] = torch.cat(rotation_parts)[ids]
        raw['logscale'] = torch.cat(scale_parts)[ids].clamp_min(extent*1.e-7).log()
        raw['opacity'] = xyz.new_full((len(ids), 1), math.log(.1/.9))
        raw['sh_dc'] = ((torch.cat(color_parts)[ids]-.5)/.28209479177387814)[:, None]
        raw['sh_rest'] = base['sh_rest'].new_zeros((len(ids), *base['sh_rest'].shape[1:]))
        scores = scores[ids]
        replacements = torch.cat(replacement_parts)[ids]
        group_ids = all_groups[ids]
        region_reports = [region_reports[index] for index in chosen_ids]
        support_domains = [support_domains[index] for index in chosen_ids]
    spatial_count = len(raw['xyz'])
    sources = nearest_sources(raw['xyz'], base['xyz'])
    repair_score = repair_gradient.norm(dim=-1)
    eligible = (base['logscale'].exp().amax(-1) < .05*extent) & \
               (base['opacity'].sigmoid().flatten() > .01) & (repair_score > 0)
    repair_ids = eligible.nonzero(as_tuple=True)[0]
    if len(replacements):
        repair_ids = repair_ids[~torch.isin(repair_ids, replacements[replacements >= 0])]
    repair_ids = repair_ids[repair_score[repair_ids].argsort(descending=True, stable=True)]
    ranking = {}
    if conservative and config.repair_candidates:
        repair_ids, ranking = rank_repair_contrast(module, fit, repair_ids[:config.max_candidates], base, proposal_cache)
        repair_score.zero_()
        if len(repair_ids):
            repair_score[repair_ids] = repair_score.new_tensor(ranking['ranked_contrast_scores'])
    repair_ids = repair_ids[:config.repair_candidates]
    for name in RAW_FIELDS:
        raw[name] = torch.cat((raw[name], base[name][repair_ids]), 0)
    if len(repair_ids):
        raw['opacity'][spatial_count:].clamp_(-9.21024, 9.21024)
        # Score units only order spatial groups; independent views decide acceptance.
        normalized = repair_score[repair_ids]/repair_score[repair_ids].max().clamp_min(1.e-12)
        scores = torch.cat((scores, normalized*max(float(scores.max()) if len(scores) else 0., 1.e-4)))
    sources = torch.cat((sources, repair_ids))
    replacements = torch.cat((replacements, repair_ids))
    for row in repair_ids.tolist():
        support_domains.append(covariance_support(base, row))
    group_ids = torch.cat((group_ids, torch.arange(next_group, next_group+len(repair_ids), device=device)))
    diagnostics = dict(allocated_error_mass=proposed_region_mass, allocated_support_pixels=proposed_region_area,
        matched_anchor_assigned_error_mass=sum(report['error_mass'] for report in region_reports),
        matched_anchor_assigned_region_pixels=sum(report['support_pixels'] for report in region_reports),
        pixel_area_scope='reference_view_region_assignment_not_compositing_coverage',
        typed_novel=typed_reports, repair_contrast_ranking=ranking,
        region_count=next_group, retained_patch_count=spatial_count, patches=region_reports)
    return BirthTransaction(raw, sources, replacements, scores,
        raw['xyz'].clone(), raw['rotation'].clone(), raw['logscale'].exp().clone(),
        torch.zeros(len(sources), dtype=torch.int64, device=device),
        tuple(fit_names), tuple(validation_names), spatial_count, len(repair_ids), group_ids, diagnostics,
        support_domains)


def surface_transaction_from_specs(module, specs, fit_names, validation_names, diagnostics, previous=None):
    from .surface_patch_model import SurfacePatchModel
    from .birth_quantization import model_raw
    specs = BirthTransaction.tensor_tree(specs, module.device)
    temporary = SurfacePatchModel(module.config.sh_degree)
    temporary.active_sh_degree = module.gaussians.active_sh_degree
    temporary.spatial_lr_scale = module.gaussians.spatial_lr_scale
    temporary.install_patches(specs, old_count=len(module.gaussians._xyz), views=module.transaction_camera_views())
    temporary.set_retention_cap(.5)
    physical = {name:value.to(module.device) for name,value in model_raw(temporary).items()}
    state = temporary.patch_state()
    count = len(physical['xyz'])
    groups = torch.cat([torch.full((len(row),), index, device=module.device, dtype=torch.long)
                        for index, row in enumerate(temporary.patch_rows)]) if count else torch.empty(0, device=module.device, dtype=torch.long)
    scores = physical['xyz'].new_zeros(count)
    for index, row in enumerate(temporary.patch_rows):
        scores[row] = float(specs[index].get('score', 1.))/max(1, len(row))
    novel = sum(len(row) for row, old in zip(temporary.patch_rows, temporary.patch_old_rows) if not len(old))
    update = module.opacity_update
    result = BirthTransaction(physical, temporary.source_rows.to(module.device).clone(),
        torch.full((count,), -1, device=module.device, dtype=torch.long), scores,
        physical['xyz'].clone(), physical['rotation'].clone(), physical['logscale'].exp().clone(),
        torch.zeros(count, device=module.device, dtype=torch.long), tuple(fit_names), tuple(validation_names),
        novel, count-novel, groups, diagnostics, [], patch_state=state, patch_specs=specs,
        old_base_symbols=(previous.old_base_symbols.clone() if previous is not None else update.symbols().detach().clone()),
        old_base_opacity=(previous.old_base_opacity.clone() if previous is not None else module.gaussians._opacity.detach().clone()),
        old_render_reference=previous.old_render_reference if previous is not None else None)
    return result


@torch.no_grad()
def propose(module, config, views, fit_names, validation_names):
    if getattr(getattr(getattr(module,'config',None),'birth',None),'radiance_mode',None) in ('quick_source_residual','shadow_source_residual'):
        return propose_quick_source(module,config,views,fit_names,validation_names)
    if getattr(getattr(getattr(module,'config',None),'birth',None),'radiance_mode',None) in ('dual_predictive_reincarnation','source_residual'):
        return propose_predictive_reincarnations(module,config,views,fit_names,validation_names)
    if getattr(getattr(getattr(module, 'config', None), 'birth', None), 'radiance_mode', None) == 'support_reincarnation':
        return propose_reincarnations(module, config, views, fit_names, validation_names)
    if getattr(getattr(getattr(module, 'config', None), 'birth', None), 'radiance_mode', None) != 'surface_patch':
        return _propose_existing(module, config, views, fit_names, validation_names)
    from dataclasses import replace
    from pathlib import Path
    from .surface_patch_proposal import build_surface_patches
    from .birth_quantization import model_raw
    from .rigid_residual_tree import quaternion_rotate
    fitting = [view for view in views if Path(view.path).name in fit_names]
    evidence = cache_surface_patch_evidence(module, fitting)
    # Two fixed exploration generations: coarse capacity followed by supported
    # subdivision, with the original overall candidate limit unchanged.
    coarse_capacity = max(1, config.max_candidates//max(1, config.max_cohorts))
    specs, report = build_surface_patches(model_raw(module.gaussians), evidence,
                                        max_kernels=coarse_capacity, max_patches=config.max_groups)
    specs = list(specs)
    for spec in specs:
        spec['retention'] = .5
    # Independent, evidence-supported changed/uncovered geometry retains the
    # synchronized RGB path. It has no old ownership or takeover gate.
    novel = _propose_existing(module, replace(config, max_candidates=coarse_capacity, repair_candidates=0),
                              views, fit_names, validation_names)
    for row in range(novel.proposed_spatial):
        raw = {name:value[row].detach() for name,value in novel.raw.items()}
        basis = quaternion_rotate(raw['rotation'].expand(3, -1), torch.eye(3, device=module.device)).T
        order = raw['logscale'].argsort()
        frame = basis[:, order[[1, 2, 0]]]
        if torch.det(frame) < 0:
            frame[:, 0] *= -1
        support = novel.support_domains[row]
        uv = (support['xyz']-raw['xyz'])@frame[:, :2]
        minimum = module.datamodule.cameras_extent*1.e-6
        bounds = torch.stack((uv.amin(0), uv.amax(0)))
        bounds[0] = torch.minimum(bounds[0], bounds[1]-minimum)
        thickness = raw['logscale'][order[0]].exp()
        tangent_scale = raw['logscale'][order[1:]].exp()
        if bool((tangent_scale < thickness).any() or (tangent_scale > bounds[1]-bounds[0]).any() or
                (bounds[1]-bounds[0] < thickness).any()):
            continue  # Uncertain/non-planar novel support is not flattened by shrinking its normal.
        specs.append(dict(center=raw['xyz'], frame=frame, uv_bounds=bounds,
            thickness=thickness.reshape(1),
            kernel_uv=raw['xyz'].new_zeros((1, 2)), kernel_scale=tangent_scale[None],
            kernel_dc=raw['sh_dc'][None], kernel_sh_rest=raw['sh_rest'][None],
            prior_dc=raw['sh_dc'][None], prior_sh_rest=raw['sh_rest'][None],
            kernel_alpha=raw['opacity'].sigmoid().reshape(1, 1), old_rows=torch.empty(0,dtype=torch.long,device=module.device),
            source_rows=novel.sources[row:row+1], retention=1., score=float(novel.scores[row]),
            support_uv=uv, support_color=support['color'], support_mass=support['mass'], kind='independent_mvs'))
    specs.sort(key=lambda spec: float(spec.get('score', 1.))/len(spec['kernel_uv']), reverse=True)
    selected, count = [], 0
    for spec in specs:
        size = len(spec['kernel_uv'])
        if count+size <= coarse_capacity:
            selected.append(spec); count += size
    selected.sort(key=lambda spec: bool(len(spec['old_rows'])))
    diagnostics = dict(surface_proposal=report, independent_mvs=novel.region_diagnostics,
        coarse_capacity=coarse_capacity, selected_kernels=count,
        selected_patches=len(selected), old_data_mutated=False,
        patches=[dict(kernels=len(spec['kernel_uv']), owned_old_count=len(spec['old_rows']),
                      score=float(spec.get('score', 1.))) for spec in selected])
    return surface_transaction_from_specs(module, selected, fit_names, validation_names, diagnostics)


@torch.no_grad()
def refine_surface_transaction(module, config, views):
    """Add supported in-plane detail without changing charts or ownership."""
    specs = module.births.export_patch_specs()
    original = module.birth_transaction.patch_specs
    candidates = []
    for patch, (spec, evidence) in enumerate(zip(specs, original)):
        if not bool((spec['kernel_scale']/math.sqrt(2) >= spec['scale_bounds'][0]).all(-1).any()):
            continue
        support = evidence.get('support_uv')
        if support is None or not len(support):
            continue
        support = support.to(module.device)
        world = spec['center']+support@spec['frame'][:, :2].T
        error, count = support.new_zeros(len(support)), support.new_zeros(len(support))
        for name in module.birth_transaction.fit_names:
            view = views[name]
            grid, z = project(view, world)
            valid = (grid.abs() < 1).all(-1) & (z > 0)
            image = module.forward(view)['render'].clamp(0, 1)
            difference = (sample_image(image, grid)-sample_image(view.image, grid)).square().mean(-1)
            error += difference*valid; count += valid
        error /= count.clamp_min(1)
        assignment = torch.cdist(support, spec['kernel_uv']).argmin(-1)
        mass = torch.as_tensor(evidence.get('support_mass', torch.ones_like(error)), device=module.device).flatten()
        for kernel in range(len(spec['kernel_uv'])):
            if bool((spec['kernel_scale'][kernel]/math.sqrt(2) < spec['scale_bounds'][0]).any()):
                continue
            ids = ((assignment == kernel)&(count >= 2)).nonzero().flatten()
            if len(ids):
                score = float((error[ids]*mass[ids]).sum())
                if score > 1.e-12:
                    candidates.append((score, patch, kernel, ids))
    slots = config.max_candidates-len(module.births._xyz)
    candidates.sort(reverse=True, key=lambda entry: entry[0])
    choices = {(patch,kernel):ids for _,patch,kernel,ids in candidates[:max(0,slots)]}
    if not choices:
        return None, dict(split_parents=0, reason='no_supported_inplane_refinement')
    result = []
    for patch, spec in enumerate(specs):
        spec = dict(spec)
        sources, uv, scale = [], [], []
        support = original[patch].get('support_uv')
        for kernel in range(len(spec['kernel_uv'])):
            ids = choices.get((patch,kernel))
            if ids is None:
                sources.append(kernel); uv.append(spec['kernel_uv'][kernel]); scale.append(spec['kernel_scale'][kernel])
                continue
            points = support.to(module.device)[ids]
            if len(points) >= 2:
                axis = points.var(0,unbiased=False).argmax(); order = points[:,axis].argsort(); cut=max(1,len(order)//2)
                centers = [points[order[:cut]].mean(0),points[order[cut:]].mean(0)]
            else:
                offset = torch.zeros_like(points[0]); axis=spec['kernel_scale'][kernel].argmax()
                offset[axis]=spec['kernel_scale'][kernel,axis]*.5
                centers=[spec['kernel_uv'][kernel]+offset,spec['kernel_uv'][kernel]-offset]
            for center in centers:
                sources.append(kernel);uv.append(center.maximum(spec['uv_bounds'][0]).minimum(spec['uv_bounds'][1]));scale.append(spec['kernel_scale'][kernel]/math.sqrt(2))
        index = torch.tensor(sources,device=module.device)
        for key in ('kernel_dc','kernel_sh_rest','kernel_alpha','kernel_angle','prior_dc','prior_sh_rest','source_rows'):
            if key in spec:spec[key]=spec[key][index]
        spec['kernel_uv']=torch.stack(uv);spec['kernel_scale']=torch.stack(scale)
        for key in ('support_uv','support_color','support_mass','score','kind'):
            if key in original[patch]:spec[key]=original[patch][key]
        result.append(spec)
    tx = module.birth_transaction
    refined = surface_transaction_from_specs(module,result,tx.fit_names,tx.validation_names,tx.region_diagnostics,tx)
    return refined, dict(split_parents=len(choices), refined_count=len(refined.sources),
        shared_plane_ownership_preserved=True,optimizer_reset='entire_fixed_cohort')


def raw_covariance(raw, row):
    from .rigid_residual_tree import quaternion_rotate
    q = F.normalize(raw['rotation'][row], dim=-1)
    basis = quaternion_rotate(q.expand(3, -1), torch.eye(3, device=q.device)).T
    return basis@torch.diag(raw['logscale'][row].exp().square())@basis.T


def covariance_support(raw, row):
    """Deterministic probes of inherited support, not reconstructed geometry."""
    covariance = raw_covariance(raw, row)
    values, axes = torch.linalg.eigh(covariance)
    grid = raw['xyz'].new_tensor([[x, y] for y in (-1., 0., 1.) for x in (-1., 0., 1.)])
    points = raw['xyz'][row]+(grid*values[1:].sqrt())@axes[:, 1:].T
    rgb = (raw['sh_dc'][row, 0]*.28209479177387814+.5).clamp(0, 1)
    return dict(kind='inherited_covariance_prior', xyz=points,
                color=rgb.expand(len(points), -1).clone(), mass=points.new_ones(len(points)))


@torch.no_grad()
def refinement_evidence(module, views):
    """Remaining error at support probes, restricted to observed fit cameras."""
    transaction = module.birth_transaction
    if not transaction.support_domains:
        return []
    points = torch.cat([domain['xyz'] for domain in transaction.support_domains])
    known_color = torch.cat([domain['color'] for domain in transaction.support_domains])
    parents = torch.cat([torch.full((len(domain['xyz']),), index, device=points.device)
                         for index, domain in enumerate(transaction.support_domains)])
    is_mvs = torch.cat([torch.full((len(domain['xyz']),), domain['kind'] == 'mvs',
                                  device=points.device, dtype=torch.bool)
                       for domain in transaction.support_domains])
    errors, colors, masks = [], [], []
    for camera, name in enumerate(transaction.fit_names):
        view = views[name]
        grid, depth = project(view, points)
        predicted = module.forward(view)['render'].clamp(0, 1)
        target = sample_image(view.image, grid)
        error = (sample_image(predicted, grid)-target).square().mean(-1)
        valid = ((grid.abs() < 1).all(-1) & (depth > 0) &
                 ((transaction.observations[parents] & (1 << camera)) != 0))
        # MVS probes must still describe the same visible surface. Inherited
        # covariance probes have only source-level contribution evidence and
        # remain explicitly marked as a prior; final coverage guards apply.
        valid &= ~is_mvs | ((target-known_color).norm(dim=-1) < .2)
        errors.append(error); colors.append(target); masks.append(valid)
    errors, colors, masks = torch.stack(errors), torch.stack(colors), torch.stack(masks)
    counts = masks.sum(0)
    remaining = (errors*masks).sum(0)/counts.clamp_min(1)
    observed_color = colors.masked_fill(~masks[..., None], float('nan')).nanmedian(0).values
    observed_color = torch.where(torch.isfinite(observed_color), observed_color, known_color)
    result, offset = [], 0
    for domain in transaction.support_domains:
        count = len(domain['xyz']); select = slice(offset, offset+count); offset += count
        result.append(dict(error=remaining[select], color=observed_color[select],
                           valid=counts[select] >= 2))
    return result


@torch.no_grad()
def split_support(raw, row, domain, evidence):
    """Deterministic residual/color bisection inside an existing support domain."""
    ids = (evidence['valid'] & (evidence['error'] > 0) & (domain['mass'] > 0)).nonzero(as_tuple=True)[0]
    minimum = 3 if domain['kind'] == 'mvs' else 2
    if len(ids) < 2*minimum:
        return None
    points = domain['xyz'][ids]
    colors = evidence['color'][ids]
    weights = domain['mass'][ids]*evidence['error'][ids]
    if float(weights.sum()) <= 1.e-12:
        return None
    features = torch.cat((points, colors), -1)
    whole_color = (colors*weights[:, None]).sum(0)/weights.sum()
    whole_error = ((colors-whole_color).square().sum(-1)*weights).sum()
    covariance = raw_covariance(raw, row)
    inverse = torch.linalg.pinv(covariance)
    proposals = []
    for axis in range(features.shape[1]):
        ordered = features[:, axis].argsort(stable=True)
        cumulative = weights[ordered].cumsum(0)
        cut = int(torch.searchsorted(cumulative, cumulative[-1]*.5))+1
        cut = max(minimum, min(len(ids)-minimum, cut))
        parts = (ordered[:cut], ordered[cut:])
        means, cost = [], weights.new_zeros(())
        for part in parts:
            weight = weights[part]; color = colors[part]
            means.append((points[part]*weight[:, None]).sum(0)/weight.sum().clamp_min(1.e-12))
            mean = (color*weight[:, None]).sum(0)/weight.sum().clamp_min(1.e-12)
            cost += ((color-mean).square().sum(-1)*weight).sum()
        separation = (means[0]-means[1])@inverse@(means[0]-means[1])
        proposals.append((float(whole_error-cost), float(separation), parts))
    # Color explanation is the primary criterion; spatial separation breaks
    # ties for residual shapes with nearly constant appearance.
    _, _, parts = max(proposals, key=lambda item: (item[0], item[1]))
    children, domains = [], []
    ratios = torch.stack([weights[part].sum() for part in parts]); ratios /= ratios.sum()
    means = torch.stack([(points[part]*weights[part, None]).sum(0)/weights[part].sum()
                         for part in parts])
    if domain['kind'] != 'mvs':
        # Moment-conserving deterministic covariance subkernels. They inherit
        # geometry and never claim that these probe positions are new surfaces.
        center = (means*ratios[:, None]).sum(0)
        offsets = means-center
        between = (offsets*ratios[:, None]).T@offsets
        evals, evecs = torch.linalg.eigh(covariance)
        inverse_root = evecs@torch.diag(evals.clamp_min(1.e-12).rsqrt())@evecs.T
        maximum = torch.linalg.eigvalsh(inverse_root@between@inverse_root).max()
        if maximum >= 1:
            offsets *= ((1-1.e-4)/maximum).sqrt()
            between = (offsets*ratios[:, None]).T@offsets
        shared_covariance = covariance-between
        means = raw['xyz'][row]+offsets
    parent_alpha = raw['opacity'][row].sigmoid().clamp(1.e-6, 1-1.e-4)
    for index, part in enumerate(parts):
        weight = weights[part]; normalized = weight/weight.sum()
        child = {name: value[row].clone() for name, value in raw.items()}
        if domain['kind'] == 'mvs':
            delta = points[part]-means[index]
            child_covariance = (delta*normalized[:, None]).T@delta
            # Keep inherited normal thickness; tangent spread is measured from
            # the child's own verified anchors, not a preset expansion factor.
            evals, evecs = torch.linalg.eigh(covariance)
            child_covariance += evals[0].clamp_min(1.e-12)*(evecs[:, :1]@evecs[:, :1].T)
        else:
            child_covariance = shared_covariance
        values, axes = torch.linalg.eigh(child_covariance)
        variance_floor = (torch.linalg.eigvalsh(covariance).min().clamp_min(1.e-12)
                          if domain['kind'] == 'mvs' else 1.e-12)
        values = values.clamp_min(variance_floor)
        if torch.det(axes) < 0:
            axes[:, 0] *= -1
        child['xyz'] = means[index]
        child['rotation'] = matrix_quaternion(axes)
        child['logscale'] = values.sqrt().log()
        optical_depth = -torch.log1p(-parent_alpha)*ratios[index]
        optical_depth *= torch.det(covariance).clamp_min(1.e-24).sqrt()/values.prod().sqrt().clamp_min(1.e-12)
        child['opacity'] = torch.logit((-torch.expm1(-optical_depth)).clamp(1.e-4, 1-1.e-4))
        # Subdivision changes support, not intrinsic radiance. Pixel RGB mixes
        # visibility and angular appearance; the cohort-end response solver
        # alone may change DC. SH_rest is inherited from the same raw parent.
        children.append(child)
        selected = ids[part]
        domains.append(dict(kind=domain['kind'], xyz=domain['xyz'][selected].clone(),
                            color=colors[part].clone(), mass=domain['mass'][selected].clone()))
    return children, domains, float(weights.sum())


@torch.no_grad()
def refine_conservative_transaction(module, config, views):
    """Replace mature repair candidates by complete coarse+detail families."""
    from .birth_quantization import model_raw
    from .conservative_family import ConservativeFamilyModel
    from .rigid_residual_tree import quaternion_rotate
    transaction = module.birth_transaction
    if len(getattr(module.births, 'family_rows', ())):
        return None, dict(reason='family_probation_already_installed', split_parents=0)
    raw = model_raw(module.births)
    evidence = refinement_evidence(module, views)
    view = module.make_birth_view()
    decoded = {name: view.raw_value(name).detach() for name in RAW_FIELDS}
    decoded['rotation'] = F.normalize(decoded['rotation'], dim=-1)
    repairs, novel_options = [], []
    for row, (domain, observed) in enumerate(zip(transaction.support_domains, evidence)):
        split = split_support(raw, row, domain, observed)
        if split is None:
            continue
        if transaction.replacements[row] >= 0:
            repairs.append((row, split))
        else:
            novel_options.append((row, split))
    priority = lambda option: float(transaction.scores[option[0]])*option[1][2]
    repairs.sort(key=priority, reverse=True); novel_options.sort(key=priority, reverse=True)
    novel_count = transaction.proposed_spatial
    family_count = min(len(repairs), max(0, (config.max_candidates-novel_count)//3))
    chosen_repairs = dict(repairs[:family_count])
    remaining = config.max_candidates-novel_count-3*family_count
    chosen_novel = dict(novel_options[:max(0, remaining)])
    if not chosen_repairs and not chosen_novel:
        return None, dict(reason='no_supported_detail_split', split_parents=0)
    records, sources, replacements, scores, groups, domains = [], [], [], [], [], []
    family_rows, anchors, offsets, parent_rows = [], [], [], []
    spatial_count = 0
    for row in range(len(raw['xyz'])):
        if transaction.replacements[row] >= 0:
            if row not in chosen_repairs:
                continue  # Withdraw candidate; its original old coverage returns.
            anchor = {name: value[row].clone() for name, value in decoded.items()}
            children, _, _ = chosen_repairs[row]
            direction = (children[0]['xyz']-children[1]['xyz'])*.5
            q = anchor['rotation']
            basis = quaternion_rotate(q.expand(3, -1), torch.eye(3, device=q.device)).T
            # Family latents use original axis order with the normal removed,
            # not tangent axes sorted by their scale magnitudes.
            axes = torch.arange(3, device=q.device)
            axes = axes[axes != anchor['logscale'].argmin()]
            local = (basis.T@direction)[axes]
            family_rows.append([len(records), len(records)+1, len(records)+2])
            anchors.append(anchor); offsets.append(local); parent_rows.append(row)
            children = [{name:value.clone() for name,value in anchor.items()} for _ in range(3)]
            support = [transaction.support_domains[row]]*3
        elif row in chosen_novel:
            children, support, _ = chosen_novel[row]
        else:
            children = [{name:value[row].clone() for name,value in raw.items()}]
            support = [transaction.support_domains[row]]
        for child, domain in zip(children, support):
            records.append(child); domains.append(domain)
            sources.append(transaction.sources[row]); replacements.append(transaction.replacements[row])
            scores.append(transaction.scores[row]/len(children)); groups.append(transaction.group_ids[row])
            spatial_count += int(transaction.replacements[row] < 0)
    refined = {name:torch.stack([record[name] for record in records]) for name in RAW_FIELDS}
    family_state = None
    if family_rows:
        family_state = dict(rows=torch.tensor(family_rows, dtype=torch.long, device=module.device),
            anchor_raw={name:torch.stack([anchor[name] for anchor in anchors]) for name in RAW_FIELDS},
            tangent_offset=torch.stack(offsets), gate_logit=refined['xyz'].new_zeros((len(family_rows), 1)),
            contrast_dc=refined['xyz'].new_zeros((len(family_rows), 3)))
    temporary = ConservativeFamilyModel(module.config.sh_degree)
    for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
        setattr(temporary, attr, torch.nn.Parameter(refined[name].clone()))
    if family_state is not None:
        temporary.install_family_state(family_state)
    physical = model_raw(temporary)
    result = BirthTransaction(refined, torch.stack(sources), torch.stack(replacements), torch.stack(scores),
        physical['xyz'], physical['rotation'], physical['logscale'].exp(),
        torch.zeros(len(records), dtype=torch.int64, device=module.device),
        transaction.fit_names, transaction.validation_names, spatial_count, len(records)-spatial_count,
        torch.stack(groups), dict(transaction.region_diagnostics), domains, family_state=family_state)
    return result, dict(split_parents=family_count+len(chosen_novel), conservative_families=family_count,
        family_members=3*family_count, family_mature_coarse_rows=parent_rows,
        withdrawn_repair_candidates=transaction.proposed_repairs-family_count,
        split_novel_parents=len(chosen_novel), coarse_count=len(raw['xyz']), refined_count=len(records),
        optimizer_reset='entire_fixed_cohort', family_gate_probation='half-share frozen until terminal',
        repair_mean_radiance='fixed_mature_coarse_anchor')


@torch.no_grad()
def refine_transaction(module, config, views):
    """Build the next fixed-topology cohort; no incumbent changes are made."""
    from .birth_quantization import model_raw
    if getattr(module.config.birth, 'radiance_mode', None) == 'support_reincarnation':
        return refine_reincarnations(module, config, views)
    if getattr(module.config.birth, 'radiance_mode', None) == 'surface_patch':
        return refine_surface_transaction(module, config, views)
    if getattr(module.config.birth, 'radiance_mode', None) == 'conservative_detail':
        return refine_conservative_transaction(module, config, views)
    transaction = module.birth_transaction
    raw = model_raw(module.births)
    evidence = refinement_evidence(module, views)
    options = []
    for row, (domain, observed) in enumerate(zip(transaction.support_domains, evidence)):
        split = split_support(raw, row, domain, observed)
        if split is not None:
            options.append((row, split))
    options.sort(key=lambda item: item[1][2], reverse=True)
    slots = max(0, config.max_candidates-len(raw['xyz']))
    chosen = dict(options[:slots])
    if not chosen:
        return None, dict(split_parents=0)
    rows, sources, replacements, scores, groups, domains = [], [], [], [], [], []
    spatial = 0
    for row in range(len(raw['xyz'])):
        if row in chosen:
            children, support, _ = chosen[row]
        else:
            children = [{name: value[row].clone() for name, value in raw.items()}]
            support = [transaction.support_domains[row]]
        for child, domain in zip(children, support):
            rows.append(child); domains.append(domain)
            sources.append(transaction.sources[row]); replacements.append(transaction.replacements[row])
            scores.append(transaction.scores[row]/len(children)); groups.append(transaction.group_ids[row])
            spatial += int(row < transaction.proposed_spatial)
    refined = {name: torch.stack([row[name] for row in rows]) for name in RAW_FIELDS}
    result = BirthTransaction(refined, torch.stack(sources), torch.stack(replacements), torch.stack(scores),
        refined['xyz'].clone(), refined['rotation'].clone(), refined['logscale'].exp().clone(),
        torch.zeros(len(rows), dtype=torch.int64, device=module.device), transaction.fit_names,
        transaction.validation_names, spatial, len(rows)-spatial, torch.stack(groups),
        dict(transaction.region_diagnostics), domains)
    return result, dict(split_parents=len(chosen), split_spatial_parents=sum(row < transaction.proposed_spatial for row in chosen),
                        split_repair_parents=sum(row >= transaction.proposed_spatial for row in chosen),
                        coarse_count=len(raw['xyz']), refined_count=len(rows), optimizer_reset='entire_fixed_cohort')


@torch.no_grad()
def constrain_candidates(model, transaction, position_floor=0., rows=None):
    from .rigid_residual_tree import quaternion_rotate
    if not len(model._xyz):
        return
    if rows is None:
        rows = slice(None)
    q = F.normalize(transaction.seed_rotation[rows], dim=-1)
    inverse = q.clone(); inverse[:, 1:] *= -1
    local = quaternion_rotate(inverse, model._xyz[rows]-transaction.seed_xyz[rows])
    limit = (transaction.seed_scale[rows]*3).clamp_min(position_floor)
    model._xyz[rows] = transaction.seed_xyz[rows]+quaternion_rotate(q,local.clamp(-limit,limit))
    initial = transaction.seed_scale[rows].log()
    model._scaling[rows] = model._scaling[rows].maximum(initial-math.log(4)).minimum(initial+math.log(4))


class RawView:
    def __init__(self, raw, degree, basis, alpha=None):
        self.get_xyz = raw['xyz']
        self.get_rotation = F.normalize(raw['rotation'], dim=-1)
        self.get_scaling = raw['logscale'].exp()
        self.get_opacity = raw['opacity'].sigmoid() if alpha is None else alpha
        self.get_features = torch.cat((raw['sh_dc'], raw['sh_rest']), 1)
        self.active_sh_degree = self.max_sh_degree = degree
        self.basis = basis

    def get_covariance(self, scaling_modifier=1.):
        return self.basis.covariance_activation(self.get_scaling, scaling_modifier,
                                                self.get_rotation)


@dataclass
class CommitTrial:
    keep: torch.Tensor
    candidates: torch.Tensor
    sources: torch.Tensor
    raw: dict
    complete_bytes: int
    motion_bytes: int
    birth_bytes: int
    opacity_bytes: int
    old_symbols: torch.Tensor = None

    def capture(self):
        return BirthTransaction.tensor_tree(self.__dict__, 'cpu')

    @classmethod
    def restore(cls, state, device):
        return cls(**BirthTransaction.tensor_tree(state, device))


def selected_population(module, trial):
    """Keep the selected decoded scene provisional, with a fixed predictor.

    All old rows remain physically present. Rendering and quantization use the
    exact surviving decoded reference used by the selected real-codec trial.
    """
    transaction = module.birth_transaction
    count = len(trial.keep)
    from .birth_quantization import model_raw
    selected = trial.candidates
    # Preserve the fitted continuous preimage of the quantizer. In particular,
    # normalizing a quantized quaternion and quantizing it again is not
    # idempotent; using decoded q here silently changes the validated scene.
    raw = {name: value[selected].detach().clone()
           for name, value in model_raw(module.births).items()}
    decoded = {name: trial.raw[name][count:].detach().clone() for name in RAW_FIELDS}
    deleted = torch.ones(len(module.gaussians._xyz), dtype=torch.bool, device=module.device)
    deleted[trial.keep] = False
    domains = transaction.support_domains
    persistent = (module.births.subset(selected).capture()
                  if hasattr(module.births,'persistent_quantized_raw') else None)
    reincarnation = (module.births.reincarnation_subset(selected).reincarnation_state()
                     if hasattr(module.births, 'reincarnation_subset') else None)
    family = (module.births.family_subset(selected).family_state()
              if reincarnation is None and hasattr(module.births, 'family_subset') else None)
    patch = (module.births.patch_subset(selected).patch_state()
             if hasattr(module.births, 'patch_subset') else None)
    specs = transaction.patch_specs
    if patch is not None:
        mask = torch.zeros(len(module.births._xyz), device=module.device, dtype=torch.bool); mask[selected] = True
        retained = [index for index,row in enumerate(module.births.patch_rows) if bool(mask[row].all())]
        specs = [specs[index] for index in retained]
    reincarnation_specs = transaction.reincarnation_specs
    if reincarnation is not None:
        parents = set(transaction.replacements[selected].tolist())
        reincarnation_specs = [spec for spec in reincarnation_specs if int(spec['old_row']) in parents]
    result = BirthTransaction(raw, trial.sources.clone(), transaction.replacements[selected].clone(),
        transaction.scores[selected].clone(), decoded['xyz'], decoded['rotation'],
        decoded['logscale'].exp(), torch.zeros(len(selected), dtype=torch.int64, device=module.device),
        transaction.fit_names, transaction.validation_names,
        int((selected < transaction.proposed_spatial).sum()),
        int((selected >= transaction.proposed_spatial).sum()),
        transaction.group_ids[selected].clone() if transaction.group_ids is not None else None,
        dict(transaction.region_diagnostics), [domains[int(row)] for row in selected] if domains else [],
        deleted, trial.keep.clone(),
        {name: trial.raw[name][:count].detach().clone() for name in RAW_FIELDS}, family,
        patch_state=patch, patch_specs=specs,
        old_base_symbols=transaction.old_base_symbols, old_base_opacity=transaction.old_base_opacity,
        fixed_old_symbols=trial.old_symbols, old_render_reference=transaction.old_render_reference,
        reincarnation_state=reincarnation, reincarnation_specs=reincarnation_specs,
        crossfit_state=BirthTransaction.tensor_tree(transaction.crossfit_state, module.device),
        predictive_model=transaction.predictive_model,persistent_state=persistent)
    return result


@torch.no_grad()
def encode_trial(module, candidate_mask, delete_mask, encode=True):
    """Dry-run the real codecs without advancing any training/decoder state."""
    from .birth_quantization import model_raw, QuantizedNewGaussianView, materialize_quantized_gaussians, patch_old_state
    from .gaussian import BirthGaussianModel
    from .rigid_tree_codec import encode_motion, decode_motion
    from .rigid_tree_birth_codec import encode_birth_symbols, encode_birth_attributes, ATTRIBUTES
    from .sparse_updates import encode_opacity_delta
    selected = candidate_mask.nonzero(as_tuple=True)[0]
    transaction = getattr(module, 'birth_transaction', None)
    fixed = getattr(transaction, 'fixed_reference', None)
    if hasattr(module.births,'early_quantized_raw'):
        candidate=module.births.subset(selected)
    elif hasattr(module.births,'persistent_quantized_raw'):
        candidate=module.births.subset(selected)
    elif hasattr(module.births, 'reincarnation_subset'):
        candidate = module.births.reincarnation_subset(selected)
    elif hasattr(module.births, 'patch_subset'):
        candidate = module.births.patch_subset(selected)
    elif hasattr(module.births, 'family_subset'):
        candidate = module.births.family_subset(selected)
    else:
        candidate = BirthGaussianModel(module.config.sh_degree)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(candidate, attr, torch.nn.Parameter(
                getattr(module.births, attr).detach()[selected].clone(), requires_grad=False))
    old_symbols = None
    delete = delete_mask.clone()
    if hasattr(candidate, 'raw_old_alpha'):
        old_logits, _, old_symbols = patch_old_state(module, candidate, transaction)
        if fixed is None and len(candidate.owned_old_rows):
            owned = candidate.owned_old_rows
            delete[owned[old_logits[owned, 0] <= -100]] = True
    keep = (~delete).nonzero(as_tuple=True)[0]
    if not len(keep):
        raise ValueError('A transaction must retain an incumbent prediction reference')
    if fixed is not None:
        if not torch.equal(keep, transaction.fixed_keep):
            raise ValueError('Population refit cannot change its selected old-row structure')
        base = {name: value.detach().clone() for name, value in fixed.items()}
    else:
        reference = transaction.old_render_reference if getattr(transaction, 'old_render_reference', None) is not None else model_raw(module.gaussians)
        base = {name: value[keep] for name, value in reference.items()}
    if old_symbols is not None:
        base['opacity'] = old_logits[keep]
    if fixed is not None:
        sources=transaction.sources[selected].clone()
    elif getattr(candidate,'independent_birth',False):
        mapping=torch.full((len(delete),),-1,device=keep.device,dtype=torch.long);mapping[keep]=torch.arange(len(keep),device=keep.device)
        sources=mapping[transaction.sources[selected]]
        if bool((sources<0).any()):raise ValueError('Independent birth codec parent was deleted')
    else:
        sources=nearest_sources(model_raw(candidate)['xyz'],base['xyz'])
    families = getattr(candidate, 'family_rows', sources.new_empty((0, 3)))
    if len(families):
        if fixed is None:
            sources[families] = sources[families[:, :1]].expand(-1, 3)
        elif not bool((sources[families] == sources[families[:, :1]]).all()):
            raise ValueError('Selected family lost its shared surviving source')
    view = (QuantizedNewGaussianView(candidate, module.config.birth,
            module.datamodule.cameras_extent, base, sources) if len(selected) else None)
    raw, _ = materialize_quantized_gaussians(base, view)
    if not encode:
        return CommitTrial(keep, selected, sources, raw, 0, 0, 0, 0, old_symbols)
    motion = module.motion
    rt, rr, gt, gr = [value.detach().cpu() for value in motion._values(hard=True)]
    packet = encode_motion(motion.frame_reference_state, motion.frame_index, motion.extent,
        motion.previous_rows[keep].cpu(), motion.owner[keep].cpu(), motion.parents.cpu(),
        rt, rr, gt[keep.cpu()], gr[keep.cpu()], sources.cpu(),
        raw['xyz'][len(keep):].cpu(), raw['rotation'][len(keep):].cpu(),
        motion.config.motion_xyz_step_extent, motion.config.motion_rotation_step_rad,
        compression='auto', gs_translation_step_extent=motion.config.gs_xyz_step_extent,
        gs_rotation_step_rad=motion.config.gs_rotation_step_rad,
        spatial_open=motion.opened.cpu(), spatial_relative=motion.relative_regions,
        inherit_birth_rotation=False,
        birth_pose_steps=((module.config.birth.new_xyz_step_extent*module.datamodule.cameras_extent,module.config.birth.new_rotation_step)
            if getattr(candidate,'independent_birth',False) else None))
    decoded = decode_motion(packet, motion.frame_reference_state)
    raw['xyz'] = decoded.xyz.to(module.device)
    raw['rotation'] = decoded.rotation.to(module.device)
    birth_packet = b''
    if view is not None:
        appearance = {name: raw[name][len(keep):] for name in ATTRIBUTES}
        birth_packet = encode_birth_symbols(appearance, len(keep),
            symbols={name: view.symbols[name] for name in ATTRIBUTES},
            steps={name: view.steps[name] for name in ATTRIBUTES},
            predictor={name: view.predictor[name] for name in ATTRIBUTES},
            source_rows=sources)
        if birth_packet is None:
            birth_packet = encode_birth_attributes(appearance, len(keep))
    opacity_packet = b''
    if module.opacity_update is not None:
        update = module.opacity_update
        rows = update.rows[keep]
        opacity_packet = encode_opacity_delta(update.reference[rows],
            old_symbols[keep] if old_symbols is not None else update.symbols()[rows], frame_index=module.current_frame,
            step=update.step, mode=update.config.mode) or b''
    # Reserve more than the current manifest actually uses; actual final files
    # remain the authoritative storage measurement.
    size = len(packet)+len(birth_packet)+len(opacity_packet)+1024
    return CommitTrial(keep, selected, sources, raw, size,
                        len(packet), len(birth_packet), len(opacity_packet), old_symbols)


@torch.no_grad()
def preview_trial(module, candidate_mask, delete_mask, incumbent):
    """Cheap rejection filter. Acceptance still requires real codec validation."""
    trial = encode_trial(module, candidate_mask, delete_mask, encode=False)
    count = len(trial.keep)
    for name in ('xyz', 'rotation'):
        trial.raw[name][:count] = incumbent[name][trial.keep]
    return trial


@torch.no_grad()
def validation_errors(module, views, raw):
    previous = module.renderer.gaussians
    collecting = getattr(module.renderer, 'collect_contributions', False)
    try:
        module.renderer.gaussians = RawView(raw, module.config.sh_degree, module.gaussians)
        if hasattr(module.renderer, 'collect_contributions'):
            module.renderer.collect_contributions = False
        return torch.stack([(module.renderer(view, module.background)['render'].clamp(0, 1)
                            - view.image).square().mean() for view in views])
    finally:
        module.renderer.gaussians = previous
        if hasattr(module.renderer, 'collect_contributions'):
            module.renderer.collect_contributions = collecting


@torch.no_grad()
def fitting_errors(module, views, old_only):
    return torch.stack([(module.forward(view, old_only=old_only)['render'].clamp(0, 1)
                        - view.image).square().mean() for view in views])


def lineage_groups(transaction, ids, xyz, limit):
    """Spatially aggregate patches without separating a shared region lineage."""
    lineage = getattr(transaction, 'group_ids', None)
    if lineage is None:
        return [ids[local] for local in spatial_groups(xyz[ids], limit)]
    labels = lineage[ids].unique(sorted=True)
    if not len(labels):
        return []
    members = [ids[lineage[ids] == label] for label in labels]
    centers = torch.stack([xyz[member].mean(0) for member in members])
    return [torch.cat([members[int(index)] for index in group])
            for group in spatial_groups(centers, limit)]


@torch.no_grad()
def admit(module, config, validation_views, retirement_views=None, fitting_views=None,capture_prefixes=False,atomic_lineage=False):
    """Validate deaths, then accept beneficial candidate groups within byte cost.

    Every trial is a complete decoded scene.  Nothing mutates the incumbent;
    rejected proposals therefore cannot leave missing sources or partial edits.
    """
    transaction = module.birth_transaction
    device = module.device
    accepted = torch.zeros(len(module.births._xyz), dtype=torch.bool, device=device)
    deleted = torch.zeros(len(module.gaussians._xyz), dtype=torch.bool, device=device)
    best = encode_trial(module, accepted, deleted)
    if capture_prefixes:
        module._birth_admission_witnesses = [dict(trial=best.capture(),material='single trained pool; no calibration lambda')]
    incumbent = best.raw
    baseline = validation_errors(module, validation_views, best.raw)
    errors = baseline
    retirement_views = validation_views if retirement_views is None else retirement_views
    retirement_baseline = validation_errors(module, retirement_views, best.raw)
    retirement_errors = retirement_baseline
    full_mask = torch.ones_like(accepted)
    entity = (module.births.shadow_entity_mask() if hasattr(module.births,'shadow_entity_mask')
              else torch.ones_like(accepted))
    full_mask &= entity
    crossfit = getattr(transaction,'crossfit_state',None)
    if crossfit is not None and not crossfit.get('eligible',False):
        full_mask[:transaction.proposed_spatial] = False
    full_delete = deleted.clone()
    full_parents = transaction.replacements[full_mask]
    full_delete[full_parents[full_parents >= 0]] = True
    full_pool = encode_trial(module, full_mask, full_delete)
    full_pool_validation = validation_errors(module, validation_views, full_pool.raw)
    full_pool_fitting = (validation_errors(module, fitting_views, full_pool.raw)
                         if fitting_views is not None else None)
    baseline_fitting = (validation_errors(module, fitting_views, best.raw)
                        if fitting_views is not None else None)
    # An occluded point is not a dead point. Only currently observed points
    # whose optimized alpha became small are eligible for tested retirement.
    alpha = module.gaussians._opacity.sigmoid().flatten()
    canonical_zero = module.gaussians._opacity.flatten() <= -100.
    eligible = (canonical_zero | ((alpha < module.config.opacity_update.prune_threshold) &
                (module.motion.observations > 0))).nonzero(as_tuple=True)[0]
    eligible = eligible[alpha[eligible].argsort(stable=True)
                        [:min(config.max_candidates, len(alpha)-1)]]
    death_groups = spatial_groups(module.gaussians._xyz[eligible], min(8, config.max_groups))
    tested_deaths = 0
    for local_ids in death_groups:
        trial_delete = deleted.clone(); trial_delete[eligible[local_ids]] = True
        trial = encode_trial(module, accepted, trial_delete)
        candidate_errors = validation_errors(module, retirement_views, trial.raw)
        if bool(candidate_errors.mean() <= retirement_errors.mean()+1.e-10):
            deleted, best, retirement_errors = trial_delete, trial, candidate_errors
            if capture_prefixes:
                module._birth_admission_witnesses.append(dict(trial=best.capture(),material='single trained pool; no calibration lambda'))
        tested_deaths += len(local_ids)
    errors = validation_errors(module, validation_views, best.raw)
    incumbent_bytes = best.complete_bytes
    available = config.birth_budget_bytes if module.current_frame == 1 else min(
        config.birth_budget_bytes, max(2048, config.target_frame_bytes-incumbent_bytes))
    budget = incumbent_bytes+available
    # Actual compositing observations, not a frustum-only visibility count.
    distinct_views = sum((transaction.observations >> index) & 1
                         for index in range(len(transaction.fit_names)))
    physical_alpha=getattr(module.births,'get_opacity',None)
    if physical_alpha is None:physical_alpha=module.births._opacity.sigmoid()
    viable = (distinct_views >= 2) & \
             (physical_alpha.flatten() >= module.config.birth.opacity_threshold)
    lineage = getattr(transaction, 'group_ids', None)
    if lineage is not None:
        family_labels = set()
        family_rows = getattr(module.births, 'family_rows', None)
        if family_rows is not None and len(family_rows):
            family_labels = set(lineage[family_rows[:, 0]].tolist())
        for label in lineage.unique(sorted=True):
            members = lineage == label
            # A root subtree is admitted atomically. A low-alpha child may
            # remain part of its coverage solution, but every child must have
            # actually received observations in independent fitting cameras.
            observed = (distinct_views[members] >= 2)
            covered = observed.any() if int(label) in family_labels else observed.all()
            viable[members] = bool(covered and viable[members].any())
    if hasattr(module.births, 'patch_rows'):
        for rows, owned in zip(module.births.patch_rows, module.births.patch_old_rows):
            seen = bool((distinct_views[rows] >= 2).any())
            carrier = bool((module.motion.observations[owned] > 0).any()) if len(owned) else False
            viable[rows] = seen or carrier
    if crossfit is not None and not crossfit.get('eligible',False):
        viable[:transaction.proposed_spatial] = False
    elif crossfit is not None and transaction.proposed_spatial:
        # Real-codec transfer already tested the whole population. Do not
        # silently reduce it to a visibility-selected, unproved subset.
        viable[:transaction.proposed_spatial] = bool((distinct_views[:transaction.proposed_spatial]>=2).any())
    if hasattr(module.births,'reincarnation_groups'):
        for rows in module.births.reincarnation_groups:
            # A copied native support inherits its parent's actual visibility;
            # a low-mass detail child does not invalidate its whole mother.
            parents = transaction.replacements[rows].unique()
            parent_seen = bool((module.motion.observations[parents] > 0).any())
            viable[rows] = parent_seen or bool((distinct_views[rows]>=2).any())
    viable &= entity
    if hasattr(module.births,'persistent_quantized_raw'):
        # Replacing a persistent identity is optional. Unchanged physical
        # fields stay in their original row, without a false wire update.
        live=model_raw(module.births)
        reference=transaction.old_render_reference or model_raw(module.gaussians)
        for row,parent in enumerate(transaction.replacements.tolist()):
            if parent>=0 and all(torch.equal(live[name][row],reference[name][parent]) for name in RAW_FIELDS):
                viable[row]=False
    ids = viable.nonzero(as_tuple=True)[0]
    groups = []
    for repair in (False, True):
        cohort = ids[(transaction.replacements[ids] >= 0) == repair]
        if atomic_lineage and transaction.group_ids is not None:
            groups.extend(cohort[transaction.group_ids[cohort]==label]
                          for label in transaction.group_ids[cohort].unique(sorted=True))
        else:
            groups.extend(lineage_groups(transaction, cohort, module.births._xyz,
                                         max(1, config.max_groups//2)))
    groups.sort(key=lambda group: float(transaction.scores[group].sum()), reverse=True)
    rejected, evaluated, tested_repair_deaths, rejected_repair_deaths = 0, 0, 0, 0
    pending = list(groups)
    evaluation_limit = len(groups) if atomic_lineage else config.max_groups*4
    while pending and evaluated < evaluation_limit:
        group = pending.pop(0)
        mask = accepted.clone(); mask[group] = True
        parents = transaction.replacements[mask]
        trial_delete = deleted.clone(); trial_delete[parents[parents >= 0]] = True
        if bool(trial_delete.all()):
            rejected += len(group); continue
        count = len(module.gaussians._xyz)-int(trial_delete.sum())+int(mask.sum())
        if count > config.max_gaussians:
            allowed = False
            trial = None
        else:
            preview = preview_trial(module, mask, trial_delete, incumbent)
            candidate_errors = validation_errors(module, validation_views, preview.raw)
            allowed = improves(candidate_errors, errors, config.validation_tolerance)
            trial = None
        evaluated += 1
        if allowed:
            trial = encode_trial(module, mask, trial_delete)
            allowed = trial.complete_bytes <= budget
            if allowed:
                candidate_errors = validation_errors(module, validation_views, trial.raw)
                allowed = improves(candidate_errors, errors, config.validation_tolerance)
            new_deaths = trial_delete & ~deleted
            if allowed and bool(new_deaths.any()):
                # Compare the atomic replacement against the current complete
                # incumbent. Adding the old source on top of the new subtree
                # would be a different, double-opacity scene, not its baseline.
                before = validation_errors(module, retirement_views, best.raw)
                after = validation_errors(module, retirement_views, trial.raw)
                tested_repair_deaths += int(new_deaths.sum())
                allowed = bool(after.mean() <= before.mean()+1.e-10)
                if not allowed:
                    rejected_repair_deaths += int(new_deaths.sum())
        if allowed:
            accepted, best, errors = mask, trial, candidate_errors
            deleted = torch.ones_like(deleted); deleted[trial.keep] = False
            if capture_prefixes:
                module._birth_admission_witnesses.append(dict(trial=best.capture(),material='single trained pool; no calibration lambda'))
        elif len(group) > 1:
            # A bad candidate must not veto every useful point in its region.
            # Split failed groups, retaining real complete-scene validation.
            children = lineage_groups(transaction, group, module.births._xyz, 2)
            if len(children) > 1:
                pending.extend(children)
            else:
                rejected += len(group)
        else:
            rejected += 1
    # Include accepted repair/source deletions and births in this final-scene
    # metric; the earlier low-alpha-only errors are no longer the final scene.
    retirement_errors = validation_errors(module, retirement_views, best.raw)
    report = dict(proposed_spatial=transaction.proposed_spatial,
        proposed_repairs=transaction.proposed_repairs,
        observed_candidates=int(viable.sum()), accepted=int(accepted.sum()),
        accepted_spatial=int(accepted[:transaction.proposed_spatial].sum()),
        rejected=len(accepted)-int(accepted.sum()), retired=int(deleted.sum()),
        tested_low_alpha_deaths=tested_deaths,
        tested_repair_source_deaths=tested_repair_deaths,
        rejected_repair_source_deaths=rejected_repair_deaths,
        candidate_groups_evaluated=evaluated,
        validation_baseline_mse=baseline.tolist(), validation_committed_mse=errors.tolist(),
        retirement_baseline_mse=retirement_baseline.tolist(),
        retirement_committed_mse=retirement_errors.tolist(),
        trial_complete_bytes=best.complete_bytes, complete_byte_budget=budget,
        validation_camera_names=list(transaction.validation_names),
        distortion_objective='mean_mse_all_standard_training_cameras',
        external_test_used=False, mutation_before_acceptance=False)
    report['region_support'] = getattr(transaction, 'region_diagnostics', {})
    report['full_pool_validation_mse'] = full_pool_validation.tolist()
    report['full_pool_codec_bytes'] = full_pool.complete_bytes
    report['full_pool_opacity_bytes'] = full_pool.opacity_bytes
    report['full_pool_birth_bytes'] = full_pool.birth_bytes
    report['full_pool_fitting_mse'] = full_pool_fitting.tolist() if full_pool_fitting is not None else None
    report['full_pool_baseline_fitting_mse'] = baseline_fitting.tolist() if baseline_fitting is not None else None
    if capture_prefixes:
        report['accepted_prefix_witnesses'] = [dict(candidate_count=len(item['trial']['candidates']),
            survivor_count=len(item['trial']['keep']),codec_bytes=item['trial']['complete_bytes'],
            material=item['material']) for item in module._birth_admission_witnesses]
    if hasattr(module.births, 'patch_rows'):
        report['patch_takeover'] = [dict(kernels=len(row), owned_old_count=len(old),
            retention=float(module.births.patch_retention[index]), selected=bool(accepted[row].all()))
            for index,(row,old) in enumerate(zip(module.births.patch_rows,module.births.patch_old_rows))]
        report['selected_opacity_bytes'] = best.opacity_bytes
    return best, report


def _reincarnation_transaction(module, model, specs, novel_template, crossfit, diagnostics, previous=None):
    """One immutable original-support root is one atomic replacement lineage."""
    from .birth_quantization import model_raw
    raw = model_raw(model)
    if hasattr(model,'shadow_entity_mask'):
        # Entity masking is not the trainable preimage. Inactive mother leaves
        # must retain theta0 alpha for later accepted density competition.
        raw = {name:getattr(model,attr).detach().clone() for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES)}
    n = int(model.novel_mask.sum())
    if hasattr(model,'freeze_predictive_endpoints'):
        # Stage0 masks N in the phenotype, never its trainable initial raw.
        # Preserve the continuous preimage for later activation and codec replay.
        for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):
            raw[name][:n] = getattr(model,attr)[:n].detach()
    count = len(raw['xyz'])
    replacement = torch.full((count,), -1, dtype=torch.long, device=module.device)
    scores = raw['xyz'].new_zeros(count)
    groups = torch.empty(count, dtype=torch.long, device=module.device)
    # The transfer proof is for this entire N hypothesis. A later all20-based
    # subset would be a different population whose transfer was never proved.
    groups[:n] = 0
    scores[:n] = novel_template.scores[:n].to(module.device)
    first = int(groups[:n].max())+1 if n else 0
    domains = list(novel_template.support_domains[:n]) if novel_template.support_domains else [{} for _ in range(n)]
    for index,(rows,spec) in enumerate(zip(model.reincarnation_groups,specs)):
        rows = rows.to(module.device)
        replacement[rows] = int(spec['old_row'])
        groups[rows] = first+index
        scores[rows] = float(spec.get('responsibility',spec.get('score',0.)))/len(rows)
        domain = {key: spec[key] for key in ('support_world','support_color','support_mass','support_residual') if key in spec}
        domains.extend([domain for _ in rows])
    if len(specs) != len(model.reincarnation_groups):
        raise ValueError('Native-support roots and model lineages differ')
    sources = nearest_sources(raw['xyz'], module.gaussians._xyz)
    families = getattr(model,'family_rows',sources.new_empty((0,3)))
    if len(families):
        sources[families] = sources[families[:,:1]].expand(-1,3)
    observations = torch.zeros(count,dtype=torch.int64,device=module.device)
    result = BirthTransaction(raw,sources,replacement,scores,raw['xyz'].clone(),raw['rotation'].clone(),
        raw['logscale'].exp(),observations,tuple(novel_template.fit_names),tuple(novel_template.validation_names),
        n,count-n,groups,diagnostics,domains,
        reincarnation_state=model.reincarnation_state(),reincarnation_specs=specs,
        crossfit_state=BirthTransaction.tensor_tree(crossfit,module.device),
        old_render_reference=previous.old_render_reference if previous is not None else None)
    return result


@torch.no_grad()
def propose_reincarnations(module, config, views, fit_names, validation_names):
    from dataclasses import replace
    from .support_reincarnation_proposal import propose_support_reincarnations, camera_complementary_groups
    from .reincarnation_model import ReincarnationModel
    # Reserve two future detail slots for every P root. N never splits.
    coarse_capacity = max(1,config.max_candidates//config.max_cohorts)
    p_quota = min(config.repair_candidates,config.max_candidates//3,coarse_capacity)
    specs,report = (propose_support_reincarnations(module,views,source_quota=p_quota,prefilter=config.max_candidates)
                    if p_quota else ([],dict(selected_count=0)))
    n_capacity = max(1,min(coarse_capacity-len(specs),config.max_candidates-3*len(specs)))
    novel = _propose_existing(module,replace(config,max_candidates=n_capacity,repair_candidates=0),
                              views,fit_names,validation_names)
    model = ReincarnationModel(module.config.sh_degree)
    model.install_reincarnations(specs,novel_raw=novel.raw,views=views)
    (a,b),camera_report = camera_complementary_groups(views)
    crossfit = dict(camera_groups={'A':tuple(a),'B':tuple(b)},training_branch='A',
                    novel_initial_raw=BirthTransaction.tensor_tree(novel.raw,module.device),
                    novel_A_raw=None,novel_B_raw=None,eligible=None,
                    initial_novel_count=len(novel.sources),
                    initialization_scope='MVS may use all20 train cameras',
                    transfer_scope='complementary views excluded from subsequent gradient optimization, not statistical image holdout')
    diagnostics = dict(native_support=report,independent_geometry=novel.region_diagnostics,
        camera_partition=camera_report,coarse_capacity=coarse_capacity,
        population_rule='P original native support; N independent complementary geometry fits',
        independent_hypothesis_acceptance='one atomic N population; one bad candidate can veto useful local births',
        reserved_detail_slots=2*len(specs),external_test_used=False)
    return _reincarnation_transaction(module,model,specs,novel,crossfit,diagnostics)


def _copy_novel_raw(model, raw):
    """Change a hypothesis preimage, never the incumbent or immutable prior."""
    n = int(model.novel_mask.sum())
    if any(len(value)!=n for value in raw.values()):
        raise ValueError('Independent hypotheses changed their initial N population')
    with torch.no_grad():
        for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):
            getattr(model,attr)[:n].copy_(raw[name].to(getattr(model,attr)))


@torch.no_grad()
def refine_reincarnations(module,config,views):
    from .birth_quantization import model_raw
    transaction = module.birth_transaction
    crossfit = BirthTransaction.tensor_tree(transaction.crossfit_state,module.device)
    n = transaction.proposed_spatial
    crossfit['novel_A_raw'] = {name:value[:n].clone() for name,value in model_raw(module.births).items()}
    crossfit['training_branch'] = 'B'
    specs = transaction.reincarnation_specs
    roots, invalid_offsets = [], 0
    for index,spec in enumerate(specs):
        if not spec.get('split_eligible',False):
            continue
        scale = torch.as_tensor(spec['raw']['logscale'],device=module.device).exp()
        axes = torch.arange(3,device=module.device);axes=axes[axes!=scale.argmin()]
        offset = torch.as_tensor(spec['tangent_offset'],device=module.device,dtype=scale.dtype)
        if not bool(torch.isfinite(offset).all() and ((offset/scale[axes]).square().sum()<1)):
            invalid_offsets += 1
            continue  # Retain its full native coarse support, never thin it.
        roots.append(index)
    roots = roots[:max(0,(config.max_candidates-len(module.births._xyz))//2)]
    # A second stage is required even when P has no high-frequency evidence:
    # N-B is a genuinely independent fit from the same original MVS values.
    offsets = torch.stack([torch.as_tensor(specs[index]['tangent_offset'],device=module.device,
                              dtype=module.gaussians._xyz.dtype) for index in roots]) if roots else None
    model = module.births.install_families_from_coarse(roots,tangent_offsets=offsets)
    _copy_novel_raw(model,crossfit['novel_initial_raw'])
    result = _reincarnation_transaction(module,model,specs,transaction,crossfit,
                                       dict(transaction.region_diagnostics),previous=transaction)
    return result,dict(split_reincarnated_roots=len(roots),refined_count=len(result.sources),
        unsupported_detail_offsets_kept_coarse=invalid_offsets,
        independent_novel_count=n,novel_geometry_restart='same_initial_MVS_raw',
        fitting_group='B',P_camera_scope='all20',optimizer_reset='entire_fixed_cohort')


@torch.no_grad()
def qualify_crossfit_novel(module,views,crossfit=None,candidate_mask=None,delete_mask=None,preferred_branch=None):
    """Both independently fitted hypotheses must transfer in the SAME scene.

    P is held at this option's learned full population. Each N increment is
    compared with the identical P-only true-codec scene, on cameras excluded
    from that hypothesis's geometry AND appearance fit. Test images are absent.
    """
    transaction = module.birth_transaction
    state = transaction.crossfit_state if crossfit is None else crossfit
    n = transaction.proposed_spatial
    if not n:
        transaction.crossfit_state = BirthTransaction.tensor_tree(state,module.device)
        transaction.crossfit_state['eligible'] = False
        return dict(eligible=False,reason='no_independent_candidates')
    a = state.get('novel_A_raw')
    b = state.get('novel_B_raw')
    if a is None or b is None:
        raise ValueError('Independent N birth requires both full complementary cohorts')
    full = (torch.ones(len(module.births._xyz),dtype=torch.bool,device=module.device)
            if candidate_mask is None else candidate_mask.clone())
    if not bool(full[:n].all()):
        raise ValueError('Optimization-transfer proof cannot accept a partial N hypothesis')
    without = full.clone();without[:n] = False
    deleted = (torch.zeros(len(module.gaussians._xyz),dtype=torch.bool,device=module.device)
               if delete_mask is None else delete_mask.clone())
    parents = transaction.replacements[full];deleted[parents[parents>=0]]=True
    baseline = encode_trial(module,without,deleted)
    baseline_errors = validation_errors(module,list(views.values()),baseline.raw)
    names = list(views)
    trials,errors,proof = {},{},{}
    for branch,raw in (('A',a),('B',b)):
        _copy_novel_raw(module.births,raw)
        trial = encode_trial(module,full,deleted)
        error = validation_errors(module,list(views.values()),trial.raw)
        heldout = state['camera_groups']['B' if branch=='A' else 'A']
        rows = [names.index(name) for name in heldout]
        improves_heldout = bool(error[rows].mean()<baseline_errors[rows].mean()-1.e-10)
        proof[branch] = dict(fitted_cameras=list(state['camera_groups'][branch]),
            heldout_cameras=list(heldout),baseline_mse=baseline_errors[rows].tolist(),
            hypothesis_mse=error[rows].tolist(),heldout_mean_gain=float(baseline_errors[rows].mean()-error[rows].mean()),
            transferable=improves_heldout,full_train_mse=error.tolist(),codec_bytes=trial.complete_bytes)
        trials[branch],errors[branch] = trial,error
    eligible = all(value['transferable'] for value in proof.values())
    chosen = preferred_branch or min(errors,key=lambda key:float(errors[key].mean()))
    _copy_novel_raw(module.births,a if chosen=='A' else b)
    transaction.crossfit_state = BirthTransaction.tensor_tree(state,module.device)
    transaction.crossfit_state.update(eligible=eligible,selected_branch=chosen,proof=proof)
    report = dict(eligible=eligible,selected_branch=chosen,hypotheses=proof,
        geometry_proof='independent_shared_MVS_restart_and_complementary_camera_optimization_transfer',
        initialization_scope='MVS may use all20 train cameras; transfer groups exclude optimization gradients only',
        P_scene='same_current_complete_P_population',external_test_used=False)
    report['P_scene'] = ('actual_selected_P_and_old_survivors' if candidate_mask is not None else 'same_current_complete_P_population')
    report['candidate_count'] = int(full.sum())
    return report


@torch.no_grad()
def propose_predictive_reincarnations(module,config,views,fit_names,validation_names):
    """Same native responsibility/initial MVS support, new independent lifecycle."""
    from .predictive_reincarnation_model import PredictiveReincarnationModel
    model_type = PredictiveReincarnationModel
    if module.config.birth.radiance_mode=='source_residual':
        from .source_residual_model import SourceResidualModel
        model_type = SourceResidualModel
    original = propose_reincarnations(module,config,views,fit_names,validation_names)
    model = model_type(module.config.sh_degree).install_reincarnations(
        original.reincarnation_specs,novel_raw=original.crossfit_state['novel_initial_raw'],views=views)
    model.set_training_stage(0)
    state = BirthTransaction.tensor_tree(original.crossfit_state,module.device)
    state.update(eligible=False,endpoint_scope='P_whole_family_only_N_current_constant',
        protocol='dual independent own-view training; conditional cross-calibration, not unseen initialization holdout')
    result = _reincarnation_transaction(module,model,original.reincarnation_specs,original,state,
        dict(original.region_diagnostics))
    result.predictive_model = True
    return result


@torch.no_grad()
def propose_quick_source(module,config,views,fit_names,validation_names):
    """One theta0-native P pool; no common fit, MVS, folds or late births."""
    from types import SimpleNamespace
    from .quick_source_residual_model import QuickSourceResidualModel
    from .support_reincarnation_proposal import propose_support_reincarnations
    quota = min(config.repair_candidates,config.max_candidates//3)
    specs,report = (propose_support_reincarnations(module,views,source_quota=quota,prefilter=config.max_candidates)
                    if quota else ([],dict(selected_count=0)))
    shadow = module.config.birth.radiance_mode=='shadow_source_residual'
    model_class = QuickSourceResidualModel
    if shadow:
        from .shadow_source_residual_model import ShadowSourceResidualModel
        model_class = ShadowSourceResidualModel
        # A no-detail clone has no innovation in this single trajectory.
        specs = [spec for spec in specs if spec.get('split_eligible',False)]
    empty = {name:value[:0].clone() for name,value in module_reference_raw(module).items()}
    model = model_class(module.config.sh_degree).install_reincarnations(specs,novel_raw=empty,views=views)
    roots = [index for index,spec in enumerate(specs) if spec.get('split_eligible',False)]
    alpha = module.config.birth.initial_alpha
    model = model.install_details_from_coarse(roots,detail_alpha=.1 if alpha is None else alpha)
    model.set_training_stage(1)
    template = SimpleNamespace(group_ids=torch.empty(0,dtype=torch.long,device=module.device),
        scores=module.gaussians._xyz.new_empty(0),support_domains=[],fit_names=tuple(fit_names),validation_names=tuple(validation_names))
    result = _reincarnation_transaction(module,model,specs,template,None,
        dict(native_support=report,method='single source-native theta0 mother plus independent D',
             common_fit=False,independent_MVS=False,training_camera_scope='all20 standard training views'))
    result.predictive_model = True
    return result


def module_reference_raw(module):
    from .birth_quantization import model_raw
    return model_raw(module.gaussians)

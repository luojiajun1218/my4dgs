#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from pathlib import Path
import functools


@functools.lru_cache(maxsize=2)
def _backend(stable_backward=False):
    from torch.utils.cpp_extension import load
    root = (Path(__file__).resolve().parents[2] / 'extensions' / 'observed_rasterizer').resolve()
    flag = ['-DMY4DGS_STABLE_COVARIANCE_BACKWARD'] if stable_backward else []
    name = 'my4dgs_observed_rasterizer_stable' if stable_backward else 'my4dgs_observed_rasterizer'
    return load(name=name,
        sources=[str(root / p) for p in ('cuda_rasterizer/rasterizer_impl.cu',
            'cuda_rasterizer/forward.cu', 'cuda_rasterizer/backward.cu',
            'rasterize_points.cu', 'ext.cpp')],
        extra_include_paths=[str(root / 'third_party/glm')],
        extra_cflags=['-O3', *flag], extra_cuda_cflags=['-O3', *flag], verbose=False)


@functools.lru_cache(maxsize=1)
def _regional_error_backend():
    """A separate extension: the historical observed rasterizer is untouched."""
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parents[2] / 'extensions' / 'regional_error_stats'
    return load(name='my4dgs_regional_error_stats',
        sources=[str(root / 'stats.cpp'), str(root / 'stats.cu')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3'], verbose=False)


class RegionalErrorStats:
    """One forward's saved buffers, consumed once before its backward pass.

    ``accumulate`` returns one numerator and denominator per input Gaussian;
    callers perform region ownership aggregation independently for each view.
    """

    def __init__(self):
        self._state = None
        self._captured = False

    def _capture(self, geom, binning, image, point_count, instance_count, height, width):
        if self._captured:
            raise RuntimeError('Regional error statistics already contain a forward')
        self._state = (geom, binning, image, point_count, instance_count, height, width)
        self._captured = True

    @torch.no_grad()
    def accumulate(self, error_map):
        if self._state is None:
            raise RuntimeError('Regional error statistics require an unconsumed forward')
        geom, binning, image, points, instances, height, width = self._state
        if not isinstance(error_map, torch.Tensor) or error_map.shape != (height, width):
            raise ValueError('Regional error map must have shape [height, width]')
        if error_map.dtype != torch.float32 or error_map.device != geom.device:
            raise ValueError('Regional error map must be float32 on the forward device')
        num, den = _regional_error_backend().accumulate(
            geom, binning, image, error_map.detach().contiguous(), points, instances)
        self._state = None
        return num.detach(), den.detach()


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    stats_sink=None,
    stable_backward=False,
):
    args = (
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )
    if stable_backward:
        return _RasterizeGaussians.apply(*args, stats_sink, True)
    # Keep the old nine-input autograd call unchanged when statistics are off.
    if stats_sink is None:
        return _RasterizeGaussians.apply(*args)
    return _RasterizeGaussians.apply(*args, stats_sink)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        stats_sink=None,
        stable_backward=False,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # The backend is selected per autograd context, so another renderer
        # cannot change which implementation a pending backward call uses.
        ctx.stable_backward = bool(stable_backward)
        backend = _backend(True) if ctx.stable_backward else _backend()
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, observed = backend.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, observed = backend.rasterize_gaussians(*args)

        ctx.has_stats_sink = stats_sink is not None
        if stats_sink is not None:
            stats_sink._capture(geomBuffer, binningBuffer, imgBuffer, means3D.shape[0],
                num_rendered, raster_settings.image_height, raster_settings.image_width)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        ctx.mark_non_differentiable(radii, observed)
        return color, radii, observed

    @staticmethod
    def backward(ctx, grad_out_color, _, __):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        backend = _backend(True) if ctx.stable_backward else _backend()
        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = backend.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = backend.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        if ctx.stable_backward:
            return grads + (None, None)
        return grads + (None,) if ctx.has_stats_sink else grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings, stable_backward=False):
        super().__init__()
        self.raster_settings = raster_settings
        self.stable_backward = stable_backward

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            backend = _backend(True) if self.stable_backward else _backend()
            visible = backend.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, stats_sink=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            stats_sink,
            self.stable_backward,
        )

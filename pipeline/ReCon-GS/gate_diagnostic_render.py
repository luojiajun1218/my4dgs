"""Separate frozen Gaussian renderer for counterfactual motion diagnostics."""

import copy

import torch

from libgs.metric import psnr, ssim
from libgs.renderer import Renderer

from .model.gaussian import GaussianModel
from .model.residual_motion import RAW_ATTRIBUTES


class FrozenImageObjective:
    def __init__(self, source, renderer_config, background, lambda_dssim, old_count):
        if old_count > len(source._xyz):
            raise ValueError('Diagnostic motion rows exceed frozen Gaussian rows')
        self.model = GaussianModel(source.max_sh_degree)
        self.model.active_sh_degree = source.active_sh_degree
        self.model.noise_scale = None
        self.model.render_attribute_cache = source.render_attribute_cache
        for attribute in RAW_ATTRIBUTES:
            # Only renderer reads use these detached attributes; no optimizer is installed.
            setattr(self.model, attribute, torch.nn.Parameter(
                getattr(source, attribute).detach(), requires_grad=False))
        self.renderer = Renderer(copy.deepcopy(renderer_config), self.model)
        self.background = background.detach()
        self.lambda_dssim = lambda_dssim
        self.old_count = old_count

    def evaluate(self, view, xyz, rotation):
        if xyz.shape != (self.old_count, 3) or rotation.shape != (self.old_count, 4):
            raise ValueError('Counterfactual poses must cover exactly the old Gaussian rows')
        self.model.rigid_xyz = torch.cat((xyz, self.model._xyz[self.old_count:]), 0)
        self.model.rigid_rotation = torch.cat((rotation, self.model._rotation[self.old_count:]), 0)
        try:
            image = self.renderer(view, self.background)['render']
            loss = ((1. - self.lambda_dssim) * torch.nn.functional.l1_loss(image, view.image)
                    + self.lambda_dssim * (1. - ssim(image, view.image)))
            return dict(image_loss=loss, psnr=psnr(image.detach(), view.image).mean())
        finally:
            self.model.rigid_xyz = self.model.rigid_rotation = None

    def __call__(self, view, xyz, rotation):
        return self.evaluate(view, xyz, rotation)['image_loss']

    def snapshot_state(self):
        return dict(max_sh_degree=self.model.max_sh_degree,
            active_sh_degree=self.model.active_sh_degree,
            render_attribute_cache=self.model.render_attribute_cache, old_count=self.old_count,
            attributes={name: getattr(self.model, name).detach() for name in RAW_ATTRIBUTES})

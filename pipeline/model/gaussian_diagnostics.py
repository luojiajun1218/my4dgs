"""诊断渲染使用的临时 GS 视图。

提供原始属性视图和点子集视图，用于检查渲染结果；
不负责修改模型中的 GS，也不承担正常训练的属性更新。
"""

import torch
from torch.nn import functional as F

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

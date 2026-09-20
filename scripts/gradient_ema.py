"""Bias-corrected renderer-gradient filter for isolated diagnostics only."""
import numpy as np


class GradientEMA:
    def __init__(self, beta=0.0):
        self.beta = float(beta)
        if not np.isfinite(self.beta) or not 0 <= self.beta < 1:
            raise ValueError('EMA beta must be finite and in [0, 1)')
        self.mean = None
        self.updates = 0

    def update(self, gradient):
        value = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite renderer gradient')
        if self.mean is not None and value.shape != self.mean.shape:
            raise ValueError('Gradient shape changed within a frame')
        if self.mean is None:
            self.mean = np.zeros_like(value)
        self.mean *= self.beta
        self.mean += (1. - self.beta) * value
        self.updates += 1
        return self.mean / (1. - self.beta ** self.updates)

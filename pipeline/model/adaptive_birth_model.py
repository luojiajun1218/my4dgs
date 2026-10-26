"""Independent detail Gaussians; the parent is a codec predictor, never replaced."""
from .early_replacement_model import EarlyReplacementModel, _logits


class AdaptiveBirthModel(EarlyReplacementModel):
    independent_birth = True
    # Disable the early replacement raw-render bypass: ordinary parent-offset
    # STE quantization is used for all six fields by QuantizedNewGaussianView.
    early_quantized_raw = None

    def birth_parameter_values(self):
        # Quantization needs all fields together. The inherited per-field hook
        # rebuilds the full pose and material graph for every requested field.
        return self.raw()

    def _material(self):
        return self._features_dc, self._features_rest

    def _raw_opacity(self):
        return _logits(self._opacity.clamp(0, 1))

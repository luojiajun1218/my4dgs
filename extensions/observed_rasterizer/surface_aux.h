/* Independent training-only surface evidence; legacy observed ABI unchanged. */
#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSurfaceAuxCUDA(
    const torch::Tensor& background, const torch::Tensor& means3D,
    const torch::Tensor& colors, const torch::Tensor& opacity,
    const torch::Tensor& scales, const torch::Tensor& rotations,
    const float scale_modifier, const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
    const float tan_fovx, const float tan_fovy, const int image_height,
    const int image_width, const torch::Tensor& sh, const int degree,
    const torch::Tensor& campos, const bool prefiltered, const bool debug);

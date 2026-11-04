/*
 * Visible-weight replay derived from the Inria observed rasterizer's forward
 * compositing loop (Copyright 2023 Inria / GRAPHDECO). The original project's
 * non-commercial research license is in ../observed_rasterizer/LICENSE.md.
 * This extension only reads its already-created forward buffers.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {
constexpr int BX = 16;
constexpr int BY = 16;
constexpr int BLOCK = BX * BY;
constexpr uintptr_t ALIGNMENT = 128;

// Layout mirrors the leading fields of GeometryState, BinningState and
// ImageState in observed_rasterizer/cuda_rasterizer/rasterizer_impl.cu.
// All pointers refer into the exact buffers of this view's forward pass.
template <typename T>
T* take(char*& chunk, size_t count) {
    const auto offset = (reinterpret_cast<uintptr_t>(chunk) + ALIGNMENT - 1)
                      & ~(ALIGNMENT - 1);
    auto* ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
    return ptr;
}

__global__ void replay_weights(
    const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
    const uint32_t* __restrict__ last_contributor,
    const float2* __restrict__ means2d, const float4* __restrict__ conic_opacity,
    const float* __restrict__ errors, int width, int height,
    float* __restrict__ numerator, float* __restrict__ denominator) {
    const int rank = threadIdx.y * BX + threadIdx.x;
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    const bool inside = x < width && y < height;
    const int pixel = y * width + x;
    const uint32_t last = inside ? last_contributor[pixel] : 0;
    const float error = inside ? errors[pixel] : 0.0f;
    bool done = !inside || last == 0;
    float transmittance = 1.0f;

    const int tiles_x = (width + BX - 1) / BX;
    const uint2 range = ranges[blockIdx.y * tiles_x + blockIdx.x];
    const int count = int(range.y - range.x);
    const int rounds = (count + BLOCK - 1) / BLOCK;
    __shared__ int ids[BLOCK];
    __shared__ float2 positions[BLOCK];
    __shared__ float4 conics[BLOCK];

    for (int round = 0; round < rounds; ++round) {
        if (__syncthreads_count(done) == BLOCK) break;
        const int offset = round * BLOCK + rank;
        if (offset < count) {
            const int id = int(point_list[range.x + offset]);
            ids[rank] = id;
            positions[rank] = means2d[id];
            conics[rank] = conic_opacity[id];
        }
        __syncthreads();
        const int entries = min(BLOCK, count - round * BLOCK);
        for (int j = 0; j < entries; ++j) {
            float weight = 0.0f;
            // Forward records the one-based position of the final accepted
            // contributor. Entries after it could not affect this pixel.
            if (!done && uint32_t(round * BLOCK + j + 1) > last) done = true;
            if (!done) {
                const float2 d = {positions[j].x - float(x), positions[j].y - float(y)};
                const float4 conic = conics[j];
                const float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y)
                                    - conic.y * d.x * d.y;
                if (!(power > 0.0f)) {
                    const float alpha = min(0.99f, conic.w * exp(power));
                    if (!(alpha < 1.0f / 255.0f)) {
                        const float next_t = transmittance * (1 - alpha);
                        if (next_t < 0.0001f) {
                            done = true;
                        } else {
                            weight = alpha * transmittance;
                            transmittance = next_t;
                        }
                    }
                }
            }

            // Every warp visits the same Gaussian; reduce pixels first to
            // limit global atomics. All 32 lanes participate, including
            // outside/done pixels whose weight is zero.
            float weighted_error = weight * error;
            for (int shift = 16; shift > 0; shift /= 2) {
                weight += __shfl_down_sync(0xffffffff, weight, shift);
                weighted_error += __shfl_down_sync(0xffffffff, weighted_error, shift);
            }
            if ((rank & 31) == 0 && weight != 0.0f) {
                atomicAdd(denominator + ids[j], weight);
                atomicAdd(numerator + ids[j], weighted_error);
            }
        }
    }
}

void check_bytes(const torch::Tensor& tensor, const torch::Device& device) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous()
                && tensor.scalar_type() == at::kByte,
                "Forward buffers must be contiguous uint8 tensors on the error-map device");
}

void check_end(char* start, char* end, const torch::Tensor& tensor) {
    TORCH_CHECK(end >= start && size_t(end - start) <= size_t(tensor.numel()),
                "Observed rasterizer buffer is too small for its declared dimensions");
}
} // namespace

std::tuple<torch::Tensor, torch::Tensor> regional_error_accumulate(
    const torch::Tensor& geom, const torch::Tensor& binning,
    const torch::Tensor& image, const torch::Tensor& errors,
    int64_t point_count, int64_t instance_count) {
    TORCH_CHECK(errors.is_cuda() && errors.is_contiguous() && errors.dim() == 2
                && errors.scalar_type() == at::kFloat,
                "Error map must be contiguous CUDA float32 [height, width]");
    TORCH_CHECK(point_count >= 0 && point_count <= std::numeric_limits<int>::max()
                && instance_count >= 0 && instance_count <= std::numeric_limits<int>::max(),
                "Invalid point or tile-instance count");
    TORCH_CHECK(errors.size(0) > 0 && errors.size(1) > 0 &&
                errors.numel() <= std::numeric_limits<int>::max(), "Invalid image dimensions");
    check_bytes(geom, errors.device());
    check_bytes(binning, errors.device());
    check_bytes(image, errors.device());
    const c10::cuda::CUDAGuard guard(errors.device());
    auto num = torch::zeros({point_count}, errors.options());
    auto den = torch::zeros({point_count}, errors.options());
    if (!point_count || !instance_count) return {num, den};

    char* geom_start = reinterpret_cast<char*>(geom.data_ptr<uint8_t>());
    char* g = geom_start;
    take<float>(g, point_count);       // depths
    take<bool>(g, point_count * 3);    // clamped
    take<int>(g, point_count);         // internal radii
    const float2* means = take<float2>(g, point_count);
    take<float>(g, point_count * 6);   // 3-D covariance
    const float4* conics = take<float4>(g, point_count);
    check_end(geom_start, g, geom);

    char* bin_start = reinterpret_cast<char*>(binning.data_ptr<uint8_t>());
    char* b = bin_start;
    const uint32_t* points = take<uint32_t>(b, instance_count);
    check_end(bin_start, b, binning);

    char* image_start = reinterpret_cast<char*>(image.data_ptr<uint8_t>());
    char* i = image_start;
    const auto pixels = errors.numel();
    take<float>(i, pixels);            // final transmittance
    const uint32_t* last = take<uint32_t>(i, pixels);
    const uint2* ranges = take<uint2>(i, pixels);
    check_end(image_start, i, image);

    const int h = int(errors.size(0)), w = int(errors.size(1));
    const dim3 block(BX, BY);
    const dim3 grid((w + BX - 1) / BX, (h + BY - 1) / BY);
    replay_weights<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        ranges, points, last, means, conics, errors.data_ptr<float>(), w, h,
        num.data_ptr<float>(), den.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {num, den};
}

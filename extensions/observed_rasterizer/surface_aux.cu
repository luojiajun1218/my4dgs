/*
 * Training-only dominant actual T*alpha evidence. The original forward,
 * backward, buffers and Python entry remain unchanged.
 * This file reuses the original implementation licensed in LICENSE.md.
 */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "rasterize_points.h"
#include "surface_aux.h"
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/auxiliary.h"
#include "cuda_rasterizer/rasterizer_impl.h"

namespace cg = cooperative_groups;

namespace {

__global__ void dominantAcceptedCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const float2* __restrict__ means2D,
    const float4* __restrict__ conic_opacity,
    const uint32_t* __restrict__ last_accepted,
    int width, int height,
    int* __restrict__ dominant_id,
    float* __restrict__ max_talpha)
{
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (width+BLOCK_X-1)/BLOCK_X;
    const uint2 pixel = {block.group_index().x*BLOCK_X+block.thread_index().x,
                         block.group_index().y*BLOCK_Y+block.thread_index().y};
    const bool inside = pixel.x < width && pixel.y < height;
    const uint32_t pixel_id = width*pixel.y+pixel.x;
    const uint32_t limit = inside ? last_accepted[pixel_id] : 0;
    bool done = !inside || limit == 0;
    const uint2 range = ranges[block.group_index().y*horizontal_blocks+block.group_index().x];
    int todo = range.y-range.x;
    const int rounds = (todo+BLOCK_SIZE-1)/BLOCK_SIZE;
    __shared__ int ids[BLOCK_SIZE];
    __shared__ float2 xy[BLOCK_SIZE];
    __shared__ float4 conic[BLOCK_SIZE];
    uint32_t position = 0;
    float transmission = 1.f;
    float best = 0.f;
    int best_id = -1;
    for (int round=0; round<rounds; ++round, todo-=BLOCK_SIZE) {
        if (__syncthreads_count(done) == BLOCK_SIZE)
            break;
        const int progress = round*BLOCK_SIZE+block.thread_rank();
        if (range.x+progress < range.y) {
            const int id = point_list[range.x+progress];
            ids[block.thread_rank()] = id;
            xy[block.thread_rank()] = means2D[id];
            conic[block.thread_rank()] = conic_opacity[id];
        }
        block.sync();
        for (int j=0; !done && j<min(BLOCK_SIZE,todo); ++j) {
            ++position;
            if (position > limit) { done=true; break; }
            const float2 d = {xy[j].x-(float)pixel.x, xy[j].y-(float)pixel.y};
            const float4 co = conic[j];
            const float power = -.5f*(co.x*d.x*d.x+co.z*d.y*d.y)-co.y*d.x*d.y;
            if (power > 0.f)
                continue;
            const float alpha = min(.99f, co.w*exp(power));
            if (alpha < 1.f/255.f)
                continue;
            // last_accepted comes from the original forward. Every passing
            // alpha before that native prefix end was truly composited, and
            // the terminating test_T<1e-4 item is never included here.
            const float weight = alpha*transmission;
            if (weight > best) { best=weight; best_id=ids[j]; }
            transmission = transmission*(1.f-alpha);
            if (position == limit)
                done = true;
        }
    }
    if (inside) {
        dominant_id[pixel_id] = best_id;
        max_talpha[pixel_id] = best;
    }
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeSurfaceAuxCUDA(
    const torch::Tensor& background, const torch::Tensor& means3D,
    const torch::Tensor& colors, const torch::Tensor& opacity,
    const torch::Tensor& scales, const torch::Tensor& rotations,
    const float scale_modifier, const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix,
    const float tan_fovx, const float tan_fovy, const int image_height,
    const int image_width, const torch::Tensor& sh, const int degree,
    const torch::Tensor& campos, const bool prefiltered, const bool debug)
{
    TORCH_CHECK(means3D.is_cuda() && means3D.scalar_type()==torch::kFloat32,
                "Surface aux requires float32 CUDA point tensors");
    TORCH_CHECK(image_height>0 && image_width>0, "Surface aux image dimensions must be positive");
    auto rendered = RasterizeObservedGaussiansCUDA(background, means3D, colors, opacity,
        scales, rotations, scale_modifier, cov3D_precomp, viewmatrix, projmatrix,
        tan_fovx, tan_fovy, image_height, image_width, sh, degree, campos, prefiltered, debug);
    const int count = means3D.size(0);
    const int instances = std::get<0>(rendered);
    auto options = means3D.options();
    auto id = torch::full({image_height,image_width}, -1, options.dtype(torch::kInt32));
    auto weight = torch::zeros({image_height,image_width}, options);
    auto transmission = torch::ones({image_height,image_width}, options);
    if (count > 0) {
        char* geometry_chunk = reinterpret_cast<char*>(std::get<3>(rendered).data_ptr());
        char* binning_chunk = reinterpret_cast<char*>(std::get<4>(rendered).data_ptr());
        char* image_chunk = reinterpret_cast<char*>(std::get<5>(rendered).data_ptr());
        auto geometry = ObservedCudaRasterizer::GeometryState::fromChunk(geometry_chunk,count);
        auto binning = ObservedCudaRasterizer::BinningState::fromChunk(binning_chunk,instances);
        auto image = ObservedCudaRasterizer::ImageState::fromChunk(image_chunk,image_height*image_width);
        // Return the actual native final T, not a newly reconstructed estimate.
        auto error = cudaMemcpy(transmission.data_ptr<float>(), image.accum_alpha,
            image_height*image_width*sizeof(float), cudaMemcpyDeviceToDevice);
        TORCH_CHECK(error==cudaSuccess, "Surface aux final-T copy failed: ",cudaGetErrorString(error));
        const dim3 grid((image_width+BLOCK_X-1)/BLOCK_X,(image_height+BLOCK_Y-1)/BLOCK_Y,1);
        const dim3 block(BLOCK_X,BLOCK_Y,1);
        dominantAcceptedCUDA<<<grid,block>>>(image.ranges,binning.point_list,geometry.means2D,
            geometry.conic_opacity,image.n_contrib,image_width,image_height,
            id.data_ptr<int>(),weight.data_ptr<float>());
        error = cudaGetLastError();
        TORCH_CHECK(error==cudaSuccess, "Surface aux dominant pass failed: ",cudaGetErrorString(error));
    }
    return std::make_tuple(std::get<1>(rendered),std::get<2>(rendered),
                           std::get<6>(rendered),id,weight,transmission);
}

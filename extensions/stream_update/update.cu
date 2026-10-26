#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
void tensor_check(const torch::Tensor& t, at::ScalarType type) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == type,
                "Stream update requires contiguous CUDA tensors of the expected type");
}

__device__ int upper_bound(const int64_t* data, int count, int64_t value) {
    int lo=0, hi=count;
    while (lo < hi) {
        int middle=(lo+hi)/2;
        if (data[middle] <= value) lo=middle+1; else hi=middle;
    }
    return lo;
}

__device__ int route(float x, float y, float z, const float* planes,
    const bool* opened, const double* ball, const double* leaves,
    const int64_t* leaf_nodes, int leaf_count, int depth, int64_t* paths) {
    int code=0, parent=0, offset=1;
    int first_leaf=0, width=1;
    for (int level=0; level<depth; ++level) {
        first_leaf += width; width *= 8;
    }
    bool far=false;
    if (leaf_count) {
        double dx=double(x)-ball[0],dy=double(y)-ball[1],dz=double(z)-ball[2];
        double distance=sqrt(__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz)));
        far=distance>ball[3];
    }
    if (far) {
        double best=INFINITY;
        int nearest=0;
        for (int i=0; i<leaf_count; ++i) {
            double dx=double(x)-leaves[3*i],dy=double(y)-leaves[3*i+1],dz=double(z)-leaves[3*i+2];
            double distance=__dadd_rn(__dadd_rn(__dmul_rn(dx,dx),__dmul_rn(dy,dy)),__dmul_rn(dz,dz));
            if (distance < best) { best=distance; nearest=i; }
        }
        code=int(leaf_nodes[nearest])-first_leaf;
        for (int level=0; level<depth; ++level) {
            paths[level]=offset+(code>>(3*(depth-1-level)));
            offset += 1<<(3*(level+1));
        }
    } else {
        for (int level=0; level<depth; ++level) {
            int octant=int(x>=planes[3*parent])+2*int(y>=planes[3*parent+1])+4*int(z>=planes[3*parent+2]);
            code=8*code+octant;
            parent=offset+code;
            paths[level]=parent;
            offset += 1<<(3*(level+1));
        }
    }
    int owner=int(paths[0]);
    bool reach=true;
    for (int level=1; level<depth; ++level) {
        reach=reach && opened[paths[level-1]];
        if (reach) owner=int(paths[level]);
    }
    return owner;
}

__global__ void old_kernel(float* bank, const int64_t* previous_ids,
    const int64_t* deleted_adjusted, int deleted_count, const int64_t* gs_rows,
    const int16_t* gs_symbols, int gs_count, float gs_step, const float* regions,
    const float* planes, const bool* opened, const double* ball,
    const double* leaves, const int64_t* leaf_nodes, int leaf_count,
    int64_t* ids, float* xyz, int64_t* owner, int64_t* paths, int count, int depth) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=count) return;
    int prior_row=row+upper_bound(deleted_adjusted,deleted_count,row);
    int64_t uid=previous_ids[prior_row];
    ids[row]=uid;
    float x=bank[3*uid],y=bank[3*uid+1],z=bank[3*uid+2];
    int node=route(x,y,z,planes,opened,ball,leaves,leaf_nodes,leaf_count,depth,paths+row*depth);
    owner[row]=node;
    int symbol=upper_bound(gs_rows,gs_count,row)-1;
    bool active=symbol>=0 && gs_rows[symbol]==row;
    for (int axis=0; axis<3; ++axis) {
        float delta=active ? __fmul_rn(float(gs_symbols[3*symbol+axis]),gs_step) : 0.f;
        float value=__fadd_rn(__fadd_rn(bank[3*uid+axis],regions[3*node+axis]),delta);
        bank[3*uid+axis]=value;
        xyz[3*row+axis]=value;
    }
}

__global__ void birth_kernel(float* bank, int64_t* ids, const int64_t* sources,
    const float* symbols, float step, int64_t next_uid, int old_count,
    const float* planes, const bool* opened, const double* ball,
    const double* leaves, const int64_t* leaf_nodes, int leaf_count,
    float* xyz, int64_t* owner, int64_t* paths, int count, int depth) {
    int birth=blockIdx.x*blockDim.x+threadIdx.x;
    if (birth>=count) return;
    int row=old_count+birth;
    int64_t uid=next_uid+birth,source=ids[sources[birth]];
    ids[row]=uid;
    for (int axis=0; axis<3; ++axis) {
        float value=__fadd_rn(bank[3*source+axis],__fmul_rn(symbols[7*birth+axis],step));
        bank[3*uid+axis]=value;
        xyz[3*row+axis]=value;
    }
    owner[row]=route(xyz[3*row],xyz[3*row+1],xyz[3*row+2],planes,opened,
        ball,leaves,leaf_nodes,leaf_count,depth,paths+row*depth);
}

void static_checks(torch::Tensor bank, torch::Tensor planes, torch::Tensor gates,
    torch::Tensor ball, torch::Tensor leaves, torch::Tensor nodes,
    torch::Tensor xyz, torch::Tensor owner, torch::Tensor paths) {
    tensor_check(bank,at::kFloat);tensor_check(planes,at::kFloat);
    tensor_check(gates,at::kBool);tensor_check(ball,at::kDouble);
    tensor_check(leaves,at::kDouble);tensor_check(nodes,at::kLong);
    tensor_check(xyz,at::kFloat);tensor_check(owner,at::kLong);tensor_check(paths,at::kLong);
    TORCH_CHECK(bank.dim()==2 && bank.size(1)==3 && planes.dim()==2 && planes.size(1)==3,
                "Invalid stream bank or split-plane shape");
    TORCH_CHECK(paths.dim()==2 && paths.size(1)>=2 && paths.size(1)<=5 &&
                xyz.dim()==2 && xyz.size(1)==3 && owner.dim()==1 &&
                paths.size(0)==xyz.size(0) && owner.size(0)==xyz.size(0),
                "Invalid stream output shapes");
    TORCH_CHECK(nodes.numel()==leaves.size(0) &&
                (nodes.numel()==0 || ball.numel()==4),"Invalid nearest-leaf dictionary");
    int planes_count=0,width=1;
    for (int level=0;level<paths.size(1);++level) { planes_count+=width;width*=8; }
    TORCH_CHECK(planes.size(0)==planes_count && gates.numel()==planes_count+width,
                "Invalid fixed octree dimensions");
    for (const auto& t : {planes,gates,ball,leaves,nodes,xyz,owner,paths})
        TORCH_CHECK(t.device()==bank.device(),"Stream tensors must share a CUDA device");
}
}

void advance_old(torch::Tensor bank, torch::Tensor previous_ids, torch::Tensor deleted,
    torch::Tensor gs_rows, torch::Tensor gs_symbols, torch::Tensor regions,
    double gs_step, torch::Tensor planes, torch::Tensor gates, torch::Tensor ball,
    torch::Tensor leaves, torch::Tensor nodes, torch::Tensor ids,
    torch::Tensor xyz, torch::Tensor owner, torch::Tensor paths, int64_t old_count) {
    static_checks(bank,planes,gates,ball,leaves,nodes,xyz,owner,paths);
    tensor_check(previous_ids,at::kLong);tensor_check(deleted,at::kLong);
    tensor_check(gs_rows,at::kLong);tensor_check(gs_symbols,at::kShort);
    tensor_check(regions,at::kFloat);tensor_check(ids,at::kLong);
    TORCH_CHECK(old_count>=1 && old_count<=xyz.size(0) &&
        previous_ids.numel()==old_count+deleted.numel(),"Invalid survivor counts");
    TORCH_CHECK(gs_symbols.dim()==2 && gs_symbols.size(0)==gs_rows.numel() &&
                gs_symbols.size(1)==3 && regions.size(0)==gates.numel() && regions.size(1)==3,
                "Invalid sparse-motion shapes");
    const c10::cuda::CUDAGuard guard(bank.device());
    old_kernel<<<(old_count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        bank.data_ptr<float>(),previous_ids.data_ptr<int64_t>(),deleted.data_ptr<int64_t>(),deleted.numel(),
        gs_rows.data_ptr<int64_t>(),gs_symbols.data_ptr<int16_t>(),gs_rows.numel(),float(gs_step),regions.data_ptr<float>(),
        planes.data_ptr<float>(),gates.data_ptr<bool>(),ball.data_ptr<double>(),leaves.data_ptr<double>(),
        nodes.data_ptr<int64_t>(),nodes.numel(),ids.data_ptr<int64_t>(),xyz.data_ptr<float>(),
        owner.data_ptr<int64_t>(),paths.data_ptr<int64_t>(),old_count,paths.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void append_births(torch::Tensor bank, torch::Tensor ids, torch::Tensor sources,
    torch::Tensor symbols, double step, int64_t next_uid, int64_t old_count,
    torch::Tensor planes, torch::Tensor gates, torch::Tensor ball, torch::Tensor leaves,
    torch::Tensor nodes, torch::Tensor xyz, torch::Tensor owner, torch::Tensor paths) {
    static_checks(bank,planes,gates,ball,leaves,nodes,xyz,owner,paths);
    tensor_check(ids,at::kLong);tensor_check(sources,at::kLong);tensor_check(symbols,at::kFloat);
    int64_t count=sources.numel();
    TORCH_CHECK(symbols.dim()==2 && symbols.size(0)==count && symbols.size(1)==7 &&
        old_count+count==xyz.size(0) && next_uid>=0 && next_uid+count<=bank.size(0),
        "Invalid newborn update shapes");
    if (!count) return;
    const c10::cuda::CUDAGuard guard(bank.device());
    birth_kernel<<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        bank.data_ptr<float>(),ids.data_ptr<int64_t>(),sources.data_ptr<int64_t>(),symbols.data_ptr<float>(),
        float(step),next_uid,old_count,planes.data_ptr<float>(),gates.data_ptr<bool>(),ball.data_ptr<double>(),
        leaves.data_ptr<double>(),nodes.data_ptr<int64_t>(),nodes.numel(),xyz.data_ptr<float>(),
        owner.data_ptr<int64_t>(),paths.data_ptr<int64_t>(),count,paths.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

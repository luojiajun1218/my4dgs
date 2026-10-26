#include <torch/extension.h>

void advance_old(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);
void append_births(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    double, int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("advance_old", &advance_old);
    m.def("append_births", &append_births);
}

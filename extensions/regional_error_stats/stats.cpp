#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> regional_error_accumulate(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("accumulate", &regional_error_accumulate,
          pybind11::call_guard<pybind11::gil_scoped_release>());
}

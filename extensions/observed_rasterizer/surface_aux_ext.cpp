/* Separate extension entry: no changes to ext.cpp or its legacy tuples. */
#include <torch/extension.h>
#include "surface_aux.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_surface_aux", &RasterizeSurfaceAuxCUDA);
}

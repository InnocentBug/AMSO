#include "dlpack_pybind.h"
#include <dlpack/dlpack.h>

namespace py = pybind11;

void pybind_dlpack(py::module_ &m) {
  py::enum_<DLDeviceType>(m, "DLDeviceType")
      .export_values(); // DLPack is C, so we don't have strongly tryped enums
}

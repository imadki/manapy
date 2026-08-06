// _core module entry point for the partitioning submodule of the
// manapy_compute_<float bits>_<int bits> packages (METIS-backed domain
// decomposition, ported from manapy's c_api). Compiled four times, once per
// precision pair, with MANAPY_COMPUTE_FLOAT_BITS / MANAPY_COMPUTE_INT_BITS
// selecting real_t/index_t -- and, through them, the METIS build this module
// links (see third_party/CMakeLists.txt).
//
// The c_api's geometry helpers, compute_cell_center_area_2d and
// compute_cell_center_volume_3d, are not here: they are mesh geometry and live
// in the `domain` submodule alongside the rest of it.

#include "manapy_compute_types.hpp" // pulls in Python.h

#include "bindings/registry.hpp"

NB_MODULE(_core, m) {
  m.doc() = "manapy domain partitioning helpers (METIS-backed) compiled for "
            "float" MANAPY_COMPUTE_STR(MANAPY_COMPUTE_FLOAT_BITS) " data and int"
            MANAPY_COMPUTE_STR(MANAPY_COMPUTE_INT_BITS) " indices";

  register_metis_partitioning(m);
  register_create_local_domains(m);
}

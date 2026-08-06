#include <algorithm>

#include "advecdiff_compute.cuh"
#include "common/advecdiff/explicitscheme_convective_2d_common.hpp"

namespace {

// One thread per face of a single face list, templated on the face kind AND the
// flux scheme (both compile-time), so the flux inlines with no per-face branch.
template <ConvectiveFaceKind Kind, FluxScheme Scheme>
__global__ void convective_2d_kernel(
    ArrayView<const index_t, 1> faces, ArrayView<real_t, 1> rez_w,
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> u_face,
    ArrayView<const real_t, 1> v_face, ArrayView<const real_t, 1> w_face,
    ArrayView<const real_t, 1> w_x, ArrayView<const real_t, 1> w_y,
    ArrayView<const real_t, 1> wx_halo, ArrayView<const real_t, 1> wy_halo,
    ArrayView<const real_t, 1> psi, ArrayView<const real_t, 1> psi_halo,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const index_t, 1> face_name,
    ArrayView<const real_t, 2> cell_shift, index_t order, index_t nfaces) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t k =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       k < nfaces; k += stride) {
    explicitscheme_convective_2d_face<Kind, Scheme>(
        faces(k), rez_w, w_c, w_ghost, w_halo, u_face, v_face, w_face, w_x, w_y,
        wx_halo, wy_halo, psi, psi_halo, cell_center, face_center, halo_centvol,
        face_cellid, face_normal, face_haloid, face_name, cell_shift, order);
  }
}

__global__ void zero_kernel(ArrayView<real_t, 1> rez_w, index_t nbcell) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t c =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       c < nbcell; c += stride) {
    rez_w(c) = real_t(0);
  }
}

int grid_blocks(index_t n, int threads) {
  return static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(n) + threads - 1) / threads, 65535));
}

// Launch one face-list kernel (no-op for an empty list).
template <ConvectiveFaceKind Kind, FluxScheme Scheme, typename... Args>
void launch_face_list(ArrayView<const index_t, 1> faces, cudaStream_t stream,
                      Args... args) {
  const index_t nfaces = static_cast<index_t>(faces.size(0));
  if (nfaces <= 0)
    return;
  constexpr int threads = 256;
  convective_2d_kernel<Kind, Scheme>
      <<<grid_blocks(nfaces, threads), threads, 0, stream>>>(faces, args...,
                                                             nfaces);
}

// Launch all four face lists for one compile-time flux scheme.
template <FluxScheme Scheme, typename... Args>
void launch_all(ArrayView<const index_t, 1> d_innerfaces,
                ArrayView<const index_t, 1> d_halofaces,
                ArrayView<const index_t, 1> d_boundaryfaces,
                ArrayView<const index_t, 1> d_periodicboundaryfaces,
                cudaStream_t stream, Args... args) {
  launch_face_list<ConvectiveFaceKind::Inner, Scheme>(d_innerfaces, stream,
                                                      args...);
  launch_face_list<ConvectiveFaceKind::Periodic, Scheme>(
      d_periodicboundaryfaces, stream, args...);
  launch_face_list<ConvectiveFaceKind::Halo, Scheme>(d_halofaces, stream,
                                                     args...);
  launch_face_list<ConvectiveFaceKind::Boundary, Scheme>(d_boundaryfaces,
                                                         stream, args...);
}

} // namespace

void launch_explicitscheme_convective_2d(
    ArrayView<real_t, 1> rez_w, ArrayView<const real_t, 1> w_c,
    ArrayView<const real_t, 1> w_ghost, ArrayView<const real_t, 1> w_halo,
    ArrayView<const real_t, 1> u_face, ArrayView<const real_t, 1> v_face,
    ArrayView<const real_t, 1> w_face, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> wx_halo,
    ArrayView<const real_t, 1> wy_halo, ArrayView<const real_t, 1> psi,
    ArrayView<const real_t, 1> psi_halo, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> face_cellid,
    ArrayView<const real_t, 2> face_normal,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_periodicboundaryfaces,
    ArrayView<const real_t, 2> cell_shift, index_t order, index_t scheme,
    cudaStream_t stream) {
  const index_t nbcell = static_cast<index_t>(rez_w.size(0));
  if (nbcell <= 0)
    return;

  constexpr int threads = 256;
  zero_kernel<<<grid_blocks(nbcell, threads), threads, 0, stream>>>(rez_w,
                                                                    nbcell);

  // Resolve the flux scheme once, here, then launch the four face-list kernels
  // of the chosen compile-time specialisation. The per-face argument pack is
  // identical across face kinds; only the face list and Kind/Scheme change.
#define MANAPY_ADVECDIFF_LAUNCH(S)                                             \
  launch_all<S>(d_innerfaces, d_halofaces, d_boundaryfaces,                    \
                d_periodicboundaryfaces, stream, rez_w, w_c, w_ghost, w_halo,  \
                u_face, v_face, w_face, w_x, w_y, wx_halo, wy_halo, psi,       \
                psi_halo, cell_center, face_center, halo_centvol, face_cellid, \
                face_normal, face_haloid, face_name, cell_shift, order)

  switch (scheme) {
  case index_t(0):
    MANAPY_ADVECDIFF_LAUNCH(FluxScheme::Upwind);
    break;
  case index_t(1):
    MANAPY_ADVECDIFF_LAUNCH(FluxScheme::Centered);
    break;
  case index_t(3):
    MANAPY_ADVECDIFF_LAUNCH(FluxScheme::LaxFriedrichs);
    break;
  default: // 2 = rusanov, and the safe fallback
    MANAPY_ADVECDIFF_LAUNCH(FluxScheme::Rusanov);
    break;
  }
#undef MANAPY_ADVECDIFF_LAUNCH
}

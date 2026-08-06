#include <algorithm>

#include "diffusion_compute.cuh"
#include "common/diffusion/time_step_common.hpp"

namespace {

// atomicMin on a real_t, valid only for NON-NEGATIVE values. The IEEE-754 bit
// pattern of a non-negative float/double is monotonic in the value, so a
// min over the reinterpreted unsigned integer is a min over the reals. Every
// time_step candidate is >= 0 (cfl, volume, the diffusion term and the 1e6 seed
// are all positive), so this is safe here.
__device__ inline void atomic_min_real(real_t *addr, real_t val) {
  if constexpr (sizeof(real_t) == sizeof(float)) {
    atomicMin(reinterpret_cast<unsigned int *>(addr), __float_as_uint(val));
  } else {
    atomicMin(reinterpret_cast<unsigned long long *>(addr),
              static_cast<unsigned long long>(__double_as_longlong(val)));
  }
}

// One thread per cell; a grid-stride loop covers meshes larger than one grid.
// Each thread computes its CFL candidate with the shared host/device routine
// and atomically folds it into the running minimum *dt (pre-seeded to
// TIME_STEP_NO_LIMIT by the launcher).
__global__ void time_step_kernel(real_t cfl,
                                 ArrayView<const real_t, 2> face_normal,
                                 ArrayView<const real_t, 1> cell_volume,
                                 ArrayView<const index_t, 2> cell_faceid,
                                 real_t Dxx, real_t Dyy, real_t Dzz,
                                 real_t *dt, index_t nbelement) {
  const index_t stride =
      static_cast<index_t>(gridDim.x) * static_cast<index_t>(blockDim.x);
  for (index_t i =
           static_cast<index_t>(blockIdx.x) * static_cast<index_t>(blockDim.x) +
           static_cast<index_t>(threadIdx.x);
       i < nbelement; i += stride) {
    const real_t cand = time_step_cell(i, cfl, face_normal, cell_volume,
                                       cell_faceid, Dxx, Dyy, Dzz);
    atomic_min_real(dt, cand);
  }
}

} // namespace

// Returns the CFL time step. Allocates a one-element device scalar, seeds it to
// TIME_STEP_NO_LIMIT, reduces into it with atomicMin, then copies it back. The
// stream is synchronised before the result is read.
real_t launch_time_step(real_t cfl, ArrayView<const real_t, 2> face_normal,
                        ArrayView<const real_t, 1> cell_volume,
                        ArrayView<const index_t, 2> cell_faceid, real_t Dxx,
                        real_t Dyy, real_t Dzz, cudaStream_t stream) {
  const index_t nbelement = static_cast<index_t>(cell_faceid.size(0));
  real_t host_dt = TIME_STEP_NO_LIMIT;
  if (nbelement <= 0)
    return host_dt;

  real_t *d_dt = nullptr;
  if (cudaMalloc(&d_dt, sizeof(real_t)) != cudaSuccess)
    return host_dt;
  cudaMemcpyAsync(d_dt, &host_dt, sizeof(real_t), cudaMemcpyHostToDevice,
                  stream);

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<std::int64_t>(
      (static_cast<std::int64_t>(nbelement) + threads - 1) / threads, 65535));
  time_step_kernel<<<blocks, threads, 0, stream>>>(
      cfl, face_normal, cell_volume, cell_faceid, Dxx, Dyy, Dzz, d_dt,
      nbelement);

  cudaMemcpyAsync(&host_dt, d_dt, sizeof(real_t), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_dt);
  return host_dt;
}

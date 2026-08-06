#pragma once

// Post-launch error handling for the CUDA bindings, shared by every
// manapy_compute module (core, boundary, solvers).
//
// Every `<kernel>_cuda_py` wrapper ends with
//
//     cuda_check(manapy_cuda_post_launch(), "<kernel> kernel launch");
//
// which checks the launch itself and returns WITHOUT waiting for the kernel to
// finish. That is the whole point: these bindings enqueue onto the legacy
// default stream (`stream = nullptr`), and CuPy's current stream is the legacy
// default stream too, so a manapy kernel, a CuPy elementwise op and a CuPy
// fancy-index gather issued back to back are already ordered against each
// other by the stream. A cudaDeviceSynchronize() after each launch buys no
// correctness -- it only forces the host to wait.
//
// Why that matters here: manapy runs under MPI, and the usual deployment puts
// several ranks on one GPU. Without MPS those ranks are separate CUDA
// contexts, which the driver time-slices, and a device-wide sync has to wait
// for a full scheduling round instead of just its own work. Measured on one
// RTX 4080 SUPER, per call:
//
//                              1 rank    4 ranks
//     launch + deviceSync       ~8 us     ~248 us
//     launch only               ~2 us       ~5 us
//
// At ~15 kernel calls per time step that was ~3.6 ms per iteration of pure
// waiting -- more than the rest of the loop put together.
//
// Where a sync IS required, it stays at the point that needs it, not here:
//
//   * a host read of the result -- `time_step_cuda` does its own
//     cudaStreamSynchronize() before copying the scalar back, and
//     ManapyArray._d2h() goes through cupy's blocking `.get()`;
//   * handing a device pointer to MPI -- NeighborCommunication.exchange()
//     synchronizes the current stream before the Neighbor_alltoallv.
//
// Debugging: an async launch reports a kernel fault at whatever call happens
// to check the error state next, which can be a confusing place. Set
//
//     MANAPY_CUDA_SYNC=1
//
// to restore a cudaDeviceSynchronize() after every launch, so faults are
// attributed to the kernel that actually caused them. Off by default; the
// environment is read once, on first use, so flipping it mid-process does
// nothing.
//
// Truthy values: anything except 0 / "" / false / no / off (case-insensitive).

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace manapy_cuda_detail {

inline bool sync_after_launch_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("MANAPY_CUDA_SYNC");
    if (value == nullptr)
      return false;

    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    return !(text.empty() || text == "0" || text == "false" || text == "no" ||
             text == "off");
  }();
  return enabled;
}

} // namespace manapy_cuda_detail

// Result of the launch, for the caller to pass to its own cuda_check().
//
// cudaGetLastError (not cudaPeekAtLastError) so the sticky CUDA error state is
// cleared here rather than bleeding into whatever unrelated call checks it
// next. Under MANAPY_CUDA_SYNC the device sync runs only once the launch
// itself is known to be good, so the reported error is always the first one.
inline cudaError_t manapy_cuda_post_launch() {
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess && manapy_cuda_detail::sync_after_launch_enabled())
    err = cudaDeviceSynchronize();
  return err;
}

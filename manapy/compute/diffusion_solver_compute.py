from functools import partial
from manapy.backends.ManapyArray import Device, ManapyArray
from manapy.backends.config import ManapyConfig
from manapy.compute._solver_common import _Solvers_common

class DiffusionSolverCompute(_Solvers_common):
  """Pure-diffusion counterpart of `AdvectionDiffusionSolverCompute`.

  Same two kernels as the advecdiff module minus the convective residual: the
  dissipative flux and a CFL time step that keeps only the diffusion term (no
  |u.n| contribution). Both are dimension-agnostic on the C side -- the unused
  axis simply carries a zero normal/gradient -- so `dim` only reaches the
  kernels as the parity argument of `time_step`, and no 2D/3D kernel pair has
  to be resolved here.
  """

  def __init__(self, config: ManapyConfig, dim: int):
    # Define self.config, self.dim, self.compute
    # Also define somme common functions like (update_new_value, initialisation_gaussian_2d, initialisation_gaussian_3d)
    super().__init__(config, dim)

    if config.device == Device.CUDA:
      acc = (ManapyArray.gpu_r, ManapyArray.gpu_rw, ManapyArray.gpu_w)
      k_time_step = self.compute.diffusion_time_step_cuda
      k_dissipative = self.compute.diffusion_explicitscheme_dissipative_cuda
    else:
      acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
      k_time_step = self.compute.diffusion_time_step
      k_dissipative = self.compute.diffusion_explicitscheme_dissipative

    self.explicitscheme_dissipative = partial(
      DiffusionSolverCompute.explicitscheme_dissipative, k_dissipative, acc)
    self.time_step = partial(DiffusionSolverCompute.time_step, k_time_step, acc)


  # ------------------------------------------------------------------ kernels

  @staticmethod
  def explicitscheme_dissipative(kernel, acc, wx_face, wy_face, wz_face,
                                 face_cellid, face_normal, face_name, dissip_w,
                                 Dxx, Dyy, Dzz):
    """explicitscheme_dissipative: anisotropic diffusion residual, one routine
    for both 2D and 3D (the unused axis simply has a zero normal/gradient). The
    kernel zeroes `dissip_w` over every cell before scattering the per-face flux
    into it (owner +q, interior neighbour -q), so it is write-only."""
    r, rw, w = acc
    kernel(
      r(wx_face), r(wy_face), r(wz_face), r(face_cellid), r(face_normal),
      r(face_name), w(dissip_w), Dxx, Dyy, Dzz
    )

  @staticmethod
  def time_step(kernel, acc, u, v, w_, cfl, face_normal, face_measure,
                cell_volume, cell_faceid, dim, Dxx, Dyy, Dzz):
    """Explicit CFL time step for pure diffusion: min over the cells of
    cfl * volume / lambda, where lambda sums only the diffusion term
    (Dxx+Dyy+Dzz) * ||n||^2 / volume over the cell's faces. Reads only, and
    returns the time step as a Python float -- the caller still has to reduce it
    across ranks. `u`, `v`, `w_`, `face_measure` and `dim` are unused by the
    computation and kept for signature parity with the advecdiff time step."""
    r, rw, w = acc
    return kernel(
      r(u), r(v), r(w_), cfl, r(face_normal), r(face_measure), r(cell_volume),
      r(cell_faceid), dim, Dxx, Dyy, Dzz
    )

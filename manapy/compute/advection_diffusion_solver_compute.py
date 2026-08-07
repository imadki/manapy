from functools import partial
from manapy.backends.ManapyArray import Device, ManapyArray
from manapy.backends.config import ManapyConfig
from manapy.compute._solver_common import _Solvers_common

class AdvectionDiffusionSolverCompute(_Solvers_common):
  def __init__(self, config: ManapyConfig, dim: int):
    # Define self.config, self.dim, self.compute
    # Also define somme common functions like (update_new_value, initialisation_gaussian_2d, initialisation_gaussian_3d)
    super().__init__(config, dim)

    if config.device == Device.CUDA:
      acc = (ManapyArray.gpu_r, ManapyArray.gpu_rw, ManapyArray.gpu_w)
      k_time_step = self.compute.advecdiff_time_step_cuda
      k_dissipative = self.compute.advecdiff_explicitscheme_dissipative_cuda
      if dim == 2:
        k_convective = self.compute.advecdiff_explicitscheme_convective_2d_cuda
      else:
        k_convective = self.compute.advecdiff_explicitscheme_convective_3d_cuda
    else:
      acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
      k_time_step = self.compute.advecdiff_time_step
      k_dissipative = self.compute.advecdiff_explicitscheme_dissipative
      if dim == 2:
        k_convective = self.compute.advecdiff_explicitscheme_convective_2d
      else:
        k_convective = self.compute.advecdiff_explicitscheme_convective_3d

    self.explicitscheme_convective = partial(
      AdvectionDiffusionSolverCompute.explicitscheme_convective, k_convective, acc)
    self.explicitscheme_dissipative = partial(
      AdvectionDiffusionSolverCompute.explicitscheme_dissipative, k_dissipative, acc)
    self.time_step = partial(AdvectionDiffusionSolverCompute.time_step, k_time_step, acc)


  # ------------------------------------------------------------------ kernels

  @staticmethod
  def explicitscheme_convective(kernel, acc, rez_w, w_c, w_ghost, w_halo,
                                u_face, v_face, w_face, w_x, w_y, w_z, wx_halo,
                                wy_halo, wz_halo, psi, psi_halo, cell_center,
                                face_center, halo_centvol, face_cellid,
                                face_normal, face_haloid, face_name,
                                d_innerfaces, d_halofaces, d_boundaryfaces,
                                d_periodicboundaryfaces, cell_shift, order,
                                scheme):
    """explicitscheme_convective_2d/3d: explicit finite-volume convective
    residual. Both kernels zero `rez_w` over every cell before scattering the
    face fluxes into it, so it is write-only. `w_z` / `wz_halo` are read by the
    3D kernel only; they stay in the signature for parity."""
    r, rw, w = acc
    kernel(
      w(rez_w), r(w_c), r(w_ghost), r(w_halo), r(u_face), r(v_face), r(w_face),
      r(w_x), r(w_y), r(w_z), r(wx_halo), r(wy_halo), r(wz_halo), r(psi),
      r(psi_halo), r(cell_center), r(face_center), r(halo_centvol),
      r(face_cellid), r(face_normal), r(face_haloid), r(face_name),
      r(d_innerfaces), r(d_halofaces), r(d_boundaryfaces),
      r(d_periodicboundaryfaces), r(cell_shift), order, scheme
    )

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
    """Explicit CFL time step: min over the cells of cfl * volume / lambda,
    where lambda sums, over the cell's faces, the convective speed |u.n| plus
    the diffusion term (Dxx+Dyy+Dzz) * ||n||^2 / volume. Reads only, and returns
    the time step as a Python float -- the caller still has to reduce it across
    ranks. `face_measure` and `dim` are unused by the computation and kept for
    signature parity."""
    r, rw, w = acc
    return kernel(
      r(u), r(v), r(w_), cfl, r(face_normal), r(face_measure), r(cell_volume),
      r(cell_faceid), dim, Dxx, Dyy, Dzz
    )

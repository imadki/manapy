"""Finite-volume kernels for the nonlinear (viscous) Burgers equation.

    u_t + div( f(u) ) = nu * lap(u),   f(u) = (u^2/2, u^2/2 [, u^2/2])

The convective machinery is the same MUSCL reconstruction used by the
advection-diffusion solver; the ONLY Burgers-specific piece is the numerical
flux body (a nonlinear Rusanov / local Lax-Friedrichs flux whose wave speed is
|u| instead of a prescribed advection velocity). Because the flux is a rebound
module global that numba inlines, the convective kernel is kept *local* to this
module (a bound `_compute_flux` in advecdiff's namespace must not be shared).

The flux-agnostic kernels (diffusion, time step, cell update) carry no such
global, so they are imported straight from the advecdiff module and recompiled
here -- no duplication.
"""
from manapy.backends.compile_fun import compile, compile_no_cache
import numpy as np

# Flux-agnostic kernels are reused verbatim (they never touch `_compute_flux`).
from manapy.solvers.advecdiff.fvm_utils_compute import (
  _explicitscheme_dissipative,
  _time_step,
  _update_new_value,
)


# --------------------------------------------------------------------------- #
# Numerical-flux bodies for the Burgers flux f(u) = u^2/2 (per direction).
# face_normal is area-weighted, so the normal flux is  F.n = (u^2/2)*(nx+ny+nz).
# setup(dim, scheme) compiles one body and binds it to the global `_compute_flux`
# that the convective kernel calls. Add a scheme by writing a body + registering.
# --------------------------------------------------------------------------- #
def _rusanov_flux(w_l: 'float', w_r: 'float', face_normal: 'float[:]', flux_w: 'float[:]'):
  # Rusanov / local Lax-Friedrichs for the convex scalar flux u^2/2.
  #   F = 0.5*(f_L + f_R).n - 0.5*alpha*(u_R - u_L),  alpha = max|f'(u).n| = |s|*max(|u_L|,|u_R|)
  s = face_normal[0] + face_normal[1] + face_normal[2]
  f_l = 0.5 * w_l * w_l * s
  f_r = 0.5 * w_r * w_r * s
  alpha = abs(s) * max(abs(w_l), abs(w_r))
  flux_w[0] = 0.5 * (f_l + f_r) - 0.5 * alpha * (w_r - w_l)


# "lax_friedrichs" == local Lax-Friedrichs == Rusanov for a scalar convex flux.
_FLUX_BODIES = {"rusanov": _rusanov_flux, "lax_friedrichs": _rusanov_flux}
_compute_flux = None
_current_scheme = None


# --------------------------------------------------------------------------- #
# Convective kernel: identical MUSCL reconstruction to advecdiff, but the flux
# is the *nonlinear* Burgers flux above (no external velocity field). Kept local
# so numba inlines this module's `_compute_flux`, not advecdiff's.
# --------------------------------------------------------------------------- #
def _explicitscheme_convective_2d(rez_w: 'float[:]', w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
                                  w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', wx_halo: 'float[:]',
                                  wy_halo: 'float[:]', wz_halo: 'float[:]', psi: 'float[:]', psi_halo: 'float[:]',
                                  cell_center: 'float[:,:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                  face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_haloid: 'int[:]',
                                  face_name: 'int[:]', d_innerfaces: 'int[:]', d_halofaces: 'int[:]',
                                  d_boundaryfaces: 'int[:]', d_periodicboundaryfaces: 'int[:]',
                                  cell_shift: 'float[:,:]', order: 'int'):
  center_left = np.zeros(2)
  center_right = np.zeros(2)
  r_l = np.zeros(2)
  r_r = np.zeros(2)

  normal = np.zeros(3)
  flux_w = np.zeros(1)

  rez_w[:] = 0.

  if order == 1:
    # Order-1 fast path: reconstruction term (order-1)*... = 0, so read only the
    # cell value and the face normal per face.
    for i in d_innerfaces:
      cl = face_cellid[i][0]; cr = face_cellid[i][1]
      _compute_flux(w_c[cl], w_c[cr], face_normal[i], flux_w)
      rez_w[cl] -= flux_w[0]
      rez_w[cr] += flux_w[0]
    for i in d_periodicboundaryfaces:
      cl = face_cellid[i][0]; cr = face_cellid[i][1]
      _compute_flux(w_c[cl], w_c[cr], face_normal[i], flux_w)
      rez_w[cl] -= flux_w[0]
    for i in d_halofaces:
      cl = face_cellid[i][0]
      _compute_flux(w_c[cl], w_halo[face_haloid[i]], face_normal[i], flux_w)
      rez_w[cl] -= flux_w[0]
    for i in d_boundaryfaces:
      cl = face_cellid[i][0]
      _compute_flux(w_c[cl], w_ghost[i], face_normal[i], flux_w)
      rez_w[cl] -= flux_w[0]
    return

  for i in d_innerfaces:
    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_c[face_cellid[i][1]]

    center_left[:] = cell_center[face_cellid[i][0]][0:2]
    center_right[:] = cell_center[face_cellid[i][1]][0:2]

    w_x_left = w_x[face_cellid[i][0]];
    w_x_right = w_x[face_cellid[i][1]]
    w_y_left = w_y[face_cellid[i][0]];
    w_y_right = w_y[face_cellid[i][1]]

    psi_left = psi[face_cellid[i][0]];
    psi_right = psi[face_cellid[i][1]]

    r_l[0] = face_center[i][0] - center_left[0];
    r_r[0] = face_center[i][0] - center_right[0];
    r_l[1] = face_center[i][1] - center_left[1];
    r_r[1] = face_center[i][1] - center_right[1];

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1])
    w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1])

    _compute_flux(w_l, w_r, normal, flux_w)

    rez_w[face_cellid[i][0]] -= flux_w[0]
    rez_w[face_cellid[i][1]] += flux_w[0]

  for i in d_periodicboundaryfaces:

    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_c[face_cellid[i][1]]

    center_left[:] = cell_center[face_cellid[i][0]][0:2]
    center_right[:] = cell_center[face_cellid[i][1]][0:2]

    w_x_left = w_x[face_cellid[i][0]];
    w_x_right = w_x[face_cellid[i][1]]
    w_y_left = w_y[face_cellid[i][0]];
    w_y_right = w_y[face_cellid[i][1]]

    psi_left = psi[face_cellid[i][0]];
    psi_right = psi[face_cellid[i][1]]

    if face_name[i] == 11 or face_name[i] == 22:
      r_l[0] = face_center[i][0] - center_left[0];
      r_r[0] = face_center[i][0] - center_right[0] - cell_shift[face_cellid[i][1]][0]
      r_l[1] = face_center[i][1] - center_left[1];
      r_r[1] = face_center[i][1] - center_right[1]

    if face_name[i] == 33 or face_name[i] == 44:
      r_l[0] = face_center[i][0] - center_left[0];
      r_r[0] = face_center[i][0] - center_right[0]
      r_l[1] = face_center[i][1] - center_left[1];
      r_r[1] = face_center[i][1] - center_right[1] - cell_shift[face_cellid[i][1]][1]

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1])
    w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1])

    _compute_flux(w_l, w_r, normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]

  for i in d_halofaces:
    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_halo[face_haloid[i]]

    center_left[:] = cell_center[face_cellid[i][0]][0:2]
    center_right[:] = halo_centvol[face_haloid[i]][0:2]

    w_x_left = w_x[face_cellid[i][0]];
    w_x_right = wx_halo[face_haloid[i]]
    w_y_left = w_y[face_cellid[i][0]];
    w_y_right = wy_halo[face_haloid[i]]

    psi_left = psi[face_cellid[i][0]];
    psi_right = psi_halo[face_haloid[i]]

    r_l[0] = face_center[i][0] - center_left[0];
    r_r[0] = face_center[i][0] - center_right[0];
    r_l[1] = face_center[i][1] - center_left[1];
    r_r[1] = face_center[i][1] - center_right[1];

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1])
    w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1])

    _compute_flux(w_l, w_r, normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]

  for i in d_boundaryfaces:
    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_ghost[i]
    center_left[:] = cell_center[face_cellid[i][0]][0:2]

    w_x_left = w_x[face_cellid[i][0]];
    w_y_left = w_y[face_cellid[i][0]];

    psi_left = psi[face_cellid[i][0]];

    r_l[0] = face_center[i][0] - center_left[0];
    r_l[1] = face_center[i][1] - center_left[1];

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1])
    w_r = w_r

    _compute_flux(w_l, w_r, normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]


############################################################################
# Nothing is compiled at import. Call setup(dim, scheme) once (uniformly on all
# MPI ranks) before using any kernel below; the solver does this in __init__.
############################################################################
_agnostic_done = False
_dims_done = set()


def setup(dim, scheme="rusanov"):
  global _agnostic_done
  if not _agnostic_done:
    global explicitscheme_dissipative, time_step, update_new_value
    explicitscheme_dissipative = compile(_explicitscheme_dissipative)
    time_step = compile(_time_step)
    update_new_value = compile(_update_new_value)
    _agnostic_done = True

  global _compute_flux, _current_scheme
  if scheme not in _FLUX_BODIES:
    raise ValueError(f"unknown scheme '{scheme}'; choose from {list(_FLUX_BODIES)}")
  if scheme != _current_scheme:
    # compile_no_cache: the convective kernel inlines this rebound global, so a
    # disk-cached compile keyed on source alone could reuse a stale flux binding.
    _compute_flux = compile_no_cache(_FLUX_BODIES[scheme])
    _current_scheme = scheme
    _dims_done.clear()

  if dim not in _dims_done:
    global explicitscheme_convective_2d
    if dim == 2:
      explicitscheme_convective_2d = compile_no_cache(_explicitscheme_convective_2d)
    else:
      raise NotImplementedError("Burgers solver currently supports dim == 2 only")
    _dims_done.add(dim)

from manapy.backends.compile_fun import compile, compile_no_cache
import numpy as np

def _explicitscheme_dissipative(wx_face: 'float[:]', wy_face: 'float[:]', wz_face: 'float[:]',
                               face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_name: 'int[:]',
                               dissip_w: 'float[:]', Dxx: 'float', Dyy: 'float', Dzz: 'float'):
  nbface = len(face_cellid)
  norm = np.zeros(3)
  dissip_w[:] = 0.

  for i in range(nbface):

    norm[:] = face_normal[i][:]
    q = Dxx * wx_face[i] * norm[0] + Dyy * wy_face[i] * norm[1] + Dzz * wz_face[i] * norm[2]

    flux_w = q

    if face_name[i] == 0:

      dissip_w[face_cellid[i][0]] += flux_w
      dissip_w[face_cellid[i][1]] -= flux_w

    else:
      dissip_w[face_cellid[i][0]] += flux_w

# Numerical-flux bodies. setup(dim, scheme) compiles one and binds it to the
# global `_compute_flux` that the convective kernel calls (single call, no
# per-face branch). Add a scheme by writing a body and registering it below.
def _upwind_flux(w_l: 'float', w_r: 'float', u_face: 'float', v_face: 'float', w_face: 'float',
                face_normal: 'float[:]', flux_w: 'float[:]'):
  sign = u_face * face_normal[0] + v_face * face_normal[1] + w_face * face_normal[2]
  if sign >= 0:
    sol = w_l
  else:
    sol = w_r
  flux_w[0] = sign * sol


def _centered_flux(w_l: 'float', w_r: 'float', u_face: 'float', v_face: 'float', w_face: 'float',
                  face_normal: 'float[:]', flux_w: 'float[:]'):
  sign = u_face * face_normal[0] + v_face * face_normal[1] + w_face * face_normal[2]
  flux_w[0] = sign * 0.5 * (w_l + w_r)


def _rusanov_flux(w_l: 'float', w_r: 'float', u_face: 'float', v_face: 'float', w_face: 'float',
                 face_normal: 'float[:]', flux_w: 'float[:]'):
  # Rusanov / local Lax-Friedrichs (== upwind for linear scalar advection).
  sign = u_face * face_normal[0] + v_face * face_normal[1] + w_face * face_normal[2]
  flux_w[0] = 0.5 * sign * (w_l + w_r) - 0.5 * abs(sign) * (w_r - w_l)


def _lax_friedrichs_flux(w_l: 'float', w_r: 'float', u_face: 'float', v_face: 'float', w_face: 'float',
                        face_normal: 'float[:]', flux_w: 'float[:]'):
  # Lax-Friedrichs (local). For linear scalar advection this matches Rusanov/upwind.
  sign = u_face * face_normal[0] + v_face * face_normal[1] + w_face * face_normal[2]
  flux_w[0] = 0.5 * sign * (w_l + w_r) - 0.5 * abs(sign) * (w_r - w_l)


_FLUX_BODIES = {"upwind": _upwind_flux, "centered": _centered_flux,
              "rusanov": _rusanov_flux, "lax_friedrichs": _lax_friedrichs_flux}
_compute_flux = None
_current_scheme = None

def _explicitscheme_convective_2d(rez_w: 'float[:]', w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
                                 u_face: 'float[:]', v_face: 'float[:]', w_face: 'float[:]',
                                 w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', wx_halo: 'float[:]',
                                 wy_halo: 'float[:]',
                                 wz_halo: 'float[:]', psi: 'float[:]', psi_halo: 'float[:]',
                                 cell_center: 'float[:,:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                 face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_haloid: 'int[:]',
                                 face_name: 'int[:]', d_innerfaces: 'int[:]', d_halofaces: 'int[:]',
                                 d_boundaryfaces: 'int[:]',
                                 d_periodicboundaryfaces: 'int[:]', cell_shift: 'float[:,:]', order: 'int'):
  center_left = np.zeros(2)
  center_right = np.zeros(2)
  r_l = np.zeros(2)
  r_r = np.zeros(2)

  normal = np.zeros(3)
  flux_w = np.zeros(1)

  rez_w[:] = 0.

  if order == 1:
    # Order-1 fast path: the higher-order reconstruction term is (order-1)*...=0,
    # so skip ALL of it (gradients w_x/w_y, limiter psi, cell/face centers). Reads
    # only w_c, U.face and the normal per face -> matches OpenFOAM's div(phi,ne)
    # memory footprint. Scheme-agnostic (uses the bound _compute_flux); bit-identical
    # to the generic path at order 1.
    for i in d_innerfaces:
      cl = face_cellid[i][0]; cr = face_cellid[i][1]
      _compute_flux(w_c[cl], w_c[cr], u_face[i], v_face[i], w_face[i], face_normal[i], flux_w)
      rez_w[cl] -= flux_w[0]
      rez_w[cr] += flux_w[0]
    for i in d_periodicboundaryfaces:
      cl = face_cellid[i][0]; cr = face_cellid[i][1]
      _compute_flux(w_c[cl], w_c[cr], u_face[i], v_face[i], w_face[i], face_normal[i], flux_w)
      rez_w[cl] -= flux_w[0]
    for i in d_halofaces:
      cl = face_cellid[i][0]
      _compute_flux(w_c[cl], w_halo[face_haloid[i]], u_face[i], v_face[i], w_face[i], face_normal[i], flux_w)
      rez_w[cl] -= flux_w[0]
    for i in d_boundaryfaces:
      cl = face_cellid[i][0]
      _compute_flux(w_c[cl], w_ghost[i], u_face[i], v_face[i], w_face[i], face_normal[i], flux_w)
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

    _compute_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)

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

    _compute_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
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

    _compute_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
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

    _compute_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]



def _explicitscheme_convective_3d(rez_w: 'float[:]', w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
                                  u_face: 'float[:]', v_face: 'float[:]', w_face: 'float[:]',
                                  w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', wx_halo: 'float[:]',
                                  wy_halo: 'float[:]',
                                  wz_halo: 'float[:]', psi: 'float[:]', psi_halo: 'float[:]',
                                  cell_center: 'float[:,:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                  face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_haloid: 'int[:]',
                                  face_name: 'int[:]',
                                  d_innerfaces: 'int[:]', d_halofaces: 'int[:]', d_boundaryfaces: 'int[:]',
                                  d_periodicboundaryfaces: 'int[:]', cell_shift: 'float[:,:]', order: 'int'):

  center_left = np.zeros(3)
  center_right = np.zeros(3)
  r_l = np.zeros(3)
  r_r = np.zeros(3)

  normal = np.zeros(3)
  flux_w = np.zeros(1)

  rez_w[:] = 0.

  for i in d_innerfaces:
    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_c[face_cellid[i][1]]

    center_left[:] = cell_center[face_cellid[i][0]][:]
    center_right[:] = cell_center[face_cellid[i][1]][:]

    w_x_left = w_x[face_cellid[i][0]];
    w_x_right = w_x[face_cellid[i][1]]
    w_y_left = w_y[face_cellid[i][0]];
    w_y_right = w_y[face_cellid[i][1]]
    w_z_left = w_z[face_cellid[i][0]];
    w_z_right = w_z[face_cellid[i][1]]

    psi_left = psi[face_cellid[i][0]];
    psi_right = psi[face_cellid[i][1]]

    r_l[0] = face_center[i][0] - center_left[0];
    r_r[0] = face_center[i][0] - center_right[0];
    r_l[1] = face_center[i][1] - center_left[1];
    r_r[1] = face_center[i][1] - center_right[1];
    r_l[2] = face_center[i][2] - center_left[2];
    r_r[2] = face_center[i][2] - center_right[2];

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1] + w_z_left * r_l[2])
    w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1] + w_z_right * r_r[2])

    _compute_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)

    rez_w[face_cellid[i][0]] -= flux_w[0]
    rez_w[face_cellid[i][1]] += flux_w[0]

  for i in d_periodicboundaryfaces:

    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_c[face_cellid[i][1]]

    center_left[:] = cell_center[face_cellid[i][0]][:]
    center_right[:] = cell_center[face_cellid[i][1]][:]

    w_x_left = w_x[face_cellid[i][0]];
    w_x_right = w_x[face_cellid[i][1]]
    w_y_left = w_y[face_cellid[i][0]];
    w_y_right = w_y[face_cellid[i][1]]
    w_z_left = w_z[face_cellid[i][0]];
    w_z_right = w_z[face_cellid[i][1]]

    psi_left = psi[face_cellid[i][0]];
    psi_right = psi[face_cellid[i][1]]

    if face_name[i] == 11 or face_name[i] == 22:
      r_l[0] = face_center[i][0] - center_left[0];
      r_r[0] = face_center[i][0] - center_right[0] - cell_shift[face_cellid[i][1]][0]
      r_l[1] = face_center[i][1] - center_left[1];
      r_r[1] = face_center[i][1] - center_right[1]
      r_l[2] = face_center[i][2] - center_left[2];
      r_r[2] = face_center[i][2] - center_right[2]

    if face_name[i] == 33 or face_name[i] == 44:
      r_l[0] = face_center[i][0] - center_left[0];
      r_r[0] = face_center[i][0] - center_right[0]
      r_l[1] = face_center[i][1] - center_left[1];
      r_r[1] = face_center[i][1] - center_right[1] - cell_shift[face_cellid[i][1]][1]
      r_l[2] = face_center[i][2] - center_left[2];
      r_r[2] = face_center[i][2] - center_right[2]

    if face_name[i] == 55 or face_name[i] == 66:
      r_l[0] = face_center[i][0] - center_left[0];
      r_r[0] = face_center[i][0] - center_right[0]
      r_l[1] = face_center[i][1] - center_left[1];
      r_r[1] = face_center[i][1] - center_right[1]
      r_l[2] = face_center[i][2] - center_left[2];
      r_r[2] = face_center[i][2] - center_right[2] - cell_shift[face_cellid[i][1]][2]

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1] + w_z_left * r_l[2])
    w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1] + w_z_right * r_r[2])

    _compute_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]

  for i in d_halofaces:
    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_halo[face_haloid[i]]

    center_left[:] = cell_center[face_cellid[i][0]][:]
    center_right[:] = halo_centvol[face_haloid[i]][0:3]

    w_x_left = w_x[face_cellid[i][0]];
    w_x_right = wx_halo[face_haloid[i]]
    w_y_left = w_y[face_cellid[i][0]];
    w_y_right = wy_halo[face_haloid[i]]
    w_z_left = w_z[face_cellid[i][0]];
    w_z_right = wz_halo[face_haloid[i]]

    psi_left = psi[face_cellid[i][0]];
    psi_right = psi_halo[face_haloid[i]]

    r_l[0] = face_center[i][0] - center_left[0];
    r_r[0] = face_center[i][0] - center_right[0]
    r_l[1] = face_center[i][1] - center_left[1];
    r_r[1] = face_center[i][1] - center_right[1]
    r_l[2] = face_center[i][2] - center_left[2];
    r_r[2] = face_center[i][2] - center_right[2]

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1] + w_z_left * r_l[2])
    w_r = w_r + (order - 1) * psi_right * (w_x_right * r_r[0] + w_y_right * r_r[1] + w_z_right * r_r[2])

    _compute_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]

  for i in d_boundaryfaces:
    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_ghost[i]
    center_left[:] = cell_center[face_cellid[i][0]][:]

    w_x_left = w_x[face_cellid[i][0]]
    w_y_left = w_y[face_cellid[i][0]]
    w_z_left = w_z[face_cellid[i][0]]

    psi_left = psi[face_cellid[i][0]]

    r_l[0] = face_center[i][0] - center_left[0]
    r_l[1] = face_center[i][1] - center_left[1]
    r_l[2] = face_center[i][2] - center_left[2]

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1] + w_z_left * r_l[2])
    w_r = w_r

    _compute_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]


def _time_step(u: 'float[:]', v: 'float[:]', w: 'float[:]', cfl: 'float', face_normal: 'float[:,:]',
               face_measure: 'float[:]', cell_volume: 'float[:]', cell_faceid: 'int[:,:]', dim: 'int',
               Dxx: 'float', Dyy: 'float', Dzz: 'float'):
  nbelement = len(cell_faceid)
  norm = np.zeros(3)
  dt = 1e6
  for i in range(nbelement):
    lam = 0.

    for j in range(cell_faceid[i][-1]):
      norm[:] = face_normal[cell_faceid[i][j]][:]

      lam_convect = np.fabs(u[i] * norm[0] + v[i] * norm[1] + w[i] * norm[2])
      lam += lam_convect

      mes = np.sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2])
      lam_diff = Dxx * mes ** 2 + Dyy * mes ** 2 + Dzz * mes ** 2
      lam += lam_diff / cell_volume[i]

    if lam != 0:
      dt = min(dt, cfl * cell_volume[i] / lam)

  return dt


def _update_new_value(ne_c: 'float[:]', rez_ne: 'float[:]', dissip_ne: 'float[:]', src_ne: 'float[:]',
                      dtime: 'float', cell_volume: 'float[:]'):
  nbelements = len(ne_c)
  for i in range(nbelements):
    ne_c[i] += dtime * ((rez_ne[i] + dissip_ne[i]) / cell_volume[i] + src_ne[i])


############################################################################
# NOTHING is compiled at import. Call setup(dim) once (uniformly on all MPI
# ranks) before using any kernel below; the solvers do this in __init__.
#   - agnostic kernels are compiled once;
#   - dimension-specific kernels are compiled only for the dimension(s) used.
# Nested helpers are compiled (and rebound to module globals) before the kernels
# that call them, so numba can resolve njit->njit calls.
_agnostic_done = False
_dims_done = set()

def setup(dim, scheme="upwind"):
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
    _compute_flux = compile_no_cache(_FLUX_BODIES[scheme])
    _current_scheme = scheme
    _dims_done.clear()

  if dim not in _dims_done:
    global explicitscheme_convective_2d, explicitscheme_convective_3d
    if dim == 2:
      explicitscheme_convective_2d = compile_no_cache(_explicitscheme_convective_2d)
    elif dim == 3:
      explicitscheme_convective_3d = compile_no_cache(_explicitscheme_convective_3d)
    else:
      raise ValueError(f"Unsupported dimension: {dim}")
    _dims_done.add(dim)
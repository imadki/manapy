from manapy.backends.compile_fun import compile
import numpy as np

def _compute_upwind_flux(w_l: 'float', w_r: 'float', u_face: 'float', v_face: 'float', w_face: 'float',
                        normal: 'float[:]', flux_w: 'float[:]'):
  sign = u_face * normal[0] + v_face * normal[1] + w_face * normal[2]

  if sign >= 0:
    sol = w_l
  else:
    sol = w_r

  flux_w[0] = sign * sol


def _explicitscheme_convective_2d(rez_w: 'float[:]', w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
                                 u_face: 'float[:]', v_face: 'float[:]', w_face: 'float[:]',
                                 w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', wx_halo: 'float[:]',
                                 wy_halo: 'float[:]',
                                 wz_halo: 'float[:]', psi: 'float[:]', psi_halo: 'float[:]',
                                 cell_center: 'float[:,:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                 face_ghostcenter: 'float[:,:]',
                                 face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_haloid: 'int[:]',
                                 face_name: 'int[:]', d_innerfaces: 'int[:]', d_halofaces: 'int[:]',
                                 d_boundaryfaces: 'int[:]',
                                 d_periodicboundaryfaces: 'int[:]', cell_shift: 'float[:,:]', order: 'int'):

  center_left = np.zeros(2)
  center_right = np.zeros(2)
  r_l = np.zeros(2)
  r_r = np.zeros(2)

  normal = np.zeros(face_normal.shape[1])  # TODO 2 or 3
  flux_w = np.zeros(1)

  rez_w[:] = 0.

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

    _compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)

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

    _compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
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

    _compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
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

    _compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]



def _explicitscheme_convective_3d(rez_w: 'float[:]', w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
                                 u_face: 'float[:]', v_face: 'float[:]', w_face: 'float[:]',
                                 w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', wx_halo: 'float[:]',
                                 wy_halo: 'float[:]',
                                 wz_halo: 'float[:]', psi: 'float[:]', psi_halo: 'float[:]',
                                 cell_center: 'float[:,:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                 face_ghostcenter: 'float[:,:]',
                                 face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_haloid: 'int[:]', face_name: 'int[:]',
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

    _compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)

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

    _compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
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

    _compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]

  for i in d_boundaryfaces:
    w_l = w_c[face_cellid[i][0]]
    normal[:] = face_normal[i][:]

    w_r = w_ghost[i]
    center_left[:] = cell_center[face_cellid[i][0]][:]

    w_x_left = w_x[face_cellid[i][0]]
    w_y_left = w_y[face_cellid[i][0]]
    w_z_left = w_z[face_cellid[i][0]]

    psi_left = psi[face_cellid[i][0]];

    r_l[0] = face_center[i][0] - center_left[0]
    r_l[1] = face_center[i][1] - center_left[1]
    r_l[2] = face_center[i][2] - center_left[2]

    w_l = w_l + (order - 1) * psi_left * (w_x_left * r_l[0] + w_y_left * r_l[1] + w_z_left * r_l[2])
    w_r = w_r

    _compute_upwind_flux(w_l, w_r, u_face[i], v_face[i], w_face[i], normal, flux_w)
    rez_w[face_cellid[i][0]] -= flux_w[0]



def _time_step(u: 'float[:]', v: 'float[:]', w: 'float[:]', cfl: 'float', face_normal: 'float[:,:]',
              face_measure: 'float[:]', cell_volume: 'float[:]', cell_faceid: 'int[:,:]', dim: 'int'):
  nbelement = len(cell_faceid)
  norm = np.zeros(3)
  dt = 1e6
  for i in range(nbelement):
    lam = 0.

    for j in range(cell_faceid[i][-1]):
      norm[:] = face_normal[cell_faceid[i][j]][:]
      lam_convect = np.fabs(u[i] * norm[0] + v[i] * norm[1] + w[i] * norm[2])
      lam += lam_convect

    dt = min(dt, cfl * cell_volume[i] / lam)

  return dt


def _update_new_value(ne_c: 'float[:]', rez_ne: 'float[:]', dissip_ne: 'float[:]', src_ne: 'float[:]',
                     dtime: 'float', cell_volume: 'float[:]'):
  nbelements = len(ne_c)
  for i in range(nbelements):
    ne_c[i] += dtime * ((rez_ne[i] + dissip_ne[i]) / cell_volume[i] + src_ne[i])


############################################################################
# Private
_compute_upwind_flux = compile(_compute_upwind_flux)

# Public
explicitscheme_convective_2d = compile(_explicitscheme_convective_2d)
explicitscheme_convective_3d = compile(_explicitscheme_convective_3d)
time_step = compile(_time_step)
update_new_value = compile(_update_new_value)
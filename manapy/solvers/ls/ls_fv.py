"""Cell-centred finite-volume (Gauss-linear corrected) Laplacian kernels.

Auto-split from the former monolithic ls_compute.py: each scheme now lives in
its own module with its own setup(dim) so that only the kernels of the scheme
actually in use get compiled.
"""
from manapy.backends.compile_fun import compile
import numpy as np

def _compute_fv_matrix_size(matrixinnerfaces: 'int[:]', d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):
  return 4 * len(matrixinnerfaces) + 2 * len(d_halofaces) + len(dirichletfaces)

def _get_triplet_fv(face_cellid: 'int[:,:]', face_fv_coeff: 'float[:]',
                          halo_halosext: 'int[:,:]', cell_volume: 'float[:]',
                          cell_loctoglob: 'int[:]', face_haloid: 'int[:]', a_loc: 'float[:]',
                          irn_loc: 'int[:]', jcn_loc: 'int[:]', matrixinnerfaces: 'int[:]',
                          d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):
  cmpt = 0

  for i in matrixinnerfaces:
    c_left = face_cellid[i, 0]
    c_right = face_cellid[i, 1]
    c_leftglob = cell_loctoglob[c_left]
    c_rightglob = cell_loctoglob[c_right]
    coeff = face_fv_coeff[i]

    value = coeff / cell_volume[c_left]
    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    a_loc[cmpt] = -value
    cmpt += 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    a_loc[cmpt] = value
    cmpt += 1

    value = coeff / cell_volume[c_right]
    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_leftglob
    a_loc[cmpt] = value
    cmpt += 1

    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_rightglob
    a_loc[cmpt] = -value
    cmpt += 1

  for i in d_halofaces:
    c_left = face_cellid[i, 0]
    c_leftglob = cell_loctoglob[c_left]
    c_rightglob = halo_halosext[face_haloid[i], 0]
    coeff = face_fv_coeff[i]
    value = coeff / cell_volume[c_left]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    a_loc[cmpt] = -value
    cmpt += 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    a_loc[cmpt] = value
    cmpt += 1

  for i in dirichletfaces:
    c_left = face_cellid[i, 0]
    c_leftglob = cell_loctoglob[c_left]
    coeff = face_fv_coeff[i]
    value = coeff / cell_volume[c_left]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    a_loc[cmpt] = -value
    cmpt += 1

def _get_rhs_fv_glob(face_cellid: 'int[:,:]', face_fv_coeff: 'float[:]',
                           cell_volume: 'float[:]', cell_loctoglob: 'int[:]',
                           Pbordface: 'float[:]', rhs: 'float[:]', dirichletfaces: 'int[:]'):
  rhs[:] = 0.0
  for i in dirichletfaces:
    c_left = face_cellid[i, 0]
    c_leftglob = cell_loctoglob[c_left]
    coeff = face_fv_coeff[i]
    rhs[c_leftglob] += -coeff * Pbordface[i] / cell_volume[c_left]

def _get_rhs_fv_loc(face_cellid: 'int[:,:]', face_fv_coeff: 'float[:]',
                          cell_volume: 'float[:]', cell_loctoglob: 'int[:]',
                          Pbordface: 'float[:]', rhs: 'float[:]', dirichletfaces: 'int[:]'):
  rhs[:] = 0.0
  for i in dirichletfaces:
    c_left = face_cellid[i, 0]
    coeff = face_fv_coeff[i]
    rhs[c_left] += -coeff * Pbordface[i] / cell_volume[c_left]

def _get_rhs_fv_correction_glob(face_cellid: 'int[:,:]', face_haloid: 'int[:]',
                                      cell_volume: 'float[:]', cell_loctoglob: 'int[:]',
                                      face_fv_corrx: 'float[:]', face_fv_corry: 'float[:]',
                                      face_fv_corrz: 'float[:]', face_fv_weight_left: 'float[:]',
                                      gradcellx: 'float[:]',
                                      gradcelly: 'float[:]', gradcellz: 'float[:]',
                                      gradhalocellx: 'float[:]', gradhalocelly: 'float[:]',
                                      gradhalocellz: 'float[:]', rhs: 'float[:]', matrixinnerfaces: 'int[:]',
                                      d_halofaces: 'int[:]', dirichletfaces: 'int[:]',
                                      d_periodicboundaryfaces: 'int[:]'):
  rhs[:] = 0.0

  for i in matrixinnerfaces:
    c_left = face_cellid[i, 0]
    c_right = face_cellid[i, 1]
    wl = face_fv_weight_left[i]
    wr = 1.0 - wl
    gx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
    gy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
    gz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
    corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
    rhs[cell_loctoglob[c_left]] += -corr / cell_volume[c_left]
    rhs[cell_loctoglob[c_right]] += corr / cell_volume[c_right]

  for i in d_periodicboundaryfaces:
    c_left = face_cellid[i, 0]
    c_right = face_cellid[i, 1]
    wl = face_fv_weight_left[i]
    wr = 1.0 - wl
    gx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
    gy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
    gz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
    corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
    rhs[cell_loctoglob[c_left]] += -corr / cell_volume[c_left]
    rhs[cell_loctoglob[c_right]] += corr / cell_volume[c_right]

  for i in d_halofaces:
    c_left = face_cellid[i, 0]
    c_right = face_haloid[i]
    wl = face_fv_weight_left[i]
    wr = 1.0 - wl
    gx = wl * gradcellx[c_left] + wr * gradhalocellx[c_right]
    gy = wl * gradcelly[c_left] + wr * gradhalocelly[c_right]
    gz = wl * gradcellz[c_left] + wr * gradhalocellz[c_right]
    corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
    rhs[cell_loctoglob[c_left]] += -corr / cell_volume[c_left]

  for i in dirichletfaces:
    c_left = face_cellid[i, 0]
    corr = (face_fv_corrx[i] * gradcellx[c_left]
            + face_fv_corry[i] * gradcelly[c_left]
            + face_fv_corrz[i] * gradcellz[c_left])
    rhs[cell_loctoglob[c_left]] += -corr / cell_volume[c_left]

def _get_rhs_fv_correction_loc(face_cellid: 'int[:,:]', face_haloid: 'int[:]',
                                     cell_volume: 'float[:]', cell_loctoglob: 'int[:]',
                                     face_fv_corrx: 'float[:]', face_fv_corry: 'float[:]',
                                     face_fv_corrz: 'float[:]', face_fv_weight_left: 'float[:]',
                                     gradcellx: 'float[:]',
                                     gradcelly: 'float[:]', gradcellz: 'float[:]',
                                     gradhalocellx: 'float[:]', gradhalocelly: 'float[:]',
                                     gradhalocellz: 'float[:]', rhs: 'float[:]', matrixinnerfaces: 'int[:]',
                                     d_halofaces: 'int[:]', dirichletfaces: 'int[:]',
                                     d_periodicboundaryfaces: 'int[:]'):
  rhs[:] = 0.0

  for i in matrixinnerfaces:
    c_left = face_cellid[i, 0]
    c_right = face_cellid[i, 1]
    wl = face_fv_weight_left[i]
    wr = 1.0 - wl
    gx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
    gy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
    gz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
    corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
    rhs[c_left] += -corr / cell_volume[c_left]
    rhs[c_right] += corr / cell_volume[c_right]

  for i in d_periodicboundaryfaces:
    c_left = face_cellid[i, 0]
    c_right = face_cellid[i, 1]
    wl = face_fv_weight_left[i]
    wr = 1.0 - wl
    gx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
    gy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
    gz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
    corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
    rhs[c_left] += -corr / cell_volume[c_left]
    rhs[c_right] += corr / cell_volume[c_right]

  for i in d_halofaces:
    c_left = face_cellid[i, 0]
    c_right = face_haloid[i]
    wl = face_fv_weight_left[i]
    wr = 1.0 - wl
    gx = wl * gradcellx[c_left] + wr * gradhalocellx[c_right]
    gy = wl * gradcelly[c_left] + wr * gradhalocelly[c_right]
    gz = wl * gradcellz[c_left] + wr * gradhalocellz[c_right]
    corr = face_fv_corrx[i] * gx + face_fv_corry[i] * gy + face_fv_corrz[i] * gz
    rhs[c_left] += -corr / cell_volume[c_left]

  for i in dirichletfaces:
    c_left = face_cellid[i, 0]
    corr = (face_fv_corrx[i] * gradcellx[c_left]
            + face_fv_corry[i] * gradcelly[c_left]
            + face_fv_corrz[i] * gradcellz[c_left])
    rhs[c_left] += -corr / cell_volume[c_left]

def _set_fv_gradient(P_left: 'float', P_right: 'float', gfx: 'float', gfy: 'float', gfz: 'float',
                     nx: 'float', ny: 'float', nz: 'float', dx: 'float', dy: 'float', dz: 'float',
                     i: 'int', Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]'):
  # Corrected face gradient: take the interpolated cell gradient (gf, a FULL
  # vector) and replace its normal component by the compact two-point normal
  # derivative (P_right - P_left)/d_ortho. This keeps the accurate normal part
  # while recovering the tangential part that a normal-only two-point gradient
  # drops (which otherwise halves the reconstructed cell field on stretched/
  # non-axis-aligned faces). Stores -grad(P): the streamer's _compute_el_field
  # sets E_face = gradface directly (E = -grad P), same convention as diamond.
  mag = np.sqrt(nx * nx + ny * ny + nz * nz)
  denom = nx * dx + ny * dy + nz * dz
  if denom < 0.0:
    denom = -denom
  if denom == 0.0 or mag == 0.0:
    raise RuntimeError("zero projected face distance in FV-like gradient")
  sn = (P_right - P_left) * mag / denom             # grad(P).n_hat (two-point)
  gdotn = (gfx * nx + gfy * ny + gfz * nz) / mag     # interpolated grad . n_hat
  corr = sn - gdotn
  Px_face[i] = -(gfx + corr * nx / mag)
  Py_face[i] = -(gfy + corr * ny / mag)
  Pz_face[i] = -(gfz + corr * nz / mag)

def _compute_P_gradient_fv(P_c: 'float[:]', P_halo: 'float[:]', face_cellid: 'int[:,:]',
                                 face_name: 'int[:]', face_normal: 'float[:,:]', face_center: 'float[:,:]',
                                 face_haloid: 'int[:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                 cell_shift: 'float[:,:]', Pbordface: 'float[:]',
                                 gradcellx: 'float[:]', gradcelly: 'float[:]', gradcellz: 'float[:]',
                                 gradhalocellx: 'float[:]', gradhalocelly: 'float[:]', gradhalocellz: 'float[:]',
                                 weight_left: 'float[:]',
                                 Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]', d_innerfaces: 'int[:]',
                                 d_halofaces: 'int[:]', neumannfaces: 'int[:]', dirichletfaces: 'int[:]',
                                 d_periodicboundaryfaces: 'int[:]'):
  for i in d_innerfaces:
    c_left = face_cellid[i, 0]
    c_right = face_cellid[i, 1]
    wl = weight_left[i]
    wr = 1.0 - wl
    gfx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
    gfy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
    gfz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
    dx = cell_center[c_right, 0] - cell_center[c_left, 0]
    dy = cell_center[c_right, 1] - cell_center[c_left, 1]
    dz = cell_center[c_right, 2] - cell_center[c_left, 2]
    _set_fv_gradient(P_c[c_left], P_c[c_right], gfx, gfy, gfz, face_normal[i, 0], face_normal[i, 1],
                           face_normal[i, 2], dx, dy, dz, i, Px_face, Py_face, Pz_face)

  for i in d_periodicboundaryfaces:
    c_left = face_cellid[i, 0]
    c_right = face_cellid[i, 1]
    wl = weight_left[i]
    wr = 1.0 - wl
    gfx = wl * gradcellx[c_left] + wr * gradcellx[c_right]
    gfy = wl * gradcelly[c_left] + wr * gradcelly[c_right]
    gfz = wl * gradcellz[c_left] + wr * gradcellz[c_right]
    dx = cell_center[c_right, 0] - cell_center[c_left, 0]
    dy = cell_center[c_right, 1] - cell_center[c_left, 1]
    dz = cell_center[c_right, 2] - cell_center[c_left, 2]
    if face_name[i] == 11 or face_name[i] == 22:
      dx += cell_shift[c_right, 0]
    elif face_name[i] == 33 or face_name[i] == 44:
      dy += cell_shift[c_right, 1]
    elif face_name[i] == 55 or face_name[i] == 66:
      dz += cell_shift[c_right, 2]
    _set_fv_gradient(P_c[c_left], P_c[c_right], gfx, gfy, gfz, face_normal[i, 0], face_normal[i, 1],
                           face_normal[i, 2], dx, dy, dz, i, Px_face, Py_face, Pz_face)

  for i in d_halofaces:
    c_left = face_cellid[i, 0]
    c_right = face_haloid[i]
    wl = weight_left[i]
    wr = 1.0 - wl
    gfx = wl * gradcellx[c_left] + wr * gradhalocellx[c_right]
    gfy = wl * gradcelly[c_left] + wr * gradhalocelly[c_right]
    gfz = wl * gradcellz[c_left] + wr * gradhalocellz[c_right]
    dx = halo_centvol[c_right, 0] - cell_center[c_left, 0]
    dy = halo_centvol[c_right, 1] - cell_center[c_left, 1]
    dz = halo_centvol[c_right, 2] - cell_center[c_left, 2]
    _set_fv_gradient(P_c[c_left], P_halo[c_right], gfx, gfy, gfz, face_normal[i, 0], face_normal[i, 1],
                           face_normal[i, 2], dx, dy, dz, i, Px_face, Py_face, Pz_face)

  for i in neumannfaces:
    # Homogeneous Neumann: grad(P).n = 0, so the face gradient is the tangential
    # part of the (owner) cell gradient. Stores -grad(P).
    c_left = face_cellid[i, 0]
    nx = face_normal[i, 0]; ny = face_normal[i, 1]; nz = face_normal[i, 2]
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    gfx = gradcellx[c_left]; gfy = gradcelly[c_left]; gfz = gradcellz[c_left]
    gdotn = (gfx * nx + gfy * ny + gfz * nz) / mag
    Px_face[i] = -(gfx - gdotn * nx / mag)
    Py_face[i] = -(gfy - gdotn * ny / mag)
    Pz_face[i] = -(gfz - gdotn * nz / mag)

  for i in dirichletfaces:
    c_left = face_cellid[i, 0]
    gfx = gradcellx[c_left]; gfy = gradcelly[c_left]; gfz = gradcellz[c_left]
    dx = face_center[i, 0] - cell_center[c_left, 0]
    dy = face_center[i, 1] - cell_center[c_left, 1]
    dz = face_center[i, 2] - cell_center[c_left, 2]
    _set_fv_gradient(P_c[c_left], Pbordface[i], gfx, gfy, gfz, face_normal[i, 0], face_normal[i, 1],
                           face_normal[i, 2], dx, dy, dz, i, Px_face, Py_face, Pz_face)


_done = False

def setup(dim):
    """Compile the cell-centred FV (Gauss-linear, optional non-orthogonal
    correction) kernels. They are dimension-generic, so `dim` is accepted for a
    uniform interface but the kernels are compiled once. Idempotent."""
    global _done
    if _done:
        return
    global compute_fv_matrix_size, _set_fv_gradient
    global get_triplet_fv, get_rhs_fv_glob, get_rhs_fv_loc
    global get_rhs_fv_correction_glob, get_rhs_fv_correction_loc
    global compute_P_gradient_fv
    compute_fv_matrix_size = compile(_compute_fv_matrix_size)
    _set_fv_gradient = compile(_set_fv_gradient)  # nested helper first
    get_triplet_fv = compile(_get_triplet_fv)
    get_rhs_fv_glob = compile(_get_rhs_fv_glob)
    get_rhs_fv_loc = compile(_get_rhs_fv_loc)
    get_rhs_fv_correction_glob = compile(_get_rhs_fv_correction_glob)
    get_rhs_fv_correction_loc = compile(_get_rhs_fv_correction_loc)
    compute_P_gradient_fv = compile(_compute_P_gradient_fv)
    _done = True

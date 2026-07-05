# -*- coding: utf-8 -*-
"""
Corps de kernels Variable 2D, forme grid-stride unifiee : `body(start, stride, *args)`.

UNE seule definition par kernel, compilable pour le CPU (njit, appele avec
start=0/stride=1) ET le GPU (device function + kernel mince fournissant
cuda.grid(1)/cuda.gridsize(1)), via backend.make_gridstride_kernel(body, size_arg).

Contrainte : n'utiliser que des constructions valides dans les deux backends
(scalaires, indexation, range) -> pas de np.zeros/cuda.local.array, pas d'atomic.
Reserve aux kernels a SORTIE INDEPENDANTE (un indice de boucle == un element de
sortie). Les annotations de type sont identiques aux kernels CPU d'origine.

Ordre des `*args` identique aux appels de Variable.py. size_arg (pour le GPU) :
  facetocell        -> 1            (u_c)
  celltoface        -> (6, 7, 8)    (inner / boundary / halo faces)
  cell_gradient_2d  -> 0            (w_c)
  barthlimiter_2d   -> 0            (w_c)
  centertovertex_2d -> 13           (nodes)
  face_gradient_2d  -> (16,17,18,19,20)
"""


def facetocell(start: 'int', stride: 'int', u_face: 'float[:]', u_c: 'float[:]',
               cell_faceid: 'int[:,:]', dim: 'int'):
  for i in range(start, u_c.shape[0], stride):
    acc = 0.0
    for j in range(cell_faceid[i][-1]):
      acc += u_face[cell_faceid[i][j]]
    u_c[i] = acc / cell_faceid[i][-1]


def celltoface(start: 'int', stride: 'int', u_cell: 'float[:]', u_face: 'float[:]',
               u_ghost: 'float[:]', u_halo: 'float[:]', face_cellid: 'int[:,:]',
               face_halofid: 'int[:]', d_innerfaces: 'int[:]', d_boundaryfaces: 'int[:]',
               d_halofaces: 'int[:]'):
  for idx in range(start, d_innerfaces.shape[0], stride):
    i = d_innerfaces[idx]
    u_face[i] = 0.5 * (u_cell[face_cellid[i][0]] + u_cell[face_cellid[i][1]])
  for idx in range(start, d_halofaces.shape[0], stride):
    i = d_halofaces[idx]
    u_face[i] = 0.5 * (u_cell[face_cellid[i][0]] + u_halo[face_halofid[i]])
  for idx in range(start, d_boundaryfaces.shape[0], stride):
    i = d_boundaryfaces[idx]
    u_face[i] = 0.5 * (u_cell[face_cellid[i][0]] + u_ghost[i])


def cell_gradient_2d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                     w_halo: 'float[:]', w_haloghost: 'float[:]', cell_center: 'float[:,:]',
                     cell_cellnid: 'int[:,:]', ghost_info_flt: 'float[:, :]',
                     ghost_ext_info_flt: 'float[:, :]', cell_ghostnid: 'int[:,:]',
                     cell_haloghostnid: 'int[:,:]', cell_halonid: 'int[:,:]', cells: 'int[:,:]',
                     cell_periodicfid: 'int[:,:]', node_periodicid: 'int[:,:]', node_oldname: 'int[:]',
                     halo_centvol: 'float[:,:]', cell_shift: 'float[:,:]',
                     w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', ghost_faceid: 'int[:]'):
  for i in range(start, w_c.shape[0], stride):
    i_xx = 0.; i_yy = 0.; i_xy = 0.
    j_xw = 0.; j_yw = 0.

    for j in range(cell_cellnid[i][-1]):
      cell = cell_cellnid[i][j]
      j_x = cell_center[cell][0] - cell_center[i][0]
      j_y = cell_center[cell][1] - cell_center[i][1]
      i_xx += j_x * j_x; i_yy += j_y * j_y; i_xy += j_x * j_y
      j_xw += j_x * (w_c[cell] - w_c[i])
      j_yw += j_y * (w_c[cell] - w_c[i])

    for j in range(cell_ghostnid[i][-1]):
      ghost_id = cell_ghostnid[i][j]
      j_x = ghost_info_flt[ghost_id][0] - cell_center[i][0]
      j_y = ghost_info_flt[ghost_id][1] - cell_center[i][1]
      i_xx += j_x * j_x; i_yy += j_y * j_y; i_xy += j_x * j_y
      j_xw += j_x * (w_ghost[ghost_faceid[ghost_id]] - w_c[i])
      j_yw += j_y * (w_ghost[ghost_faceid[ghost_id]] - w_c[i])

    for k in range(cells[i][-1]):
      nod = cells[i][k]
      if node_oldname[nod] == 11 or node_oldname[nod] == 22:
        for j in range(node_periodicid[nod][-1]):
          cell = node_periodicid[nod][j]
          j_x = cell_center[cell][0] + cell_shift[cell][0] - cell_center[i][0]
          j_y = cell_center[cell][1] - cell_center[i][1]
          i_xx += j_x * j_x; i_yy += j_y * j_y; i_xy += j_x * j_y
          j_xw += j_x * (w_c[cell] - w_c[i])
          j_yw += j_y * (w_c[cell] - w_c[i])
      if node_oldname[nod] == 33 or node_oldname[nod] == 44:
        for j in range(node_periodicid[nod][-1]):
          cell = node_periodicid[nod][j]
          j_x = cell_center[cell][0] - cell_center[i][0]
          j_y = cell_center[cell][1] + cell_shift[cell][1] - cell_center[i][1]
          i_xx += j_x * j_x; i_yy += j_y * j_y; i_xy += j_x * j_y
          j_xw += j_x * (w_c[cell] - w_c[i])
          j_yw += j_y * (w_c[cell] - w_c[i])

    for j in range(cell_halonid[i, -1]):
      cell = cell_halonid[i][j]
      j_x = halo_centvol[cell][0] - cell_center[i][0]
      j_y = halo_centvol[cell][1] - cell_center[i][1]
      i_xx += j_x * j_x; i_yy += j_y * j_y; i_xy += j_x * j_y
      j_xw += j_x * (w_halo[cell] - w_c[i])
      j_yw += j_y * (w_halo[cell] - w_c[i])

    for j in range(cell_haloghostnid[i][-1]):
      ghost_id = cell_haloghostnid[i][j]
      j_x = ghost_ext_info_flt[ghost_id][0] - cell_center[i][0]
      j_y = ghost_ext_info_flt[ghost_id][1] - cell_center[i][1]
      i_xx += j_x * j_x; i_yy += j_y * j_y; i_xy += j_x * j_y
      j_xw += j_x * (w_haloghost[ghost_id] - w_c[i])
      j_yw += j_y * (w_haloghost[ghost_id] - w_c[i])

    dia = i_xx * i_yy - i_xy * i_xy
    w_x[i] = (i_yy * j_xw - i_xy * j_yw) / dia
    w_y[i] = (i_xx * j_yw - i_xy * j_xw) / dia
    w_z[i] = 0.


def barthlimiter_2d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                    w_halo: 'float[:]', w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]',
                    psi: 'float[:]', face_cellid: 'int[:,:]', cell_faceid: 'int[:,:]',
                    face_name: 'int[:]', face_haloid: 'int[:]', cell_center: 'float[:,:]',
                    face_center: 'float[:,:]'):
  val = 1.0
  for i in range(start, w_c.shape[0], stride):
    psi[i] = val
    w_max = w_c[i]
    w_min = w_c[i]
    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      if face_name[face] == 0 or face_name[face] > 10:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_c[face_cellid[face][1]])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_c[face_cellid[face][1]])
      elif face_name[face] == 1 or face_name[face] == 2 or face_name[face] == 3 or face_name[face] == 4:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_ghost[face])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_ghost[face])
      else:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_halo[face_haloid[face]])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_halo[face_haloid[face]])
    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      r_xyz1 = face_center[face][0] - cell_center[i][0]
      r_xyz2 = face_center[face][1] - cell_center[i][1]
      delta2 = w_x[i] * r_xyz1 + w_y[i] * r_xyz2
      psi_ij = 1.0
      if abs(delta2) >= 1e-8:
        if delta2 > 0.:
          psi_ij = min(val, (w_max - w_c[i]) / delta2)
        if delta2 < 0.:
          psi_ij = min(val, (w_min - w_c[i]) / delta2)
      psi[i] = min(psi[i], psi_ij)


def vanalbadalimiter_2d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                        w_halo: 'float[:]', w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]',
                        psi: 'float[:]', face_cellid: 'int[:,:]', cell_faceid: 'int[:,:]',
                        face_name: 'int[:]', face_haloid: 'int[:]', cell_center: 'float[:,:]',
                        face_center: 'float[:,:]'):
  # Smooth (van Albada / Venkatakrishnan) slope limiter: same neighbourhood-min/max
  # structure as Barth-Jespersen, but the per-face factor is the smooth function
  # phi(y) = (y^2 + 2y)/(y^2 + y + 2) of the Barth argument y = (w_max-w_c)/delta
  # instead of min(1, y). phi(y) ~ y for small y and -> 1 for large y, so it stops
  # "clipping" smooth extrema (Barth's min(1,y) kills the 2nd order there) -> much
  # smoother convergence, while staying bounded (<= 1). Drop-in for barthlimiter_2d
  # (identical signature); selected by Variable(limiter='vanalbada').
  val = 1.0
  for i in range(start, w_c.shape[0], stride):
    psi[i] = val
    w_max = w_c[i]
    w_min = w_c[i]
    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      if face_name[face] == 0 or face_name[face] > 10:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_c[face_cellid[face][1]])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_c[face_cellid[face][1]])
      elif face_name[face] == 1 or face_name[face] == 2 or face_name[face] == 3 or face_name[face] == 4:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_ghost[face])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_ghost[face])
      else:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_halo[face_haloid[face]])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_halo[face_haloid[face]])
    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      r_xyz1 = face_center[face][0] - cell_center[i][0]
      r_xyz2 = face_center[face][1] - cell_center[i][1]
      delta2 = w_x[i] * r_xyz1 + w_y[i] * r_xyz2
      psi_ij = 1.0
      if abs(delta2) >= 1e-8:
        if delta2 > 0.:
          y = (w_max - w_c[i]) / delta2
        else:
          y = (w_min - w_c[i]) / delta2
        psi_ij = (y * y + 2.0 * y) / (y * y + y + 2.0)
      psi[i] = min(psi[i], psi_ij)


def centertovertex_2d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                      w_halo: 'float[:]', w_haloghost: 'float[:]', cell_center: 'float[:,:]',
                      halo_centvol: 'float[:,:]', node_cellid: 'int[:,:]', ghost_info_flt: 'float[:, :]',
                      ghost_ext_info_flt: 'float[:, :]', node_ghostid: 'int[:,:]',
                      node_haloghostid: 'int[:,:]', node_periodicid: 'int[:,:]', node_halonid: 'int[:,:]',
                      nodes: 'float[:,:]', node_oldname: 'int[:]', node_R_x: 'float[:]',
                      node_R_y: 'float[:]', node_R_z: 'float[:]', node_lambda_x: 'float[:]',
                      node_lambda_y: 'float[:]', node_lambda_z: 'float[:]', node_number: 'int[:]',
                      cell_shift: 'float[:,:]', w_n: 'float[:]', ghost_faceid: 'int[:]'):
  for i in range(start, nodes.shape[0], stride):
    acc = 0.0
    denom = node_number[i] + node_lambda_x[i] * node_R_x[i] + node_lambda_y[i] * node_R_y[i]
    for j in range(node_cellid[i][-1]):
      cell = node_cellid[i][j]
      xdiff = cell_center[cell][0] - nodes[i][0]
      ydiff = cell_center[cell][1] - nodes[i][1]
      alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / denom
      acc += alpha * w_c[cell]
    for j in range(node_ghostid[i][-1]):
      ghost_id = node_ghostid[i][j]
      xdiff = ghost_info_flt[ghost_id][0] - nodes[i][0]
      ydiff = ghost_info_flt[ghost_id][1] - nodes[i][1]
      alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / denom
      acc += alpha * w_ghost[ghost_faceid[ghost_id]]
    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i][j]
      xdiff = ghost_ext_info_flt[ghost_id][0] - nodes[i][0]
      ydiff = ghost_ext_info_flt[ghost_id][1] - nodes[i][1]
      alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / denom
      acc += alpha * w_haloghost[ghost_id]
    for j in range(node_halonid[i, -1]):
      cell = node_halonid[i][j]
      xdiff = halo_centvol[cell][0] - nodes[i][0]
      ydiff = halo_centvol[cell][1] - nodes[i][1]
      alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / denom
      acc += alpha * w_halo[cell]
    if node_oldname[i] == 11 or node_oldname[i] == 22:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        xdiff = cell_center[cell][0] + cell_shift[cell][0] - nodes[i][0]
        ydiff = cell_center[cell][1] - nodes[i][1]
        alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / denom
        acc += alpha * w_c[cell]
    elif node_oldname[i] == 33 or node_oldname[i] == 44:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        xdiff = cell_center[cell][0] - nodes[i][0]
        ydiff = cell_center[cell][1] + cell_shift[cell][1] - nodes[i][1]
        alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / denom
        acc += alpha * w_c[cell]
    w_n[i] = acc


def face_gradient_2d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                     w_halo: 'float[:]', w_node: 'float[:]', face_cellid: 'int[:,:]', faces: 'int[:,:]',
                     face_halofid: 'int[:]', face_airDiamond: 'float[:]', face_normal: 'float[:,:]',
                     face_f1: 'float[:,:]', face_f2: 'float[:,:]', face_f3: 'float[:,:]', face_f4: 'float[:,:]',
                     wx_face: 'float[:]', wy_face: 'float[:]', wz_face: 'float[:]', d_innerfaces: 'int[:]',
                     d_halofaces: 'int[:]', dirichletfaces: 'int[:]', neumann: 'int[:]',
                     d_periodicfaces: 'int[:]'):
  for idx in range(start, d_innerfaces.shape[0], stride):
    i = d_innerfaces[idx]
    vi1 = w_node[faces[i][0]]; vi2 = w_node[faces[i][1]]
    vv1 = w_c[face_cellid[i][0]]; vv2 = w_c[face_cellid[i][1]]
    inv = 1. / (2 * face_airDiamond[i])
    wx_face[i] = inv * ((vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    wy_face[i] = -inv * ((vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])
  for idx in range(start, d_periodicfaces.shape[0], stride):
    i = d_periodicfaces[idx]
    vi1 = w_node[faces[i][0]]; vi2 = w_node[faces[i][1]]
    vv1 = w_c[face_cellid[i][0]]; vv2 = w_c[face_cellid[i][1]]
    inv = 1. / (2 * face_airDiamond[i])
    wx_face[i] = inv * ((vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    wy_face[i] = -inv * ((vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])
  for idx in range(start, d_halofaces.shape[0], stride):
    i = d_halofaces[idx]
    vi1 = w_node[faces[i][0]]; vi2 = w_node[faces[i][1]]
    vv1 = w_c[face_cellid[i][0]]; vv2 = w_halo[face_halofid[i]]
    inv = 1. / (2 * face_airDiamond[i])
    wx_face[i] = inv * ((vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    wy_face[i] = -inv * ((vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])
  for idx in range(start, dirichletfaces.shape[0], stride):
    i = dirichletfaces[idx]
    vi1 = w_node[faces[i][0]]; vi2 = w_node[faces[i][1]]
    vv1 = w_c[face_cellid[i][0]]; vv2 = w_ghost[i]
    inv = 1. / (2 * face_airDiamond[i])
    wx_face[i] = inv * ((vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    wy_face[i] = -inv * ((vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])
  for idx in range(start, neumann.shape[0], stride):
    i = neumann[idx]
    vi1 = w_node[faces[i][0]]; vi2 = w_node[faces[i][1]]
    vv1 = w_c[face_cellid[i][0]]; vv2 = w_ghost[i]
    inv = 1. / (2 * face_airDiamond[i])
    wx_face[i] = inv * ((vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    wy_face[i] = -inv * ((vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])

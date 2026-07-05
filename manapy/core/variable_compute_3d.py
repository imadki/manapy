# -*- coding: utf-8 -*-
"""
Kernels Variable 3D en forme grid-stride : body(start, stride, *args).
Meme source pour CPU et GPU quand le kernel s'y prete.
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


def centertovertex_3d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
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
    denom = (node_number[i] + node_lambda_x[i] * node_R_x[i] +
             node_lambda_y[i] * node_R_y[i] + node_lambda_z[i] * node_R_z[i])

    for j in range(node_cellid[i][-1]):
      cell = node_cellid[i][j]
      xdiff = cell_center[cell][0] - nodes[i][0]
      ydiff = cell_center[cell][1] - nodes[i][1]
      zdiff = cell_center[cell][2] - nodes[i][2]
      alpha = (1.0 + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / denom
      acc += alpha * w_c[cell]

    for j in range(node_ghostid[i][-1]):
      ghost_id = node_ghostid[i][j]
      xdiff = ghost_info_flt[ghost_id][0] - nodes[i][0]
      ydiff = ghost_info_flt[ghost_id][1] - nodes[i][1]
      zdiff = ghost_info_flt[ghost_id][2] - nodes[i][2]
      alpha = (1.0 + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / denom
      acc += alpha * w_ghost[ghost_faceid[ghost_id]]

    for j in range(node_haloghostid[i, -1]):
      ghost_id = node_haloghostid[i][j]
      xdiff = ghost_ext_info_flt[ghost_id][0] - nodes[i][0]
      ydiff = ghost_ext_info_flt[ghost_id][1] - nodes[i][1]
      zdiff = ghost_ext_info_flt[ghost_id][2] - nodes[i][2]
      alpha = (1.0 + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / denom
      acc += alpha * w_haloghost[ghost_id]

    for j in range(node_halonid[i, -1]):
      cell = node_halonid[i][j]
      xdiff = halo_centvol[cell][0] - nodes[i][0]
      ydiff = halo_centvol[cell][1] - nodes[i][1]
      zdiff = halo_centvol[cell][2] - nodes[i][2]
      alpha = (1.0 + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / denom
      acc += alpha * w_halo[cell]

    if node_oldname[i] == 11 or node_oldname[i] == 22:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        xdiff = cell_center[cell][0] + cell_shift[cell][0] - nodes[i][0]
        ydiff = cell_center[cell][1] - nodes[i][1]
        zdiff = cell_center[cell][2] - nodes[i][2]
        alpha = (1.0 + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / denom
        acc += alpha * w_c[cell]
    elif node_oldname[i] == 33 or node_oldname[i] == 44:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        xdiff = cell_center[cell][0] - nodes[i][0]
        ydiff = cell_center[cell][1] + cell_shift[cell][1] - nodes[i][1]
        zdiff = cell_center[cell][2] - nodes[i][2]
        alpha = (1.0 + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / denom
        acc += alpha * w_c[cell]
    elif node_oldname[i] == 55 or node_oldname[i] == 66:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        xdiff = cell_center[cell][0] - nodes[i][0]
        ydiff = cell_center[cell][1] - nodes[i][1]
        zdiff = cell_center[cell][2] + cell_shift[cell][2] - nodes[i][2]
        alpha = (1.0 + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / denom
        acc += alpha * w_c[cell]

    w_n[i] = acc


def face_gradient_3d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                     w_halo: 'float[:]', w_node: 'float[:]', face_cellid: 'int[:,:]', faces: 'int[:,:]',
                     face_haloid: 'int[:]', face_air_diamond: 'float[:]', face_normal: 'float[:,:]',
                     face_f1: 'float[:,:]', face_f2: 'float[:,:]', face_f3: 'float[:,:]', face_f4: 'float[:,:]',
                     wx_face: 'float[:]', wy_face: 'float[:]', wz_face: 'float[:]', d_innerfaces: 'int[:]',
                     d_halofaces: 'int[:]', dirichletfaces: 'int[:]', neumann: 'int[:]',
                     d_periodicboundaryfaces: 'int[:]'):
  for idx in range(start, d_innerfaces.shape[0], stride):
    i = d_innerfaces[idx]
    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]
    i_1 = faces[i][0]; i_2 = faces[i][1]; i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]
    V_A = w_node[i_1]; V_B = w_node[i_2]; V_C = w_node[i_3]; V_D = w_node[i_4]
    V_L = w_c[c_left]; V_R = w_c[c_right]
    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for idx in range(start, d_periodicboundaryfaces.shape[0], stride):
    i = d_periodicboundaryfaces[idx]
    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]
    i_1 = faces[i][0]; i_2 = faces[i][1]; i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]
    V_A = w_node[i_1]; V_B = w_node[i_2]; V_C = w_node[i_3]; V_D = w_node[i_4]
    V_L = w_c[c_left]; V_R = w_c[c_right]
    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for idx in range(start, d_halofaces.shape[0], stride):
    i = d_halofaces[idx]
    c_left = face_cellid[i][0]
    c_right = face_haloid[i]
    i_1 = faces[i][0]; i_2 = faces[i][1]; i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]
    V_A = w_node[i_1]; V_B = w_node[i_2]; V_C = w_node[i_3]; V_D = w_node[i_4]
    V_L = w_c[c_left]; V_R = w_halo[c_right]
    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for idx in range(start, dirichletfaces.shape[0], stride):
    i = dirichletfaces[idx]
    c_left = face_cellid[i][0]
    c_right = i
    i_1 = faces[i][0]; i_2 = faces[i][1]; i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]
    V_A = w_node[i_1]; V_B = w_node[i_2]; V_C = w_node[i_3]; V_D = w_node[i_4]
    V_L = w_c[c_left]; V_R = w_ghost[c_right]
    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for idx in range(start, neumann.shape[0], stride):
    i = neumann[idx]
    c_left = face_cellid[i][0]
    c_right = i
    i_1 = faces[i][0]; i_2 = faces[i][1]; i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]
    V_A = w_node[i_1]; V_B = w_node[i_2]; V_C = w_node[i_3]; V_D = w_node[i_4]
    V_L = w_c[c_left]; V_R = w_ghost[c_right]
    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]


def cell_gradient_3d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                     w_halo: 'float[:]', w_haloghost: 'float[:]', cell_center: 'float[:,:]',
                     cell_cellnid: 'int[:,:]', ghost_info_flt: 'float[:, :]', ghost_ext_info_flt: 'float[:, :]',
                     cell_ghostnid: 'int[:,:]', cell_haloghostnid: 'int[:,:]', cell_halonid: 'int[:,:]',
                     cells: 'int[:,:]', cell_periodicfid: 'int[:,:]', node_periodicid: 'int[:,:]',
                     node_oldname: 'int[:]', halo_centvol: 'float[:,:]', cell_shift: 'float[:,:]',
                     w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', ghost_faceid: 'int[:]'):
  for i in range(start, w_c.shape[0], stride):
    i_xx = 0.0; i_yy = 0.0; i_zz = 0.0
    i_xy = 0.0; i_xz = 0.0; i_yz = 0.0
    j_x = 0.0; j_y = 0.0; j_z = 0.0

    for j in range(cell_cellnid[i][-1]):
      cell = cell_cellnid[i][j]
      jx = cell_center[cell][0] - cell_center[i][0]
      jy = cell_center[cell][1] - cell_center[i][1]
      jz = cell_center[cell][2] - cell_center[i][2]
      i_xx += jx * jx; i_yy += jy * jy; i_zz += jz * jz
      i_xy += jx * jy; i_xz += jx * jz; i_yz += jy * jz
      j_x += jx * (w_c[cell] - w_c[i])
      j_y += jy * (w_c[cell] - w_c[i])
      j_z += jz * (w_c[cell] - w_c[i])

    for j in range(cell_ghostnid[i][-1]):
      ghost_id = cell_ghostnid[i][j]
      jx = ghost_info_flt[ghost_id][0] - cell_center[i][0]
      jy = ghost_info_flt[ghost_id][1] - cell_center[i][1]
      jz = ghost_info_flt[ghost_id][2] - cell_center[i][2]
      i_xx += jx * jx; i_yy += jy * jy; i_zz += jz * jz
      i_xy += jx * jy; i_xz += jx * jz; i_yz += jy * jz
      j_x += jx * (w_ghost[ghost_faceid[ghost_id]] - w_c[i])
      j_y += jy * (w_ghost[ghost_faceid[ghost_id]] - w_c[i])
      j_z += jz * (w_ghost[ghost_faceid[ghost_id]] - w_c[i])

    for j in range(cell_periodicfid[i][-1]):
      cell = cell_periodicfid[i][j]
      jx = cell_center[cell][0] + cell_shift[cell][0] - cell_center[i][0]
      jy = cell_center[cell][1] + cell_shift[cell][1] - cell_center[i][1]
      jz = cell_center[cell][2] + cell_shift[cell][2] - cell_center[i][2]
      i_xx += jx * jx; i_yy += jy * jy; i_zz += jz * jz
      i_xy += jx * jy; i_xz += jx * jz; i_yz += jy * jz
      j_x += jx * (w_c[cell] - w_c[i])
      j_y += jy * (w_c[cell] - w_c[i])
      j_z += jz * (w_c[cell] - w_c[i])

    for j in range(cell_halonid[i, -1]):
      cell = cell_halonid[i][j]
      jx = halo_centvol[cell][0] - cell_center[i][0]
      jy = halo_centvol[cell][1] - cell_center[i][1]
      jz = halo_centvol[cell][2] - cell_center[i][2]
      i_xx += jx * jx; i_yy += jy * jy; i_zz += jz * jz
      i_xy += jx * jy; i_xz += jx * jz; i_yz += jy * jz
      j_x += jx * (w_halo[cell] - w_c[i])
      j_y += jy * (w_halo[cell] - w_c[i])
      j_z += jz * (w_halo[cell] - w_c[i])

    for j in range(cell_haloghostnid[i][-1]):
      ghost_id = cell_haloghostnid[i][j]
      jx = ghost_ext_info_flt[ghost_id][0] - cell_center[i][0]
      jy = ghost_ext_info_flt[ghost_id][1] - cell_center[i][1]
      jz = ghost_ext_info_flt[ghost_id][2] - cell_center[i][2]
      i_xx += jx * jx; i_yy += jy * jy; i_zz += jz * jz
      i_xy += jx * jy; i_xz += jx * jz; i_yz += jy * jz
      j_x += jx * (w_haloghost[ghost_id] - w_c[i])
      j_y += jy * (w_haloghost[ghost_id] - w_c[i])
      j_z += jz * (w_haloghost[ghost_id] - w_c[i])

    dia = i_xx * i_yy * i_zz + 2.0 * i_xy * i_xz * i_yz - i_xx * i_yz ** 2 - i_yy * i_xz ** 2 - i_zz * i_xy ** 2
    w_x[i] = ((i_yy * i_zz - i_yz ** 2) * j_x + (i_xz * i_yz - i_xy * i_zz) * j_y + (i_xy * i_yz - i_xz * i_yy) * j_z) / dia
    w_y[i] = ((i_xz * i_yz - i_xy * i_zz) * j_x + (i_xx * i_zz - i_xz ** 2) * j_y + (i_xy * i_xz - i_yz * i_xx) * j_z) / dia
    w_z[i] = ((i_xy * i_yz - i_xz * i_yy) * j_x + (i_xy * i_xz - i_yz * i_xx) * j_y + (i_xx * i_yy - i_xy ** 2) * j_z) / dia


def barthlimiter_3d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                    w_halo: 'float[:]', w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]',
                    psi: 'float[:]', face_cellid: 'int[:,:]', cell_faceid: 'int[:,:]',
                    face_name: 'int[:]', face_haloid: 'int[:]', cell_center: 'float[:,:]',
                    face_center: 'float[:,:]'):
  for i in range(start, w_c.shape[0], stride):
    psi[i] = 1.0
    w_max = w_c[i]
    w_min = w_c[i]

    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      if face_name[face] == 0 or face_name[face] > 10:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_c[face_cellid[face][1]])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_c[face_cellid[face][1]])
      elif face_name[face] == 10:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_halo[face_haloid[face]])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_halo[face_haloid[face]])
      else:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_ghost[face])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_ghost[face])

    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      r_xyz1 = face_center[face][0] - cell_center[i][0]
      r_xyz2 = face_center[face][1] - cell_center[i][1]
      r_xyz3 = face_center[face][2] - cell_center[i][2]
      delta2 = w_x[i] * r_xyz1 + w_y[i] * r_xyz2 + w_z[i] * r_xyz3
      psi_ij = 1.0
      if abs(delta2) >= 1e-10:
        if delta2 > 0.0:
          psi_ij = min(1.0, (w_max - w_c[i]) / delta2)
        if delta2 < 0.0:
          psi_ij = min(1.0, (w_min - w_c[i]) / delta2)
      psi[i] = min(psi[i], psi_ij)


def vanalbadalimiter_3d(start: 'int', stride: 'int', w_c: 'float[:]', w_ghost: 'float[:]',
                        w_halo: 'float[:]', w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]',
                        psi: 'float[:]', face_cellid: 'int[:,:]', cell_faceid: 'int[:,:]',
                        face_name: 'int[:]', face_haloid: 'int[:]', cell_center: 'float[:,:]',
                        face_center: 'float[:,:]'):
  # Smooth (van Albada / Venkatakrishnan) slope limiter -- smooth phi(y) =
  # (y^2 + 2y)/(y^2 + y + 2) of the Barth argument instead of min(1, y); reduces
  # smooth-region clipping. Drop-in for barthlimiter_3d (identical signature),
  # selected by Variable(limiter='vanalbada').
  for i in range(start, w_c.shape[0], stride):
    psi[i] = 1.0
    w_max = w_c[i]
    w_min = w_c[i]

    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      if face_name[face] == 0 or face_name[face] > 10:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_c[face_cellid[face][1]])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_c[face_cellid[face][1]])
      elif face_name[face] == 10:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_halo[face_haloid[face]])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_halo[face_haloid[face]])
      else:
        w_max = max(w_max, w_c[face_cellid[face][0]], w_ghost[face])
        w_min = min(w_min, w_c[face_cellid[face][0]], w_ghost[face])

    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      r_xyz1 = face_center[face][0] - cell_center[i][0]
      r_xyz2 = face_center[face][1] - cell_center[i][1]
      r_xyz3 = face_center[face][2] - cell_center[i][2]
      delta2 = w_x[i] * r_xyz1 + w_y[i] * r_xyz2 + w_z[i] * r_xyz3
      psi_ij = 1.0
      if abs(delta2) >= 1e-10:
        if delta2 > 0.0:
          y = (w_max - w_c[i]) / delta2
        else:
          y = (w_min - w_c[i]) / delta2
        psi_ij = (y * y + 2.0 * y) / (y * y + y + 2.0)
      psi[i] = min(psi[i], psi_ij)

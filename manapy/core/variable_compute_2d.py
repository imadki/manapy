import numpy as np
from manapy.backends.compile_fun import compile
# import manapy.backends.types as types

def _centertovertex_2d(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]', w_haloghost: 'float[:]',
                      cell_center: 'float[:,:]', halo_centvol: 'float[:,:]', node_cellid: 'int[:,:]', node_ghostid: 'int[:,:]',
                      node_haloghostid: 'int[:,:]',
                      node_periodicid: 'int[:,:]',
                      node_halonid: 'int[:,:]', nodes: 'float[:,:]', node_oldname:'int[:]', face_ghostcenter: 'float[:,:]',
                      cell_haloghostcenter: 'float[:,:]',
                      node_R_x: 'float[:]', node_R_y: 'float[:]', node_R_z: 'float[:]', node_lambda_x: 'float[:]', node_lambda_y: 'float[:]',
                      node_lambda_z: 'float[:]', node_number: 'int[:]', cell_shift: 'float[:,:]', w_n: 'float[:]'):
  w_n[:] = 0.

  nbnode = len(nodes)
  center = np.zeros(3)

  for i in range(nbnode):
    for j in range(node_cellid[i][-1]):
      cell = node_cellid[i][j]
      center[:] = cell_center[cell][:]

      xdiff = center[0] - nodes[i][0]
      ydiff = center[1] - nodes[i][1]
      alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / (
                node_number[i] + node_lambda_x[i] * node_R_x[i] + node_lambda_y[i] * node_R_y[i])

      w_n[i] += alpha * w_c[cell]

    for j in range(node_ghostid[i][-1]):
      cell = node_ghostid[i][j]
      center[:] = face_ghostcenter[cell][0:3]

      xdiff = center[0] - nodes[i][0]
      ydiff = center[1] - nodes[i][1]
      alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / (
                node_number[i] + node_lambda_x[i] * node_R_x[i] + node_lambda_y[i] * node_R_y[i])

      w_n[i] += alpha * w_ghost[cell]

    for j in range(node_haloghostid[i, -1]):
      cell = node_haloghostid[i][j]
      center[:] = cell_haloghostcenter[cell][0:3]

      xdiff = center[0] - nodes[i][0]
      ydiff = center[1] - nodes[i][1]

      alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / (
                node_number[i] + node_lambda_x[i] * node_R_x[i] + node_lambda_y[i] * node_R_y[i])

      w_n[i] += alpha * w_haloghost[cell]

    for j in range(node_halonid[i, -1]):
      cell = node_halonid[i][j]
      center[:] = halo_centvol[cell][0:3]

      xdiff = center[0] - nodes[i][0]
      ydiff = center[1] - nodes[i][1]
      alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / (
                node_number[i] + node_lambda_x[i] * node_R_x[i] + node_lambda_y[i] * node_R_y[i])

      w_n[i] += alpha * w_halo[cell]

    # TODO Must be keeped like that checked ok ;)
    if node_oldname[i] == 11 or node_oldname[i] == 22:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center[:] = cell_center[cell][0:3]

        xdiff = center[0] + cell_shift[cell][0] - nodes[i][0]
        ydiff = center[1] - nodes[i][1]
        alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / (
                  node_number[i] + node_lambda_x[i] * node_R_x[i] + node_lambda_y[i] * node_R_y[i])

        w_n[i] += alpha * w_c[cell]

    elif node_oldname[i] == 33 or node_oldname[i] == 44:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center[:] = cell_center[cell][0:3]

        xdiff = center[0] - nodes[i][0]
        ydiff = center[1] + cell_shift[cell][1] - nodes[i][1]
        alpha = (1. + node_lambda_x[i] * xdiff + node_lambda_y[i] * ydiff) / (
                  node_number[i] + node_lambda_x[i] * node_R_x[i] + node_lambda_y[i] * node_R_y[i])

        w_n[i] += alpha * w_c[cell]

def _face_gradient_2d(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]', w_node: 'float[:]',
                     cellidf: 'int[:,:]',
                     nodeidf: 'int[:,:]', face_ghostcenter: 'float[:,:]', halofid: 'int[:]', cell_center: 'float[:,:]',
                     halo_centvol: 'float[:,:]', nodes: 'float[:,:]', airDiamond: 'float[:]', normalf: 'float[:,:]',
                     f_1: 'float[:,:]', f_2: 'float[:,:]', f_3: 'float[:,:]', f_4: 'float[:,:]', cell_shift: 'float[:,:]',
                     wx_face: 'float[:]', wy_face: 'float[:]', wz_face: 'float[:]', d_innerfaces: 'int[:]',
                     d_halofaces: 'int[:]',
                     dirichletfaces: 'int[:]', neumann: 'int[:]', d_periodicfaces: 'int[:]'):

  for i in d_innerfaces:
    c_left = cellidf[i][0]
    c_right = cellidf[i][1]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = w_node[i_1]
    vi2 = w_node[i_2]
    vv1 = w_c[c_left]
    vv2 = w_c[c_right]

    wx_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    wy_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

  for i in d_periodicfaces:
    c_left = cellidf[i][0]
    c_right = cellidf[i][1]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = w_node[i_1]
    vi2 = w_node[i_2]
    vv1 = w_c[c_left]
    vv2 = w_c[c_right]

    wx_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    wy_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

  for i in d_halofaces:
    c_left = cellidf[i][0]
    c_right = halofid[i]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = w_node[i_1]
    vi2 = w_node[i_2]
    vv1 = w_c[c_left]
    vv2 = w_halo[c_right]

    wx_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    wy_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

  for i in dirichletfaces:
    c_left = cellidf[i][0]
    c_right = i

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = w_node[i_1]
    vi2 = w_node[i_2]
    vv1 = w_c[c_left]
    vv2 = w_ghost[c_right]

    wx_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    wy_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

  for i in neumann:
    c_left = cellidf[i][0]
    c_right = i

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = w_node[i_1]
    vi2 = w_node[i_2]
    vv1 = w_c[c_left]
    vv2 = w_ghost[c_right]

    wx_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    wy_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

def _cell_gradient_2d(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]', w_haloghost: 'float[:]',
                     cell_center: 'float[:,:]', cell_cellnid: 'int[:,:]', cell_ghostnid: 'int[:,:]', cell_haloghostnid: 'int[:,:]',
                     cell_halonid: 'int[:,:]',
                     cells: 'int[:,:]', cell_periodicfid: 'int[:,:]', node_periodicid: 'int[:,:]', face_ghostcenter: 'float[:,:]',
                     cell_haloghostcenter: 'float[:,:]', node_oldname: 'int[:]', halo_centvol: 'float[:,:]', cell_shift: 'float[:,:]',
                     w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]'):
  center = np.zeros(3)
  nbelement = len(w_c)

  for i in range(nbelement):
    i_xx = 0.;
    i_yy = 0.;
    i_xy = 0.
    j_xw = 0.;
    j_yw = 0.

    for j in range(cell_cellnid[i][-1]):
      cell = cell_cellnid[i][j]
      j_x = cell_center[cell][0] - cell_center[i][0]
      j_y = cell_center[cell][1] - cell_center[i][1]
      i_xx += j_x * j_x
      i_yy += j_y * j_y
      i_xy += (j_x * j_y)

      j_xw += (j_x * (w_c[cell] - w_c[i]))
      j_yw += (j_y * (w_c[cell] - w_c[i]))

    for j in range(cell_ghostnid[i][-1]):
      cell = cell_ghostnid[i][j]
      j_x = face_ghostcenter[cell][0] - cell_center[i][0]
      j_y = face_ghostcenter[cell][1] - cell_center[i][1]
      i_xx += j_x * j_x
      i_yy += j_y * j_y
      i_xy += (j_x * j_y)

      j_xw += (j_x * (w_ghost[cell] - w_c[i]))
      j_yw += (j_y * (w_ghost[cell] - w_c[i]))

    for k in range(cells[i][-1]):
      nod = cells[i][k]
      if node_oldname[nod] == 11 or node_oldname[nod] == 22:
        for j in range(node_periodicid[nod][-1]):
          cell = node_periodicid[nod][j]
          center[:] = cell_center[cell][0:3]
          j_x = center[0] + cell_shift[cell][0] - cell_center[i][0]
          j_y = center[1] - cell_center[i][1]

          i_xx += j_x * j_x
          i_yy += j_y * j_y
          i_xy += (j_x * j_y)

          j_xw += (j_x * (w_c[cell] - w_c[i]))
          j_yw += (j_y * (w_c[cell] - w_c[i]))

      if node_oldname[nod] == 33 or node_oldname[nod] == 44:
        for j in range(node_periodicid[nod][-1]):
          cell = node_periodicid[nod][j]
          center[:] = cell_center[cell][0:3]
          j_x = center[0] - cell_center[i][0]
          j_y = center[1] + cell_shift[cell][1] - cell_center[i][1]

          i_xx += j_x * j_x
          i_yy += j_y * j_y
          i_xy += (j_x * j_y)

          j_xw += (j_x * (w_c[cell] - w_c[i]))
          j_yw += (j_y * (w_c[cell] - w_c[i]))

    for j in range(cell_halonid[i, -1]):
      cell = cell_halonid[i][j]
      j_x = halo_centvol[cell][0] - cell_center[i][0]
      j_y = halo_centvol[cell][1] - cell_center[i][1]

      i_xx += j_x * j_x
      i_yy += j_y * j_y
      i_xy += (j_x * j_y)

      j_xw += (j_x * (w_halo[cell] - w_c[i]))
      j_yw += (j_y * (w_halo[cell] - w_c[i]))

      for j in range(cell_haloghostnid[i][-1]):
        cell = cell_haloghostnid[i][j]
        center[:] = cell_haloghostcenter[cell]

        j_x = center[0] - cell_center[i][0]
        j_y = center[1] - cell_center[i][1]

        i_xx += j_x * j_x
        i_yy += j_y * j_y
        i_xy += (j_x * j_y)

        j_xw += (j_x * (w_haloghost[cell] - w_c[i]))
        j_yw += (j_y * (w_haloghost[cell] - w_c[i]))

    dia = i_xx * i_yy - i_xy * i_xy

    w_x[i] = (i_yy * j_xw - i_xy * j_yw) / dia
    w_y[i] = (i_xx * j_yw - i_xy * j_xw) / dia
    w_z[i] = 0.

def _barthlimiter_2d(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
                    w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', psi: 'float[:]',
                    face_cellid: 'int[:,:]', cell_faceid: 'int[:,:]', face_name: 'int[:]',
                    face_haloid: 'int[:]', cell_center: 'float[:,:]', face_center: 'float[:,:]'):
  nbelement = len(w_c)
  val = 1.
  psi[:] = val

  for i in range(nbelement):
    w_max = w_c[i]
    w_min = w_c[i]

    for j in range(cell_faceid[i][-1]):
      face = cell_faceid[i][j]
      if face_name[face] == 0 or face_name[face] > 10:  #
        # 11 or face_name[face] == 22 or face_name[face] == 33 or face_name[face] == 44:
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

      # TODO choice of epsilon
      if np.fabs(delta2) < 1e-8:
        psi_ij = 1.
      else:
        if delta2 > 0.:
          value = (w_max - w_c[i]) / delta2
          psi_ij = min(val, value)
        if delta2 < 0.:
          value = (w_min - w_c[i]) / delta2
          psi_ij = min(val, value)

      psi[i] = min(psi[i], psi_ij)

############################################################################33
# Public
centertovertex_2d = compile(_centertovertex_2d)
face_gradient_2d = compile(_face_gradient_2d)
cell_gradient_2d = compile(_cell_gradient_2d)
barthlimiter_2d = compile(_barthlimiter_2d)


from manapy.backends.compile_fun import compile
import numpy as np

def _centertovertex_3d(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]', w_haloghost: 'float[:]',
                      cell_center: 'float[:,:]', halo_centvol: 'float[:,:]', node_cellid: 'int[:,:]', node_ghostid: 'int[:,:]',
                      node_haloghostid: 'int[:,:]',
                      node_periodicid: 'int[:,:]',
                      node_halonid: 'int[:,:]', nodes: 'float[:,:]', node_oldname: 'int[:]', face_ghostcenter: 'float[:,:]', cell_haloghostcenter: 'float[:,:]',
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
      zdiff = center[2] - nodes[i][2]

      alpha = (1. + node_lambda_x[i] * xdiff + \
               node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / (node_number[i] + node_lambda_x[i] * node_R_x[i] + \
                                                             node_lambda_y[i] * node_R_y[i] + node_lambda_z[i] * node_R_z[i])
      w_n[i] += alpha * w_c[cell]

    for j in range(node_ghostid[i][-1]):
      cell = node_ghostid[i][j]
      center[:] = face_ghostcenter[cell][0:3]

      xdiff = center[0] - nodes[i][0]
      ydiff = center[1] - nodes[i][1]
      zdiff = center[2] - nodes[i][2]

      alpha = (1. + node_lambda_x[i] * xdiff + \
               node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / (node_number[i] + node_lambda_x[i] * node_R_x[i] + \
                                                             node_lambda_y[i] * node_R_y[i] + node_lambda_z[i] * node_R_z[i])

      w_n[i] += alpha * w_ghost[cell]

    for j in range(node_haloghostid[i][-1]):
      cell = node_haloghostid[i][j]
      center[:] = cell_haloghostcenter[cell]

      xdiff = center[0] - nodes[i][0]
      ydiff = center[1] - nodes[i][1]
      zdiff = center[2] - nodes[i][2]

      alpha = (1. + node_lambda_x[i] * xdiff + \
               node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / (node_number[i] + node_lambda_x[i] * node_R_x[i] + \
                                                             node_lambda_y[i] * node_R_y[i] + node_lambda_z[i] * node_R_z[i])

      w_n[i] += alpha * w_haloghost[cell]

    for j in range(node_halonid[i][-1]):
      cell = node_halonid[i][j]
      center[:] = halo_centvol[cell][0:3]

      xdiff = center[0] - nodes[i][0]
      ydiff = center[1] - nodes[i][1]
      zdiff = center[2] - nodes[i][2]

      alpha = (1. + node_lambda_x[i] * xdiff + \
               node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / (node_number[i] + node_lambda_x[i] * node_R_x[i] + \
                                                             node_lambda_y[i] * node_R_y[i] + node_lambda_z[i] * node_R_z[i])

      w_n[i] += alpha * w_halo[cell]

    if node_oldname[i] == 11 or node_oldname[i] == 22:

      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center[:] = cell_center[cell][0:3]

        xdiff = center[0] + cell_shift[cell][0] - nodes[i][0]
        ydiff = center[1] - nodes[i][1]
        zdiff = center[2] - nodes[i][2]

        alpha = (1. + node_lambda_x[i] * xdiff + \
                 node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / (node_number[i] + node_lambda_x[i] * node_R_x[i] + \
                                                               node_lambda_y[i] * node_R_y[i] + node_lambda_z[i] * node_R_z[i])

        w_n[i] += alpha * w_c[cell]

    elif node_oldname[i] == 33 or node_oldname[i] == 44:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center[:] = cell_center[cell][0:3]

        xdiff = center[0] - nodes[i][0]
        ydiff = center[1] + cell_shift[cell][1] - nodes[i][1]
        zdiff = center[2] - nodes[i][2]

        alpha = (1. + node_lambda_x[i] * xdiff + \
                 node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / (node_number[i] + node_lambda_x[i] * node_R_x[i] + \
                                                               node_lambda_y[i] * node_R_y[i] + node_lambda_z[i] * node_R_z[i])
        w_n[i] += alpha * w_c[cell]

    elif node_oldname[i] == 55 or node_oldname[i] == 66:
      for j in range(node_periodicid[i][-1]):
        cell = node_periodicid[i][j]
        center[:] = cell_center[cell][0:3]

        xdiff = center[0] - nodes[i][0]
        ydiff = center[1] - nodes[i][1]
        zdiff = center[2] + cell_shift[cell][2] - nodes[i][2]

        alpha = (1. + node_lambda_x[i] * xdiff + \
                 node_lambda_y[i] * ydiff + node_lambda_z[i] * zdiff) / (node_number[i] + node_lambda_x[i] * node_R_x[i] + \
                                                               node_lambda_y[i] * node_R_y[i] + node_lambda_z[i] * node_R_z[i])
        w_n[i] += alpha * w_c[cell]


def _face_gradient_3d(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]', w_node: 'float[:]',
                     face_cellid: 'int[:,:]',
                     faces: 'int[:,:]', face_ghostcenter: 'float[:,:]', face_haloid: 'int[:]', cell_center: 'float[:,:]',
                     halo_centvol: 'float[:,:]', nodes: 'float[:,:]', face_air_diamond: 'float[:]', face_normal: 'float[:,:]',
                     face_f1: 'float[:,:]', face_f2: 'float[:,:]', face_f3: 'float[:,:]', face_f4: 'float[:,:]', cell_shift: 'float[:,:]',
                     wx_face: 'float[:]', wy_face: 'float[:]', wz_face: 'float[:]', d_innerfaces: 'int[:]',
                     d_halofaces: 'int[:]',
                     dirichletfaces: 'int[:]', neumann: 'int[:]', d_periodicboundaryfaces: 'int[:]'):

  for i in d_innerfaces:

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = w_node[i_1]
    V_B = w_node[i_2]
    V_C = w_node[i_3]
    V_D = w_node[i_4]

    V_L = w_c[c_left]
    V_R = w_c[c_right]

    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for i in d_periodicboundaryfaces:

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = w_node[i_1]
    V_B = w_node[i_2]
    V_C = w_node[i_3]
    V_D = w_node[i_4]

    V_L = w_c[c_left]
    V_R = w_c[c_right]

    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for i in d_halofaces:

    c_left = face_cellid[i][0]
    c_right = face_haloid[i]

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = w_node[i_1]
    V_B = w_node[i_2]
    V_C = w_node[i_3]
    V_D = w_node[i_4]

    V_L = w_c[c_left]
    V_R = w_halo[c_right]

    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for i in dirichletfaces:

    c_left = face_cellid[i][0]
    c_right = i

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = w_node[i_1]
    V_B = w_node[i_2]
    V_C = w_node[i_3]
    V_D = w_node[i_4]

    V_L = w_c[c_left]
    V_R = w_ghost[c_right]

    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for i in neumann:

    c_left = face_cellid[i][0]
    c_right = i

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = w_node[i_1]
    V_B = w_node[i_2]
    V_C = w_node[i_3]
    V_D = w_node[i_4]

    V_L = w_c[c_left]
    V_R = w_ghost[c_right]

    wx_face[i] = (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    wy_face[i] = (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    wz_face[i] = (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

def _cell_gradient_3d(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]', w_haloghost: 'float[:]',
                     cell_center: 'float[:,:]', cell_cellnid: 'int[:,:]', cell_ghostnid: 'int[:,:]', cell_haloghostnid: 'int[:,:]',
                     cell_halonid: 'int[:,:]',
                     cells: 'int[:,:]', cell_periodicfid: 'int[:,:]', node_periodicid: 'int[:,:]', face_ghostcenter: 'float[:,:]',
                     cell_haloghostcenter: 'float[:,:]', nodes: 'float[:,:]', halo_centvol: 'float[:,:]', cell_shift: 'float[:,:]',
                     w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]'):

  nbelement = len(w_c)
  center = np.zeros(3)

  for i in range(nbelement):
    i_xx = 0.
    i_yy = 0.
    i_zz = 0.
    i_xy = 0.
    i_xz = 0.
    i_yz = 0.

    j_x = 0.
    j_y = 0.
    j_z = 0.

    for j in range(cell_cellnid[i][-1]):
      cell = cell_cellnid[i][j]
      jx = cell_center[cell][0] - cell_center[i][0]
      jy = cell_center[cell][1] - cell_center[i][1]
      jz = cell_center[cell][2] - cell_center[i][2]

      i_xx += jx * jx
      i_yy += jy * jy
      i_zz += jz * jz
      i_xy += (jx * jy)
      i_xz += (jx * jz)
      i_yz += (jy * jz)

      j_x += (jx * (w_c[cell] - w_c[i]))
      j_y += (jy * (w_c[cell] - w_c[i]))
      j_z += (jz * (w_c[cell] - w_c[i]))

    for j in range(cell_ghostnid[i][-1]):
      cell = cell_ghostnid[i][j]
      jx = face_ghostcenter[cell][0] - cell_center[i][0]
      jy = face_ghostcenter[cell][1] - cell_center[i][1]
      jz = face_ghostcenter[cell][2] - cell_center[i][2]

      i_xx += jx * jx
      i_yy += jy * jy
      i_zz += jz * jz
      i_xy += (jx * jy)
      i_xz += (jx * jz)
      i_yz += (jy * jz)

      j_x += (jx * (w_ghost[cell] - w_c[i]))
      j_y += (jy * (w_ghost[cell] - w_c[i]))
      j_z += (jz * (w_ghost[cell] - w_c[i]))

    for j in range(cell_periodicfid[i][-1]):
      cell = cell_periodicfid[i][j]
      center[:] = cell_center[cell][0:3]
      jx = center[0] + cell_shift[cell][0] - cell_center[i][0]
      jy = center[1] + cell_shift[cell][1] - cell_center[i][1]
      jz = center[2] + cell_shift[cell][2] - cell_center[i][2]

      i_xx += jx * jx
      i_yy += jy * jy
      i_zz += jz * jz
      i_xy += (jx * jy)
      i_xz += (jx * jz)
      i_yz += (jy * jz)

      j_x += (jx * (w_c[cell] - w_c[i]))
      j_y += (jy * (w_c[cell] - w_c[i]))
      j_z += (jz * (w_c[cell] - w_c[i]))

    # if nbproc > 1:
    for j in range(cell_halonid[i][-1]):
      cell = cell_halonid[i][j]

      jx = halo_centvol[cell][0] - cell_center[i][0]
      jy = halo_centvol[cell][1] - cell_center[i][1]
      jz = halo_centvol[cell][2] - cell_center[i][2]

      i_xx += jx * jx
      i_yy += jy * jy
      i_zz += jz * jz
      i_xy += (jx * jy)
      i_xz += (jx * jz)
      i_yz += (jy * jz)

      j_x += (jx * (w_halo[cell] - w_c[i]))
      j_y += (jy * (w_halo[cell] - w_c[i]))
      j_z += (jz * (w_halo[cell] - w_c[i]))

    for j in range(cell_haloghostnid[i][-1]):
      # -3 the index of global face
      cell = cell_haloghostnid[i][j]
      center[:] = cell_haloghostcenter[cell]

      jx = center[0] - cell_center[i][0]
      jy = center[1] - cell_center[i][1]
      jz = center[2] - cell_center[i][2]

      i_xx += jx * jx
      i_yy += jy * jy
      i_zz += jz * jz
      i_xy += (jx * jy)
      i_xz += (jx * jz)
      i_yz += (jy * jz)

      j_x += (jx * (w_haloghost[cell] - w_c[i]))
      j_y += (jy * (w_haloghost[cell] - w_c[i]))
      j_z += (jz * (w_haloghost[cell] - w_c[i]))

    dia = i_xx * i_yy * i_zz + 2. * i_xy * i_xz * i_yz - i_xx * i_yz ** 2 - i_yy * i_xz ** 2 - i_zz * i_xy ** 2

    w_x[i] = ((i_yy * i_zz - i_yz ** 2) * j_x + (i_xz * i_yz - i_xy * i_zz) * j_y + (
              i_xy * i_yz - i_xz * i_yy) * j_z) / dia
    w_y[i] = ((i_xz * i_yz - i_xy * i_zz) * j_x + (i_xx * i_zz - i_xz ** 2) * j_y + (
              i_xy * i_xz - i_yz * i_xx) * j_z) / dia
    w_z[i] = ((i_xy * i_yz - i_xz * i_yy) * j_x + (i_xy * i_xz - i_yz * i_xx) * j_y + (
              i_xx * i_yy - i_xy ** 2) * j_z) / dia


def _barthlimiter_3d(w_c: 'float[:]', w_ghost: 'float[:]', w_halo: 'float[:]',
                    w_x: 'float[:]', w_y: 'float[:]', w_z: 'float[:]', psi: 'float[:]',
                    face_cellid: 'int[:,:]', cell_faceid: 'int[:,:]', face_name: 'int[:]',
                    face_haloid: 'int[:]', cell_center: 'float[:,:]', face_center: 'float[:,:]'):

  nbelement = len(w_c)
  psi[:] = 1.

  for i in range(nbelement):
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

      # TODO choice of epsilon
      if np.fabs(delta2) < 1e-10:
        psi_ij = 1.
      else:
        if delta2 > 0.:
          value = (w_max - w_c[i]) / delta2
          psi_ij = min(1., value)
        if delta2 < 0.:
          value = (w_min - w_c[i]) / delta2
          psi_ij = min(1., value)

      psi[i] = min(psi[i], psi_ij)


############################################################################
# Public

cell_gradient_3d = compile(_cell_gradient_3d)
face_gradient_3d = compile(_face_gradient_3d)
centertovertex_3d = compile(_centertovertex_3d)
barthlimiter_3d = compile(_barthlimiter_3d)



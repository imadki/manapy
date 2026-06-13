from manapy.backends.compile_fun import compile
import numpy as np

### UTILS
def _convert_solution(x1: 'float[:]', x1converted: 'float[:]', cell_tc: 'int[:]', b0Size: 'int'):
  for i in range(b0Size):
    x1converted[i] = x1[cell_tc[i]]

def _search_element(a: 'int[:]', target_value: 'int'):
  find = 0
  for val in a:
    if val == target_value:
      find = 1
      break
  return find


def _rhs_value_dirichlet_node(Pbordnode: 'float[:]', nodes: 'int[:]', value: 'float[:]'):
  for i in nodes:
    Pbordnode[i] = value[i]


def _rhs_value_dirichlet_face(Pbordface: 'float[:]', faces: 'int[:]', value: 'float[:]'):
  for i in faces:
    Pbordface[i] = value[i]

def _compute_P_gradient_2d_diamond(P_c: 'float[:]', P_ghost: 'float[:]', P_halo: 'float[:]', P_node: 'float[:]',
                                  face_cellid: 'int[:,:]',
                                  faces: 'int[:,:]', face_haloid: 'int[:]', node_oldname: 'int[:]', face_air_diamond: 'float[:]',
                                  face_f1: 'float[:,:]', face_f2: 'float[:,:]',
                                  face_f3: 'float[:,:]', face_f4: 'float[:,:]', face_normal: 'float[:,:]', cell_shift: 'float[:,:]',
                                  Pbordnode: 'float[:]',
                                  Pbordface: 'float[:]',
                                  Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]',
                                  BCdirichlet: 'int[:]', d_innerfaces: 'int[:]',
                                  d_halofaces: 'int[:]', neumannfaces: 'int[:]', dirichletfaces: 'int[:]',
                                  d_periodicboundaryfaces: 'int[:]'):

  for i in d_innerfaces:

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    vi1 = P_node[i_1]
    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      vi1 = Pbordnode[i_1]
    vi2 = P_node[i_2]
    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      vi2 = Pbordnode[i_2]

    vv1 = P_c[c_left]
    vv2 = P_c[c_right]

    Px_face[i] = -1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    Py_face[i] = 1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])

  for i in d_periodicboundaryfaces:

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    vi1 = P_node[i_1]
    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      vi1 = Pbordnode[i_1]
    vi2 = P_node[i_2]
    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      vi2 = Pbordnode[i_2]

    vv1 = P_c[c_left]
    vv2 = P_c[c_right]

    Px_face[i] = -1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    Py_face[i] = 1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])

  for i in neumannfaces:

    c_left = face_cellid[i][0]
    c_right = i

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    vi1 = P_node[i_1]
    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      vi1 = Pbordnode[i_1]
    vi2 = P_node[i_2]
    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      vi2 = Pbordnode[i_2]

    vv1 = P_c[c_left]
    vv2 = P_ghost[c_right]

    Px_face[i] = -1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    Py_face[i] = 1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])

  for i in d_halofaces:

    c_left = face_cellid[i][0]
    c_right = face_haloid[i]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    vi1 = P_node[i_1]
    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      vi1 = Pbordnode[i_1]
    vi2 = P_node[i_2]
    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      vi2 = Pbordnode[i_2]

    vv1 = P_c[c_left]
    vv2 = P_halo[c_right]

    Px_face[i] = -1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    Py_face[i] = 1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])

  for i in dirichletfaces:
    c_left = face_cellid[i][0]
    c_right = i

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    vi1 = Pbordnode[i_1]
    vi2 = Pbordnode[i_2]
    vv1 = P_c[c_left]

    VK = Pbordface[i]
    vv2 = 2. * VK - vv1

    Px_face[i] = -1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][1] + (vv1 + vi2) * face_f2[i][1] + (vi2 + vv2) * face_f3[i][1] + (vv2 + vi1) * face_f4[i][1])
    Py_face[i] = 1 / (2 * face_air_diamond[i]) * (
              (vi1 + vv1) * face_f1[i][0] + (vv1 + vi2) * face_f2[i][0] + (vi2 + vv2) * face_f3[i][0] + (vv2 + vi1) * face_f4[i][0])

def _get_triplet_2d(face_cellid: 'int[:,:]', faces: 'int[:,:]', nodes: 'float[:,:]', face_haloid: 'int[:]',
                   halo_halosext: 'int[:,:]', node_oldname: 'int[:]', cell_volume: 'float[:]',
                   node_cellid: 'int[:,:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                   node_haloid: 'int[:,:]', node_periodicid: 'int[:,:]',
                   ghost_info_flt: 'float[:, :]', ghost_ext_info_flt: 'float[:, :]', ghost_info_int: 'int[:, :]', ghost_ext_info_int: 'int[:, :]', node_ghostid: 'int[:, :]', node_haloghostid: 'int[:, :]',
                   face_air_diamond: 'float[:]',
                   node_lambda_x: 'float[:]', node_lambda_y: 'float[:]', node_lambda_z: 'float[:]', node_number: 'int[:]',
                   node_R_x: 'float[:]', node_R_y: 'float[:]',
                   node_R_z: 'float[:]', face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]', face_param4: 'float[:]',
                   cell_shift: 'float[:,:]',
                   nbelements: 'int', cell_loctoglob: 'int[:]', BCdirichlet: 'int[:]', a_loc: 'float[:]',
                   irn_loc: 'int[:]', jcn_loc: 'int[:]',
                   matrixinnerfaces: 'int[:]', d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):

  center = np.zeros(2)
  parameters = np.zeros(2)
  cmpt = 0

  for i in matrixinnerfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    parameters[0] = face_param4[i];
    parameters[1] = face_param2[i]

    c_right = face_cellid[i][1]
    c_rightglob = cell_loctoglob[c_right]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = face_param1[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    cmptparam = 0
    for nod in faces[i][:faces[i][-1]]:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          center[:] = cell_center[node_cellid[nod][j]][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          ghost_id = node_ghostid[nod, j]
          center[:] = ghost_info_flt[ghost_id][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          index = ghost_info_int[ghost_id, 0]
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[index]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[index]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          ghost_id = node_haloghostid[nod, j]
          center[:] = ghost_ext_info_flt[ghost_id][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          index = ghost_ext_info_int[ghost_id, 0]
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[index][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = halo_halosext[index][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_periodicid[nod][-1]):
          if nodes[nod][3] == 11 or nodes[nod][3] == 22:
            center[0] = cell_center[node_periodicid[nod][j]][0] + cell_shift[node_periodicid[nod][j]][0]
            center[1] = cell_center[node_periodicid[nod][j]][1]
          if nodes[nod][3] == 33 or nodes[nod][3] == 44:
            center[0] = cell_center[node_periodicid[nod][j]][0]
            center[1] = cell_center[node_periodicid[nod][j]][1] + cell_shift[node_periodicid[nod][j]][1]

          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_periodicid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[node_periodicid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          center[:] = halo_centvol[node_haloid[nod][j]][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
      cmptparam += 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    # right cell------------------------------------------------------
    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * face_param1[i] / cell_volume[c_right]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_rightglob
    value = -1. * face_param3[i] / cell_volume[c_right]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

  for i in d_halofaces:

    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    parameters[0] = face_param4[i];
    parameters[1] = face_param2[i]

    c_rightglob = halo_halosext[face_haloid[i]][0]
    c_right = face_haloid[i]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = face_param1[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    cmptparam = 0
    for nod in faces[i][:faces[i][-1]]:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          center[:] = cell_center[node_cellid[nod][j]][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          ghost_id = node_ghostid[nod][j]
          center[:] = ghost_info_flt[ghost_id][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          index = ghost_info_int[ghost_id, 0]
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[index]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          ghost_id = node_haloghostid[nod][j]
          center[:] = ghost_ext_info_flt[ghost_id][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          index = ghost_ext_info_int[ghost_id, 0]
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[index][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          center[:] = halo_centvol[node_haloid[nod][j]][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
      cmptparam += 1

  for i in dirichletfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = face_param1[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1


def _compute_2dmatrix_size(faces: 'int[:,:]', node_cellid: 'int[:,:]', node_haloid: 'int[:,:]',
                          node_periodicid: 'int[:,:]',
                          node_ghostid: 'int[:, :]', node_haloghostid: 'int[:, :]', node_oldname: 'int[:]',
                          BCdirichlet: 'int[:]',
                          matrixinnerfaces: 'int[:]', d_halofaces: 'int[:]',
                          dirichletfaces: 'int[:]'):

  cmpt = 0

  for i in matrixinnerfaces:
    cmpt = cmpt + 1

    for nod in faces[i][:faces[i][-1]]:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:  # and search_element(BCneumannNH, node_oldname[nod]) == 0:
        # if nodes[nod][3] not in BCdirichlet:
        for j in range(node_cellid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(node_periodicid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

    cmpt = cmpt + 1
    # right cell------------------------------------------------------
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  # elif namef[i] == 10:
  for i in d_halofaces:
    cmpt = cmpt + 1

    cmpt = cmpt + 1
    cmpt = cmpt + 1

    for nod in faces[i][:faces[i][-1]]:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          cmpt = cmpt + 1

  for i in dirichletfaces:
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  return cmpt

def _get_rhs_glob_2d(face_cellid: 'int[:,:]', faces: 'int[:,:]', node_oldname: 'int[:]',
                    cell_volume: 'float[:]', node_ghostid: 'int[:,:]', cell_loctoglob: 'int[:]', face_param1: 'float[:]',
                    face_param2: 'float[:]',
                    face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'int[:]', matrixinnerfaces: 'int[:]',
                    d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):

  rhs[:] = 0.
  for i in matrixinnerfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    c_right = face_cellid[i][1]
    c_rightglob = cell_loctoglob[c_right]

    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

      value_right = V * face_param4[i] / cell_volume[c_right]
      rhs[c_rightglob] += value_right

    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

      value_right = V * face_param2[i] / cell_volume[c_right]
      rhs[c_rightglob] += value_right

  for i in d_halofaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

  for i in dirichletfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    if node_ghostid[i_1, -1] > 0:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    if node_ghostid[i_2, -1] > 0:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs[c_leftglob] += value

def _get_rhs_loc_2d(face_cellid: 'int[:,:]', faces: 'int[:,:]', node_oldname: 'int[:]',
                   cell_volume: 'float[:]', node_ghostid: 'int[:,:]', cell_loctoglob: 'int[:]', face_param1: 'float[:]',
                   face_param2: 'float[:]',
                   face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                   rhs_loc: 'float[:]',
                   BCdirichlet: 'int[:]', matrixinnerfaces: 'int[:]',
                   d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):


  rhs_loc[:] = 0.

  for i in matrixinnerfaces:
    c_right = face_cellid[i][1]
    c_left = face_cellid[i][0]
    i_1 = faces[i][0]
    i_2 = faces[i][1]

    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs_loc[c_left] += value_left

      value_right = V * face_param4[i] / cell_volume[c_right]
      rhs_loc[c_right] += value_right

    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs_loc[c_left] += value_left

      value_right = V * face_param2[i] / cell_volume[c_right]
      rhs_loc[c_right] += value_right

  for i in d_halofaces:
    c_left = face_cellid[i][0]
    i_1 = faces[i][0]
    i_2 = faces[i][1]

    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs_loc[c_left] += value_left

    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs_loc[c_left] += value_left

  # TODO verify
  for i in dirichletfaces:

    c_left = face_cellid[i][0]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    if node_ghostid[i_1, -1] > 0:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs_loc[c_left] += value_left

    if node_ghostid[i_2, -1] > 0:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs_loc[c_left] += value_left

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs_loc[c_left] += value


def _get_triplet_2d_with_contrib(face_cellid: 'int[:,:]', faces: 'int[:,:]', cell_faceid: 'int[:,:]',
                                nodes: 'float[:,:]',
                                halo_halosext: 'int[:,:]', node_oldname: 'int[:]', cell_volume: 'float[:]',
                                node_cellid: 'int[:,:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                node_haloid: 'int[:,:]',
                                ghost_info_flt: 'float[:,:]', ghost_info_int: 'int[:, :]', ghost_ext_info_flt: 'float[:,:]', ghost_ext_info_int: 'int[:, :]', node_ghostid: 'int[:, :]', node_haloghostid: 'int[:, :]',
                                node_lambda_x: 'float[:]', node_lambda_y: 'float[:]', node_number: 'int[:]', node_R_x: 'float[:]',
                                node_R_y: 'float[:]', param1: 'float[:]',
                                param2: 'float[:]', param3: 'float[:]', param4: 'float[:]', cell_loctoglob: 'int[:]',
                                BCdirichlet: 'int[:]', a_loc: 'float[:]', irn_loc: 'int[:]', jcn_loc: 'int[:]',
                                matrixinnerfaces: 'int[:]', dirichletfaces: 'int[:]',
                                Icell: 'float[:]',
                                alpha_P: 'float', perm_vec: 'float[:]',
                                visc_vec: 'float[:]'):


  center = np.zeros(2)
  parameters = np.zeros(2)
  cmpt = 0

  for i in matrixinnerfaces:
    nbfL = cell_faceid[face_cellid[i][0]][-1]
    nbfR = cell_faceid[face_cellid[i][1]][-1]
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    parameters[0] = param4[i]
    parameters[1] = param2[i]

    c_right = face_cellid[i][1]
    c_rightglob = cell_loctoglob[c_right]

    # perm_visc = dist[i][0]/(dist[i][1]/(perm_vec[c_rightglob]/visc_vec[c_rightglob]) + dist[i][2]/(perm_vec[c_leftglob]/visc_vec[c_leftglob]))

    # perm = dist[i][0]/(dist[i][1]/perm_vec[c_rightglob] + dist[i][2]/perm_vec[c_leftglob])
    # visc = dist[i][0]/(dist[i][1]/visc_vec[c_rightglob] + dist[i][2]/visc_vec[c_leftglob])

    perm = 0.5 * (perm_vec[c_rightglob] + perm_vec[c_leftglob])
    visc = 0.5 * (visc_vec[c_rightglob] + visc_vec[c_leftglob])

    perm_visc = perm / visc

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = param1[i] / cell_volume[c_left]
    a_loc[cmpt] = value * Icell[c_left] * perm_visc + (1 / nbfL) * cell_volume[c_left] * alpha_P * (1 - Icell[c_left])
    cmpt = cmpt + 1

    cmptparam = 0
    for nod in faces[i][:faces[i][-1]]:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:  # and search_element(BCneumannNH, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          center[:] = cell_center[node_cellid[nod][j]][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value * Icell[c_left] * perm_visc
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value * Icell[c_right] * perm_visc
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          ghost_id = node_ghostid[nod, j]
          center[:] = ghost_info_flt[ghost_id][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          index = ghost_info_int[ghost_id, 0]
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[index]
          a_loc[cmpt] = value * Icell[c_left] * perm_visc
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[index]
          a_loc[cmpt] = value * Icell[c_right] * perm_visc
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          ghost_id = node_haloghostid[nod, j]
          center[:] = ghost_ext_info_flt[ghost_id][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          index = ghost_ext_info_int[ghost_id, 0]
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[index][0]
          a_loc[cmpt] = value * Icell[c_left] * perm_visc
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = halo_halosext[index][0]
          # TODO
          a_loc[cmpt] = value * Icell[c_right] * (
            perm_visc)  # value*Ihaloghost[ghost_ext_info_int[ghost_id][0]]*(perm/visc)
          cmpt = cmpt + 1

          for j in range(node_haloid[nod][-1]):
            center[:] = halo_centvol[node_haloid[nod][j]][0:2]
            xdiff = center[0] - nodes[nod][0]
            ydiff = center[1] - nodes[nod][1]
            alpha = (1. + node_lambda_x[nod] * xdiff +
                     node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            value = alpha / cell_volume[c_left] * parameters[cmptparam]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
            a_loc[cmpt] = value * Icell[c_left] * perm_visc
            cmpt = cmpt + 1
            # right cell-----------------------------------
            value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
            a_loc[cmpt] = value * Icell[c_right] * perm_visc  # value*Ihalo[node_haloid[nod][j]]*(perm/visc)
            cmpt = cmpt + 1

      cmptparam = +1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value * Icell[c_left] * perm_visc
    cmpt = cmpt + 1

    # right cell------------------------------------------------------
    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * param1[i] / cell_volume[c_right]
    a_loc[cmpt] = value * Icell[c_right] * perm_visc
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_rightglob
    value = -1. * param3[i] / cell_volume[c_right]
    a_loc[cmpt] = value * Icell[c_right] * perm_visc + (1 / nbfR) * cell_volume[c_right] * alpha_P * (1 - Icell[c_right])
    cmpt = cmpt + 1
  '''
  for i in halofaces:
      nbfL = faceidc[cellfid[i][0]][-1]

      c_left = cellfid[i][0]
      c_leftglob  = loctoglob[c_left]

      perm = perm_vec[c_leftglob] 
      visc = visc_vec[c_leftglob]

      parameters[0] = param4[i]; parameters[1] = param2[i]

      c_rightglob = haloext[halofid[i]][0]
      c_right     = halofid[i]

      irn_loc[cmpt] = c_leftglob
      jcn_loc[cmpt] = c_leftglob
      value =  param1[i] / volume[c_left]
      a_loc[cmpt] = value*Icell[c_left]*(perm/visc) + (1/nbfL)*volume[c_left]*alpha_P*(1 - Icell[c_left])
      cmpt = cmpt + 1

      irn_loc[cmpt] = c_leftglob
      jcn_loc[cmpt] = c_rightglob
      value =  param3[i] / volume[c_left]
      a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
      cmpt = cmpt + 1

      cmptparam = 0
      for nod in nodeidf[i]:
          if search_element(BCdirichlet, oldnamen[nod]) == 0 and search_element(BCneumannNH, oldnamen[nod]) == 0: 
              for j in range(cellnid[nod][-1]):
                  center[:] = centerc[cellnid[nod][j]][0:2]
                  xdiff = center[0] - vertexn[nod][0]
                  ydiff = center[1] - vertexn[nod][1]
                  alpha = (1. + lambda_x[nod]*xdiff + \
                            lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                  value =  alpha / volume[c_left] * parameters[cmptparam]
                  irn_loc[cmpt] = c_leftglob
                  jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
                  a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
                  cmpt = cmpt + 1

              for j in range(len(centergn[nod])):
                  if centergn[nod][j][-1] != -1:
                      center[:] = centergn[nod][j][0:2]
                      xdiff = center[0] - vertexn[nod][0]
                      ydiff = center[1] - vertexn[nod][1]
                      alpha = (1. + lambda_x[nod]*xdiff + \
                                lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                      index = int(centergn[nod][j][2])
                      value = alpha / volume[c_left] * parameters[cmptparam]
                      irn_loc[cmpt] = c_leftglob
                      jcn_loc[cmpt] = loctoglob[index]
                      a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
                      cmpt = cmpt + 1

              for j in range(len(halocentergn[nod])):
                  if halocentergn[nod][j][-1] != -1:
                      center[:] = halocentergn[nod][j][0:2]
                      xdiff = center[0] - vertexn[nod][0]
                      ydiff = center[1] - vertexn[nod][1]
                      alpha = (1. + lambda_x[nod]*xdiff + \
                                lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                      index = int(halocentergn[nod][j][2])
                      value = alpha / volume[c_left] * parameters[cmptparam]
                      irn_loc[cmpt] = c_leftglob
                      jcn_loc[cmpt] = haloext[index][0]
                      a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
                      cmpt = cmpt + 1

              for j in range(halonid[nod][-1]):
                  center[:] = centerh[halonid[nod][j]][0:2]
                  xdiff = center[0] - vertexn[nod][0]
                  ydiff = center[1] - vertexn[nod][1]
                  alpha = (1. + lambda_x[nod]*xdiff + \
                            lambda_y[nod]*ydiff)/(number[nod] + lambda_x[nod]*R_x[nod] + lambda_y[nod]*R_y[nod])
                  value =  alpha / volume[c_left] * parameters[cmptparam]
                  irn_loc[cmpt] = c_leftglob
                  jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
                  a_loc[cmpt] = value*Icell[c_left]*(perm/visc)
                  cmpt = cmpt + 1
          cmptparam +=1
  '''
  for i in dirichletfaces:
    nbfL = cell_faceid[face_cellid[i][0]][-1]
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    perm = perm_vec[c_leftglob]
    visc = visc_vec[c_leftglob]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = param1[i] / cell_volume[c_left]
    a_loc[cmpt] = value * Icell[c_left] * (perm / visc) + (1 / nbfL) * cell_volume[c_left] * alpha_P * (1 - Icell[c_left])
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value * Icell[c_left] * (perm / visc) + (1 / nbfL) * cell_volume[c_left] * alpha_P * (1 - Icell[c_left])
    cmpt = cmpt + 1


def _get_rhs_glob_2d_with_contrib(face_cellid: 'int[:,:]', faces: 'int[:,:]', node_oldname: 'int[:]',
                                 cell_volume: 'float[:]', node_ghostid: 'int[:, :]', cell_loctoglob: 'int[:]',
                                 face_param1: 'float[:]', face_param2: 'float[:]',
                                 face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]',
                                 Pbordface: 'float[:]', rhs: 'float[:]',
                                 BCdirichlet: 'int[:]', matrixinnerfaces: 'int[:]',
                                 d_halofaces: 'int[:]', dirichletfaces: 'int[:]', neumannNHfaces: 'int[:]',
                                 Icell: 'float[:]', perm_vec: 'float[:]', visc_vec: 'float[:]',
                                 cst: 'float', normalf: 'float[:,:]'):

  rhs[:] = 0.
  for i in matrixinnerfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    c_right = face_cellid[i][1]
    c_rightglob = cell_loctoglob[c_right]

    perm = 0.5 * (perm_vec[c_rightglob] + perm_vec[c_leftglob])
    visc = 0.5 * (visc_vec[c_rightglob] + visc_vec[c_leftglob])

    perm_visc = perm / visc

    # perm = dist[i][0]/(dist[i][1]/perm_vec[c_rightglob] + dist[i][2]/perm_vec[c_leftglob])
    # visc = dist[i][0]/(dist[i][1]/visc_vec[c_rightglob] + dist[i][2]/visc_vec[c_leftglob])

    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      VL = Pbordnode[i_1] * Icell[c_left] * perm_visc
      value_left = -1. * VL * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

      VR = Pbordnode[i_1] * Icell[c_right] * perm_visc
      value_right = VR * face_param4[i] / cell_volume[c_right]
      rhs[c_rightglob] += value_right

    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      VL = Pbordnode[i_2] * Icell[c_left] * perm_visc
      value_left = -1. * VL * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

      VR = Pbordnode[i_2] * Icell[c_right] * perm_visc
      value_right = VR * face_param2[i] / cell_volume[c_right]
      rhs[c_rightglob] += value_right

  for i in d_halofaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    perm = perm_vec[c_leftglob]
    visc = visc_vec[c_leftglob]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      VL = Pbordnode[i_1] * Icell[c_left] * (perm / visc)
      value_left = -1. * VL * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      VR = Pbordnode[i_2] * Icell[c_left] * (perm / visc)
      value_left = -1. * VR * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

  for i in dirichletfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    perm = perm_vec[c_leftglob]
    visc = visc_vec[c_leftglob]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    if node_ghostid[i_1, -1] > 0:
      VL = Pbordnode[i_1] * Icell[c_left] * (perm / visc)
      value_left = -1. * VL * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    if node_ghostid[i_2, -1] > 0:
      VL = Pbordnode[i_2] * Icell[c_left] * (perm / visc)
      value_left = -1. * VL * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    V_K = Pbordface[i] * Icell[c_left] * (perm / visc)
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs[c_leftglob] += value

  for i in neumannNHfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    perm = perm_vec[c_left]
    visc = visc_vec[c_left]
    rhs[c_leftglob] -= 1 * Icell[c_left] * (perm / visc) * cst * (np.sqrt(normalf[i][0] ** 2 + normalf[i][1] ** 2)) / \
                       cell_volume[c_left]



def _compute_P_gradient_2d_FV4():
  pass

###########################################################################
def _get_rhs_glob_3d(face_cellid: 'int[:,:]', faces: 'int[:,:]', node_oldname: 'int[:]',
                    cell_volume: 'float[:]', node_ghostid: 'int[:,:]', cell_loctoglob: 'int[:]', face_param1: 'float[:]',
                    face_param2: 'float[:]',
                    face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'int[:]', matrixinnerfaces: 'int[:]',
                    d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):

  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=faces.dtype)

  for i in matrixinnerfaces:

    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    nodes[0:3] = faces[i][0:3]
    nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      nodes[3] = faces[i][3]

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
    parameters[3] = -1. * face_param2[i]

    c_right = face_cellid[i][1]
    c_rightglob = cell_loctoglob[c_right]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, node_oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs[c_leftglob] += value_left

        value_right = V * parameters[cmpt] / cell_volume[c_right]
        rhs[c_rightglob] += value_right

      cmpt = cmpt + 1

  for i in d_halofaces:

    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    nodes[0:3] = faces[i][0:3]
    nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      nodes[3] = faces[i][3]

    parameters[0] = face_param1[i]
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i]
    parameters[3] = -1. * face_param2[i]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, node_oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs[c_leftglob] += value_left
      cmpt = cmpt + 1

  for i in dirichletfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    nodes[0:3] = faces[i][0:3]
    nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      nodes[3] = faces[i][3]

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
    parameters[3] = -1. * face_param2[i]

    cmpt = 0
    for nod in nodes:
      if node_ghostid[nod, -1] > 0:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs[c_leftglob] += value_left

      cmpt = cmpt + 1

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs[c_leftglob] += value



def _get_rhs_loc_3d(face_cellid: 'int[:,:]', faces: 'int[:,:]', node_oldname: 'int[:]',
                   cell_volume: 'float[:]', node_ghostid: 'int[:,:]', cell_loctoglob: 'int[:]', face_param1: 'float[:]',
                   face_param2: 'float[:]',
                   face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                   rhs_loc: 'float[:]',
                   BCdirichlet: 'int[:]', matrixinnerfaces: 'int[:]',
                   d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):

  rhs_loc[:] = 0.
  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=faces.dtype)

  for i in matrixinnerfaces:

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    nodes[0:3] = faces[i][0:3]
    nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      nodes[3] = faces[i][3]

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
    parameters[3] = -1. * face_param2[i]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, node_oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs_loc[c_left] += value_left

        value_right = V * parameters[cmpt] / cell_volume[c_right]
        rhs_loc[c_right] += value_right

      cmpt = cmpt + 1

  for i in d_halofaces:

    c_left = face_cellid[i][0]

    nodes[0:3] = faces[i][0:3]
    nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      nodes[3] = faces[i][3]

    parameters[0] = face_param1[i]
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i]
    parameters[3] = -1. * face_param2[i]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, node_oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs_loc[c_left] += value_left
      cmpt = cmpt + 1

  for i in dirichletfaces:
    c_left = face_cellid[i][0]

    nodes[0:3] = faces[i][0:3]
    nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      nodes[3] = faces[i][3]

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
    parameters[3] = -1. * face_param2[i]

    cmpt = 0
    for nod in nodes:
      if node_ghostid[nod, -1] > 0:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs_loc[c_left] += value_left

      cmpt += 1

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs_loc[c_left] += value


def _compute_P_gradient_3d_diamond(val_c: 'float[:]', v_ghost: 'float[:]', v_halo: 'float[:]', v_node: 'float[:]',
                                  face_cellid: 'int[:,:]',
                                  faces: 'int[:,:]', face_haloid: 'int[:]', node_oldname: 'int[:]', face_air_diamond: 'float[:]',
                                  face_f1: 'float[:,:]', face_f2: 'float[:,:]',
                                  face_f3: 'float[:,:]', face_f4: 'float[:,:]', face_normal: 'float[:,:]', cell_shift: 'float[:,:]',
                                  Pbordnode: 'float[:]',
                                  Pbordface: 'float[:]',
                                  Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]',
                                  BCdirichlet: 'int[:]', d_innerfaces: 'int[:]',
                                  d_halofaces: 'int[:]', neumannfaces: 'int[:]', dirichletfaces: 'int[:]',
                                  d_periodicboundaryfaces: 'int[:]'):

  for i in d_innerfaces:

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = v_node[i_1]
    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      V_A = Pbordnode[i_1]
    V_B = v_node[i_2]
    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      V_B = Pbordnode[i_2]
    V_C = v_node[i_3]
    if _search_element(BCdirichlet, node_oldname[i_3]) == 1:
      V_C = Pbordnode[i_3]
    V_D = v_node[i_4]
    if _search_element(BCdirichlet, node_oldname[i_4]) == 1:
      V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_R = val_c[c_right]

    Px_face[i] = -1. * (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    Py_face[i] = -1. * (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    Pz_face[i] = -1. * (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for i in d_periodicboundaryfaces:

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = v_node[i_1]
    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      V_A = Pbordnode[i_1]
    V_B = v_node[i_2]
    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      V_B = Pbordnode[i_2]
    V_C = v_node[i_3]
    if _search_element(BCdirichlet, node_oldname[i_3]) == 1:
      V_C = Pbordnode[i_3]
    V_D = v_node[i_4]
    if _search_element(BCdirichlet, node_oldname[i_4]) == 1:
      V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_R = val_c[c_right]

    Px_face[i] = -1. * (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    Py_face[i] = -1. * (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    Pz_face[i] = -1. * (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for i in neumannfaces:
    c_left = face_cellid[i][0]
    c_right = i

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = v_node[i_1]
    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      V_A = Pbordnode[i_1]
    V_B = v_node[i_2]
    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      V_B = Pbordnode[i_2]
    V_C = v_node[i_3]
    if _search_element(BCdirichlet, node_oldname[i_3]) == 1:
      V_C = Pbordnode[i_3]
    V_D = v_node[i_4]
    if _search_element(BCdirichlet, node_oldname[i_4]) == 1:
      V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_R = v_ghost[c_right]

    Px_face[i] = -1. * (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    Py_face[i] = -1. * (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    Pz_face[i] = -1. * (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for i in d_halofaces:
    c_left = face_cellid[i][0]
    c_right = face_haloid[i]

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = v_node[i_1]
    if _search_element(BCdirichlet, node_oldname[i_1]) == 1:
      V_A = Pbordnode[i_1]
    V_B = v_node[i_2]
    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      V_B = Pbordnode[i_2]
    V_C = v_node[i_3]
    if _search_element(BCdirichlet, node_oldname[i_3]) == 1:
      V_C = Pbordnode[i_3]
    V_D = v_node[i_4]
    if _search_element(BCdirichlet, node_oldname[i_4]) == 1:
      V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_R = v_halo[c_right]

    Px_face[i] = -1. * (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    Py_face[i] = -1. * (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    Pz_face[i] = -1. * (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

  for i in dirichletfaces:
    c_left = face_cellid[i][0]
    c_right = i

    i_1 = faces[i][0]
    i_2 = faces[i][1]
    i_3 = faces[i][2]
    i_4 = i_3
    if faces[i][-1] == 4:
      i_4 = faces[i][3]

    V_A = Pbordnode[i_1]
    V_B = Pbordnode[i_2]
    V_C = Pbordnode[i_3]
    V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_K = Pbordface[i]
    V_R = 2. * V_K - V_L

    Px_face[i] = -1. * (face_f1[i][0] * (V_A - V_C) + face_f2[i][0] * (V_B - V_D) + face_normal[i][0] * (V_R - V_L)) / face_air_diamond[i]
    Py_face[i] = -1. * (face_f1[i][1] * (V_A - V_C) + face_f2[i][1] * (V_B - V_D) + face_normal[i][1] * (V_R - V_L)) / face_air_diamond[i]
    Pz_face[i] = -1. * (face_f1[i][2] * (V_A - V_C) + face_f2[i][2] * (V_B - V_D) + face_normal[i][2] * (V_R - V_L)) / face_air_diamond[i]

def _get_triplet_3d(face_cellid: 'int[:,:]', faces: 'int[:,:]', nodes: 'float[:,:]', face_haloid: 'int[:]',
                   halo_halosext: 'int[:,:]', node_oldname: 'int[:]', cell_volume: 'float[:]',
                   node_cellid: 'int[:,:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]', node_haloid: 'int[:,:]',
                   node_periodicid: 'int[:,:]',
                   ghost_info_flt: 'float[:, :]', ghost_ext_info_flt: 'float[:, :]', ghost_info_int: 'int[:, :]', ghost_ext_info_int: 'int[:, :]', node_ghostid: 'int[:, :]', node_haloghostid: 'int[:, :]', face_air_diamond: 'float[:]',
                   node_lambda_x: 'float[:]', node_lambda_y: 'float[:]', node_lambda_z: 'float[:]', node_number: 'int[:]',
                   node_R_x: 'float[:]', node_R_y: 'float[:]',
                   node_R_z: 'float[:]', face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]', face_param4: 'float[:]',
                   cell_shift: 'float[:,:]',
                   nbelements: 'intc', cell_loctoglob: 'int[:]', BCdirichlet: 'int[:]', a_loc: 'float[:]',
                   irn_loc: 'int[:]', jcn_loc: 'int[:]',
                   matrixinnerfaces: 'int[:]', d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):

  parameters = np.zeros(4)
  face_nodes = np.zeros(4, dtype=faces.dtype)

  cmpt = 0

  for i in matrixinnerfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    face_nodes[0:3] = faces[i][0:3]
    face_nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      face_nodes[3] = faces[i][3]

    parameters[0] = face_param1[i]
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i]
    parameters[3] = -1. * face_param2[i]

    c_right = face_cellid[i][1]
    c_rightglob = cell_loctoglob[c_right]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1 * face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    cmptparam = 0
    for nod in face_nodes:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          center = cell_center[node_cellid[nod][j]]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          center = halo_centvol[node_haloid[nod][j]]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_periodicid[nod][-1]):
          if nodes[nod][3] == 11 or nodes[nod][3] == 22:
            center[0] = cell_center[node_periodicid[nod][j]][0] + cell_shift[node_periodicid[nod][j]][0]
            center[1] = cell_center[node_periodicid[nod][j]][1]
            center[2] = cell_center[node_periodicid[nod][j]][2]
          if nodes[nod][3] == 33 or nodes[nod][3] == 44:
            center[0] = cell_center[node_periodicid[nod][j]][0]
            center[1] = cell_center[node_periodicid[nod][j]][1] + cell_shift[node_periodicid[nod][j]][1]
            center[2] = cell_center[node_periodicid[nod][j]][2]
          if nodes[nod][3] == 55 or nodes[nod][3] == 66:
            center[0] = cell_center[node_periodicid[nod][j]][0]
            center[1] = cell_center[node_periodicid[nod][j]][1]
            center[2] = cell_center[node_periodicid[nod][j]][2] + cell_shift[node_periodicid[nod][j]][2]

          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_periodicid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[node_periodicid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          ghost_id = node_ghostid[nod][j]
          center = ghost_info_flt[ghost_id][0:3]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]

          index = ghost_info_int[ghost_id, 0]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[index]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[index]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          ghost_id = node_haloghostid[nod][j]
          center = ghost_ext_info_flt[ghost_id][0:3]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          index = ghost_ext_info_int[ghost_id, 0]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[index][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = halo_halosext[index][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
      cmptparam = cmptparam + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    # right cell------------------------------------------------------
    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_leftglob
    value = face_param3[i] / cell_volume[c_right]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_rightglob
    value = -1. * face_param3[i] / cell_volume[c_right]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

  for i in d_halofaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    face_nodes[0:3] = faces[i][0:3]
    face_nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      face_nodes[3] = faces[i][3]

    parameters[0] = face_param1[i]
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i]
    parameters[3] = -1. * face_param2[i]

    c_rightglob = halo_halosext[face_haloid[i]][0]
    c_right = face_haloid[i]

    cmptparam = 0
    for nod in face_nodes:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          center = cell_center[node_cellid[nod][j]]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          center = halo_centvol[node_haloid[nod][j]]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          ghost_id = node_ghostid[nod][j]
          center = ghost_info_flt[ghost_id][0:3]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]

          index = ghost_info_int[ghost_id, 0]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[index]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          ghost_id = node_haloghostid[nod][j]
          center = ghost_ext_info_flt[ghost_id][0:3]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          zdiff = center[2] - nodes[nod][2]
          alpha = (1. + node_lambda_x[nod] * xdiff +
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          index = ghost_ext_info_int[ghost_id, 0]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[index][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
      cmptparam = cmptparam + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1 * face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

  for i in dirichletfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    parameters[0] = face_param1[i]
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i]
    parameters[3] = -1. * face_param2[i]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1 * face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * face_param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

def _compute_3dmatrix_size(faces: 'int[:,:]', node_cellid: 'int[:,:]', node_haloid: 'int[:,:]',
                          node_periodicid: 'int[:,:]',
                          node_ghostid: 'int[:,:]', node_haloghostid: 'int[:,:]', node_oldname: 'int[:]',
                          BCdirichlet: 'int[:]',
                          matrixinnerfaces: 'int[:]', d_halofaces: 'int[:]', dirichletfaces: 'int[:]'):

  cmpt = 0
  nodes = np.zeros(4, dtype=faces.dtype)

  for i in matrixinnerfaces:

    nodes[0:3] = faces[i][0:3]
    nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      nodes[3] = faces[i][3]

    cmpt = cmpt + 1

    for nod in nodes:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(node_periodicid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(node_haloid[nod, -1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

    cmpt = cmpt + 1
    # right cell------------------------------------------------------
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  for i in d_halofaces:

    nodes[0:3] = faces[i][0:3]
    nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      nodes[3] = faces[i][3]

    cmpt = cmpt + 1
    cmpt = cmpt + 1

    for nod in nodes:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          cmpt = cmpt + 1

        for j in range(node_ghostid[nod, -1]):
          cmpt = cmpt + 1

        for j in range(node_haloghostid[nod, -1]):
          cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          cmpt = cmpt + 1

  for i in dirichletfaces:
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  return cmpt


############################################################################
# NOTHING is compiled at import. Call setup(dim) once (uniformly on all MPI
# ranks) before using any kernel below; LinearSolver does this in __init__.
#   - agnostic kernels are compiled once;
#   - dimension-specific kernels are compiled only for the dimension(s) used.
# The nested helper _search_element is compiled (and rebound to the module
# global) before the kernels that call it, so numba can resolve njit->njit calls.
_agnostic_done = False
_dims_done = set()

def setup(dim):
  global _agnostic_done
  if not _agnostic_done:
    global _search_element  # nested helper first
    global convert_solution, rhs_value_dirichlet_node, rhs_value_dirichlet_face
    _search_element = compile(_search_element)
    convert_solution = compile(_convert_solution)
    rhs_value_dirichlet_node = compile(_rhs_value_dirichlet_node)
    rhs_value_dirichlet_face = compile(_rhs_value_dirichlet_face)
    _agnostic_done = True

  if dim not in _dims_done:
    if dim == 2:
      global compute_P_gradient_2d_diamond, compute_P_gradient_2d_FV4, get_triplet_2d
      global compute_2dmatrix_size, get_rhs_glob_2d, get_rhs_loc_2d
      global get_triplet_2d_with_contrib, get_rhs_glob_2d_with_contrib
      compute_P_gradient_2d_diamond = compile(_compute_P_gradient_2d_diamond)
      compute_P_gradient_2d_FV4 = compile(_compute_P_gradient_2d_FV4)
      get_triplet_2d = compile(_get_triplet_2d)
      compute_2dmatrix_size = compile(_compute_2dmatrix_size)
      get_rhs_glob_2d = compile(_get_rhs_glob_2d)
      get_rhs_loc_2d = compile(_get_rhs_loc_2d)
      get_triplet_2d_with_contrib = compile(_get_triplet_2d_with_contrib)
      get_rhs_glob_2d_with_contrib = compile(_get_rhs_glob_2d_with_contrib)
    elif dim == 3:
      global compute_P_gradient_3d_diamond, get_triplet_3d, compute_3dmatrix_size
      global get_rhs_glob_3d, get_rhs_loc_3d
      compute_P_gradient_3d_diamond = compile(_compute_P_gradient_3d_diamond)
      get_triplet_3d = compile(_get_triplet_3d)
      compute_3dmatrix_size = compile(_compute_3dmatrix_size)
      get_rhs_glob_3d = compile(_get_rhs_glob_3d)
      get_rhs_loc_3d = compile(_get_rhs_loc_3d)
    else:
      raise ValueError(f"Unsupported dimension: {dim}")
    _dims_done.add(dim)
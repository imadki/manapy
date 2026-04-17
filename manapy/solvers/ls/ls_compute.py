from manapy.backends.compile_fun import compile
import numpy as np

### UTILS
def _convert_solution(x1: 'float[:]', x1converted: 'float[:]', cell_tc: 'int32[:]', b0Size: 'int32'):
  for i in range(b0Size):
    x1converted[i] = x1[cell_tc[i]]

def _search_element(a: 'int32[:]', target_value: 'int32'):
  find = 0
  for val in a:
    if val == target_value:
      find = 1
      break
  return find


def _rhs_value_dirichlet_node(Pbordnode: 'float[:]', nodes: 'uint32[:]', value: 'float[:]'):
  for i in nodes:
    Pbordnode[i] = value[i]


def _rhs_value_dirichlet_face(Pbordface: 'float[:]', faces: 'uint32[:]', value: 'float[:]'):
  for i in faces:
    Pbordface[i] = value[i]

def _compute_P_gradient_2d_diamond(P_c: 'float[:]', P_ghost: 'float[:]', P_halo: 'float[:]', P_node: 'float[:]',
                                  face_cellid: 'int32[:,:]',
                                  faces: 'int32[:,:]', face_ghostcenter: 'float[:,:]', face_haloid: 'int32[:]',
                                  cell_center: 'float[:,:]',
                                  halo_centvol: 'float[:,:]', node_oldname: 'uint32[:]', face_air_diamond: 'float[:]',
                                  face_f1: 'float[:,:]', face_f2: 'float[:,:]',
                                  face_f3: 'float[:,:]', face_f4: 'float[:,:]', face_normal: 'float[:,:]', cell_shift: 'float[:,:]',
                                  Pbordnode: 'float[:]',
                                  Pbordface: 'float[:]',
                                  Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]',
                                  BCdirichlet: 'uint32[:]', d_innerfaces: 'uint32[:]',
                                  d_halofaces: 'uint32[:]', neumannfaces: 'uint32[:]', dirichletfaces: 'uint32[:]',
                                  d_periodicboundaryfaces: 'uint32[:]'):

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

def _get_triplet_2d(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', nodes: 'float[:,:]', face_haloid: 'int32[:]',
                   halo_halosext: 'int32[:,:]', node_oldname: 'uint32[:]', cell_volume: 'float[:]',
                   node_cellid: 'int32[:,:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]', node_haloid: 'int32[:,:]',
                   node_periodicid: 'int32[:,:]',
                   node_ghostcenter: 'float[:,:,:]', node_ghostcenter_info: 'int[:, :, :]', node_haloghostcenter: 'float[:,:,:]', node_haloghostcenter_info: 'int[:, :, :]', face_air_diamond: 'float[:]',
                   node_lambda_x: 'float[:]', node_lambda_y: 'float[:]', node_lambda_z: 'float[:]', node_number: 'uint32[:]',
                   node_R_x: 'float[:]', node_R_y: 'float[:]',
                   node_R_z: 'float[:]', face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]', face_param4: 'float[:]',
                   cell_shift: 'float[:,:]',
                   nbelements: 'int32', cell_loctoglob: 'int32[:]', BCdirichlet: 'uint32[:]', a_loc: 'float[:]',
                   irn_loc: 'int32[:]', jcn_loc: 'int32[:]',
                   matrixinnerfaces: 'uint32[:]', d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  center = np.zeros(2)
  parameters = np.zeros(2)
  cmpt = 0
  one_rank = (node_haloghostcenter_info.shape[0] == 0)

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
          alpha = (1. + node_lambda_x[nod] * xdiff + \
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

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            center[:] = node_ghostcenter[nod][j][0:2]
            xdiff = center[0] - nodes[nod][0]
            ydiff = center[1] - nodes[nod][1]
            alpha = (1. + node_lambda_x[nod] * xdiff + \
                     node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            index = node_ghostcenter_info[nod][j][0]
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

        if not one_rank:
          for j in range(len(node_haloghostcenter_info[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
              center[:] = node_haloghostcenter[nod][j][0:2]
              xdiff = center[0] - nodes[nod][0]
              ydiff = center[1] - nodes[nod][1]
              alpha = (1. + node_lambda_x[nod] * xdiff + \
                       node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
              index = node_ghostcenter_info[nod][j][0]
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
          alpha = (1. + node_lambda_x[nod] * xdiff + \
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

        if not one_rank:
          for j in range(node_haloid[nod][-1]):
            center[:] = halo_centvol[node_haloid[nod][j]][0:2]
            xdiff = center[0] - nodes[nod][0]
            ydiff = center[1] - nodes[nod][1]
            alpha = (1. + node_lambda_x[nod] * xdiff + \
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
          alpha = (1. + node_lambda_x[nod] * xdiff + \
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            center[:] = node_ghostcenter[nod][j][0:2]
            xdiff = center[0] - nodes[nod][0]
            ydiff = center[1] - nodes[nod][1]
            alpha = (1. + node_lambda_x[nod] * xdiff + \
                     node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            index = node_ghostcenter_info[nod][j][2]
            value = alpha / cell_volume[c_left] * parameters[cmptparam]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = cell_loctoglob[index]
            a_loc[cmpt] = value
            cmpt = cmpt + 1

        if not one_rank:
          for j in range(len(node_haloghostcenter_info[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
              center[:] = node_haloghostcenter[nod][j][0:2]
              xdiff = center[0] - nodes[nod][0]
              ydiff = center[1] - nodes[nod][1]
              alpha = (1. + node_lambda_x[nod] * xdiff + \
                       node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
              index = node_haloghostcenter_info[nod][j][0]
              value = alpha / cell_volume[c_left] * parameters[cmptparam]
              irn_loc[cmpt] = c_leftglob
              jcn_loc[cmpt] = halo_halosext[index][0]
              a_loc[cmpt] = value
              cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          center[:] = halo_centvol[node_haloid[nod][j]][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff + \
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

def _compute_2dmatrix_size(faces: 'int32[:,:]', face_haloid: 'int32[:]', node_cellid: 'int32[:,:]', node_haloid: 'int32[:,:]',
                          node_periodicid: 'int32[:,:]',
                          node_ghostcenter_info: 'int[:,:,:]', node_haloghostcenter_info: 'int[:, :, :]', node_oldname: 'uint32[:]',
                          BCdirichlet: 'uint32[:]',
                          matrixinnerfaces: 'uint32[:]', d_halofaces: 'uint32[:]',
                          dirichletfaces: 'uint32[:]'):

  cmpt = 0
  one_rank = (node_haloghostcenter_info.shape[0] == 0)

  for i in matrixinnerfaces:
    cmpt = cmpt + 1

    for nod in faces[i][:faces[i][-1]]:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:  # and search_element(BCneumannNH, node_oldname[nod]) == 0:
        # if nodes[nod][3] not in BCdirichlet:
        for j in range(node_cellid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            cmpt = cmpt + 1
            # right cell-----------------------------------
            cmpt = cmpt + 1

        if not one_rank:
          for j in range(len(node_haloghostcenter_info[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
              cmpt = cmpt + 1
              # right cell-----------------------------------
              cmpt = cmpt + 1

        for j in range(node_periodicid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        if not one_rank:
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

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            cmpt = cmpt + 1

        if not one_rank:
          for j in range(len(node_haloghostcenter_info[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
              cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          cmpt = cmpt + 1

  for i in dirichletfaces:
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  return cmpt

def _get_rhs_glob_2d(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', node_oldname: 'uint32[:]',
                    cell_volume: 'float[:]', node_ghostcenter_info: 'int[:,:,:]', cell_loctoglob: 'int32[:]', face_param1: 'float[:]',
                    face_param2: 'float[:]',
                    face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'uint32[:]', face_ghostcenter: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                    d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

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

    if node_ghostcenter_info[i_1][0][2] != -1:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    if node_ghostcenter_info[i_2][0][2] != -1:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs[c_leftglob] += value

def _get_rhs_loc_2d(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', node_oldname: 'uint32[:]',
                   cell_volume: 'float[:]', node_ghostcenter_info: 'int[:,:,:]', cell_loctoglob: 'int32[:]', face_param1: 'float[:]',
                   face_param2: 'float[:]',
                   face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                   rhs_loc: 'float[:]',
                   BCdirichlet: 'uint32[:]', face_ghostcenter: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                   d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):


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

    if node_ghostcenter_info[i_1][0][0] != -1:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs_loc[c_left] += value_left

    if node_ghostcenter_info[i_2][0][0] != -1:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs_loc[c_left] += value_left

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs_loc[c_left] += value

def _get_rhs_glob_2d(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', node_oldname: 'uint32[:]',
                    cell_volume: 'float[:]', node_ghostcenter_info: 'int[:,:,:]', cell_loctoglob: 'int32[:]', face_param1: 'float[:]',
                    face_param2: 'float[:]',
                    face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'uint32[:]', face_ghostcenter: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                    d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

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

    if node_ghostcenter_info[i_1][0][2] != -1:
      V = Pbordnode[i_1]
      value_left = -1. * V * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    if node_ghostcenter_info[i_2][0][2] != -1:
      V = Pbordnode[i_2]
      value_left = -1. * V * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs[c_leftglob] += value

def _get_triplet_2d_with_contrib(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', cell_faceid: 'int32[:,:]',
                                nodes: 'float[:,:]', face_haloid: 'int32[:]',
                                halo_halosext: 'int32[:,:]', node_oldname: 'uint32[:]', cell_volume: 'float[:]',
                                node_cellid: 'int32[:,:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                node_haloid: 'int32[:,:]',
                                node_periodicid: 'int32[:,:]',
                                node_ghostcenter: 'float[:,:,:]', node_ghostcenter_info: 'int[:, :, :]', node_haloghostcenter: 'float[:,:,:]', node_haloghostcenter_info: 'int[:, :, :]', face_air_diamond: 'float[:]',
                                node_lambda_x: 'float[:]', node_lambda_y: 'float[:]', node_number: 'uint32[:]', node_R_x: 'float[:]',
                                node_R_y: 'float[:]', param1: 'float[:]',
                                param2: 'float[:]', param3: 'float[:]', param4: 'float[:]', cell_shift: 'float[:,:]',
                                nbelements: 'int32', cell_loctoglob: 'int32[:]',
                                BCdirichlet: 'uint32[:]', a_loc: 'float[:]', irn_loc: 'int32[:]', jcn_loc: 'int32[:]',
                                matrixinnerfaces: 'uint32[:]', d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]',
                                Icell: 'float[:]',  # Ihalo:'float[:]', Ihaloghost:'float[:]',
                                alpha_P: 'float', perm_vec: 'float[:]',
                                visc_vec: 'float[:]', BCneumannNH: 'uint32[:]', dist: 'float[:]'):


  center = np.zeros(2)
  parameters = np.zeros(2)
  cmpt = 0
  one_rank = (node_haloghostcenter_info.shape[0] == 0)

  for i in matrixinnerfaces:
    nbfL = cell_faceid[face_cellid[i][0]][-1]
    nbfR = cell_faceid[face_cellid[i][1]][-1]
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    parameters[0] = param4[i];
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
    a_loc[cmpt] = value * Icell[c_left] * (perm_visc) + (1 / nbfL) * cell_volume[c_left] * alpha_P * (1 - Icell[c_left])
    cmpt = cmpt + 1

    cmptparam = 0
    for nod in faces[i][:faces[i][-1]]:
      if _search_element(BCdirichlet, node_oldname[nod]) == 0:  # and search_element(BCneumannNH, node_oldname[nod]) == 0:
        for j in range(node_cellid[nod][-1]):
          center[:] = cell_center[node_cellid[nod][j]][0:2]
          xdiff = center[0] - nodes[nod][0]
          ydiff = center[1] - nodes[nod][1]
          alpha = (1. + node_lambda_x[nod] * xdiff + \
                   node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value * Icell[c_left] * (perm_visc)
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = cell_loctoglob[node_cellid[nod][j]]
          a_loc[cmpt] = value * Icell[c_right] * (perm_visc)
          cmpt = cmpt + 1

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            center[:] = node_ghostcenter[nod][j][0:2]
            xdiff = center[0] - nodes[nod][0]
            ydiff = center[1] - nodes[nod][1]
            alpha = (1. + node_lambda_x[nod] * xdiff + \
                     node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            index = node_ghostcenter_info[nod][j][0]
            value = alpha / cell_volume[c_left] * parameters[cmptparam]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = cell_loctoglob[index]
            a_loc[cmpt] = value * Icell[c_left] * (perm_visc)
            cmpt = cmpt + 1
            # right cell-----------------------------------
            value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = cell_loctoglob[index]
            a_loc[cmpt] = value * Icell[c_right] * (perm_visc)
            cmpt = cmpt + 1

        if not one_rank:
          for j in range(len(node_haloghostcenter_info[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
              center[:] = node_haloghostcenter[nod][j][0:2]
              xdiff = center[0] - nodes[nod][0]
              ydiff = center[1] - nodes[nod][1]
              alpha = (1. + node_lambda_x[nod] * xdiff + \
                       node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
              index = int(node_haloghostcenter[nod][j][2])
              #                        cell  = int(node_haloghostcenter[nod][j][-1])
              value = alpha / cell_volume[c_left] * parameters[cmptparam]
              irn_loc[cmpt] = c_leftglob
              jcn_loc[cmpt] = halo_halosext[index][0]
              a_loc[cmpt] = value * Icell[c_left] * (perm_visc)
              cmpt = cmpt + 1
              # right cell-----------------------------------
              value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
              irn_loc[cmpt] = c_rightglob
              jcn_loc[cmpt] = halo_halosext[index][0]
              # TODO
              a_loc[cmpt] = value * Icell[c_right] * (
                perm_visc)  # value*Ihaloghost[int(node_haloghostcenter[nod][j][-1])]*(perm/visc)
              cmpt = cmpt + 1

          for j in range(node_haloid[nod][-1]):
            center[:] = halo_centvol[node_haloid[nod][j]][0:2]
            xdiff = center[0] - nodes[nod][0]
            ydiff = center[1] - nodes[nod][1]
            alpha = (1. + node_lambda_x[nod] * xdiff + \
                     node_lambda_y[nod] * ydiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + node_lambda_y[nod] * node_R_y[nod])
            value = alpha / cell_volume[c_left] * parameters[cmptparam]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
            a_loc[cmpt] = value * Icell[c_left] * (perm_visc)
            cmpt = cmpt + 1
            # right cell-----------------------------------
            value = -1. * alpha / cell_volume[c_right] * parameters[cmptparam]
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
            a_loc[cmpt] = value * Icell[c_right] * (perm_visc)  # value*Ihalo[node_haloid[nod][j]]*(perm/visc)
            cmpt = cmpt + 1

      cmptparam = +1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = param3[i] / cell_volume[c_left]
    a_loc[cmpt] = value * Icell[c_left] * (perm_visc)
    cmpt = cmpt + 1

    # right cell------------------------------------------------------
    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * param1[i] / cell_volume[c_right]
    a_loc[cmpt] = value * Icell[c_right] * (perm_visc)
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_rightglob
    value = -1. * param3[i] / cell_volume[c_right]
    a_loc[cmpt] = value * Icell[c_right] * (perm_visc) + (1 / nbfR) * cell_volume[c_right] * alpha_P * (1 - Icell[c_right])
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


def _get_rhs_glob_2d_with_contrib(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', node_oldname: 'uint32[:]',
                                 cell_volume: 'float[:]', node_ghostcenter: 'float[:,:,:]', node_ghostcenter_info: 'int[:, :, :]', cell_loctoglob: 'int32[:]',
                                 face_param1: 'float[:]', face_param2: 'float[:]',
                                 face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]',
                                 Pbordface: 'float[:]', rhs: 'float[:]',
                                 BCdirichlet: 'uint32[:]', centergf: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                                 d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]', neumannNHfaces: 'uint32[:]',
                                 Icell: 'float[:]', Inode: 'float[:]', perm_vec: 'float[:]', visc_vec: 'float[:]',
                                 cst: 'float', mesuref: 'float[:]', normalf: 'float[:,:]', dist: 'float[:]'):

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
      VL = Pbordnode[i_1] * Icell[c_left] * (perm_visc)
      value_left = -1. * VL * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

      VR = Pbordnode[i_1] * Icell[c_right] * (perm_visc)
      value_right = VR * face_param4[i] / cell_volume[c_right]
      rhs[c_rightglob] += value_right

    if _search_element(BCdirichlet, node_oldname[i_2]) == 1:
      VL = Pbordnode[i_2] * Icell[c_left] * (perm_visc)
      value_left = -1. * VL * face_param2[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

      VR = Pbordnode[i_2] * Icell[c_right] * (perm_visc)
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

    if node_ghostcenter_info[i_1][0][0] != -1:
      VL = Pbordnode[i_1] * Icell[c_left] * (perm / visc)
      value_left = -1. * VL * face_param4[i] / cell_volume[c_left]
      rhs[c_leftglob] += value_left

    if node_ghostcenter_info[i_2][0][0] != -1:
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
def _get_rhs_glob_3d(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', node_oldname: 'uint32[:]',
                    cell_volume: 'float[:]', node_ghostcenter_info: 'int[:,:,:]', cell_loctoglob: 'int32[:]', face_param1: 'float[:]',
                    face_param2: 'float[:]',
                    face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'uint32[:]', face_ghostcenter: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                    d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=np.int32)

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

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
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
      if node_ghostcenter_info[nod][0][3] != -1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs[c_leftglob] += value_left

      cmpt = cmpt + 1

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs[c_leftglob] += value



def _get_rhs_loc_3d(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', node_oldname: 'uint32[:]',
                   cell_volume: 'float[:]', node_ghostcenter_info: 'int[:,:,:]', cell_loctoglob: 'int32[:]', face_param1: 'float[:]',
                   face_param2: 'float[:]',
                   face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                   rhs_loc: 'float[:]',
                   BCdirichlet: 'uint32[:]', face_ghostcenter: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                   d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  rhs_loc[:] = 0.
  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=np.int32)

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
      if node_ghostcenter_info[nod][0][3] != -1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs_loc[c_left] += value_left

      cmpt += 1

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs_loc[c_left] += value



def _get_rhs_glob_3d(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', node_oldname: 'uint32[:]',
                    cell_volume: 'float[:]', node_ghostcenter_info: 'int[:,:,:]', cell_loctoglob: 'int32[:]', face_param1: 'float[:]',
                    face_param2: 'float[:]',
                    face_param3: 'float[:]', face_param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'uint32[:]', face_ghostcenter: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                    d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=np.int32)

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

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
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
      if node_ghostcenter_info[nod][0][3] != -1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / cell_volume[c_left]
        rhs[c_leftglob] += value_left

      cmpt = cmpt + 1

    V_K = Pbordface[i]
    value = -2. * face_param3[i] / cell_volume[c_left] * V_K
    rhs[c_leftglob] += value


def _compute_P_gradient_3d_diamond(val_c: 'float[:]', v_ghost: 'float[:]', v_halo: 'float[:]', v_node: 'float[:]',
                                  face_cellid: 'int32[:,:]',
                                  faces: 'int32[:,:]', face_ghostcenter: 'float[:,:]', face_haloid: 'int32[:]',
                                  cell_center: 'float[:,:]',
                                  halo_centvol: 'float[:,:]', node_oldname: 'uint32[:]', face_air_diamond: 'float[:]',
                                  face_f1: 'float[:,:]', face_f2: 'float[:,:]',
                                  face_f3: 'float[:,:]', face_f4: 'float[:,:]', face_normal: 'float[:,:]', cell_shift: 'float[:,:]',
                                  Pbordnode: 'float[:]',
                                  Pbordface: 'float[:]',
                                  Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]',
                                  BCdirichlet: 'uint32[:]', d_innerfaces: 'uint32[:]',
                                  d_halofaces: 'uint32[:]', neumannfaces: 'uint32[:]', dirichletfaces: 'uint32[:]',
                                  d_periodicboundaryfaces: 'uint32[:]'):

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

def _get_triplet_3d(face_cellid: 'int32[:,:]', faces: 'int32[:,:]', nodes: 'float[:,:]', face_haloid: 'int32[:]',
                   halo_halosext: 'int32[:,:]', node_oldname: 'uint32[:]', cell_volume: 'float[:]',
                   node_cellid: 'int32[:,:]', cell_center: 'float[:,:]', halo_centvol: 'float[:,:]', node_haloid: 'int32[:,:]',
                   node_periodicid: 'int32[:,:]',
                   node_ghostcenter: 'float[:,:,:]', node_ghostcenter_info: 'int[:,:,:]', node_haloghostcenter: 'float[:,:,:]', node_haloghostcenter_info: 'int[:, :, :]', face_air_diamond: 'float[:]',
                   node_lambda_x: 'float[:]', node_lambda_y: 'float[:]', node_lambda_z: 'float[:]', node_number: 'uint32[:]',
                   node_R_x: 'float[:]', node_R_y: 'float[:]',
                   node_R_z: 'float[:]', face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]', face_param4: 'float[:]',
                   cell_shift: 'float[:,:]',
                   nbelements: 'intc', cell_loctoglob: 'int32[:]', BCdirichlet: 'uint32[:]', a_loc: 'float[:]',
                   irn_loc: 'int32[:]', jcn_loc: 'int32[:]',
                   matrixinnerfaces: 'uint32[:]', d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  parameters = np.zeros(4)
  face_nodes = np.zeros(4, dtype=np.int32)
  one_rank = (node_haloghostcenter_info.shape[0] == 0)

  cmpt = 0

  for i in matrixinnerfaces:
    c_left = face_cellid[i][0]
    c_leftglob = cell_loctoglob[c_left]

    face_nodes[0:3] = faces[i][0:3]
    face_nodes[3] = faces[i][2]
    if faces[i][-1] == 4:
      face_nodes[3] = faces[i][3]

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
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
          alpha = (1. + node_lambda_x[nod] * xdiff + \
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
          alpha = (1. + node_lambda_x[nod] * xdiff + \
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
          alpha = (1. + node_lambda_x[nod] * xdiff + \
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

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            center = node_ghostcenter[nod][j][0:3]
            xdiff = center[0] - nodes[nod][0]
            ydiff = center[1] - nodes[nod][1]
            zdiff = center[2] - nodes[nod][2]
            alpha = (1. + node_lambda_x[nod] * xdiff + \
                     node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                       node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                         nod])
            value = alpha / cell_volume[c_left] * parameters[cmptparam]

            index = node_ghostcenter_info[nod][j][0]
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

        if not one_rank:
          for j in range(len(node_haloghostcenter[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
              center = node_haloghostcenter[nod][j][0:3]
              xdiff = center[0] - nodes[nod][0]
              ydiff = center[1] - nodes[nod][1]
              zdiff = center[2] - nodes[nod][2]
              alpha = (1. + node_lambda_x[nod] * xdiff + \
                       node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                         node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                           nod])
              value = alpha / cell_volume[c_left] * parameters[cmptparam]
              index = node_haloghostcenter_info[nod][j][0]
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

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
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
          alpha = (1. + node_lambda_x[nod] * xdiff + \
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
          alpha = (1. + node_lambda_x[nod] * xdiff + \
                   node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                     node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                       nod])
          value = alpha / cell_volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = halo_halosext[node_haloid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            center = node_ghostcenter[nod][j][0:3]
            xdiff = center[0] - nodes[nod][0]
            ydiff = center[1] - nodes[nod][1]
            zdiff = center[2] - nodes[nod][2]
            alpha = (1. + node_lambda_x[nod] * xdiff + \
                     node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                       node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                         nod])
            value = alpha / cell_volume[c_left] * parameters[cmptparam]

            index = node_ghostcenter_info[nod][j][3]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = cell_loctoglob[index]
            a_loc[cmpt] = value
            cmpt = cmpt + 1

        if not one_rank:
          for j in range(len(node_haloghostcenter[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
              center = node_haloghostcenter[nod][j][0:3]
              xdiff = center[0] - nodes[nod][0]
              ydiff = center[1] - nodes[nod][1]
              zdiff = center[2] - nodes[nod][2]
              alpha = (1. + node_lambda_x[nod] * xdiff + \
                       node_lambda_y[nod] * ydiff + node_lambda_z[nod] * zdiff) / (node_number[nod] + node_lambda_x[nod] * node_R_x[nod] + \
                                                                         node_lambda_y[nod] * node_R_y[nod] + node_lambda_z[nod] * node_R_z[
                                                                           nod])
              value = alpha / cell_volume[c_left] * parameters[cmptparam]
              index = node_haloghostcenter_info[nod][j][0]
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

    parameters[0] = face_param1[i];
    parameters[1] = face_param2[i]
    parameters[2] = -1. * face_param1[i];
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

def _compute_3dmatrix_size(faces: 'int32[:,:]', face_haloid: 'int32[:]', node_cellid: 'int32[:,:]', node_haloid: 'int32[:,:]',
                          node_periodicid: 'int32[:,:]',
                          node_ghostcenter_info: 'int[:,:,:]', node_haloghostcenter_info: 'float[:,:,:]', node_oldname: 'uint32[:]',
                          BCdirichlet: 'uint32[:]',
                          matrixinnerfaces: 'uint32[:]', d_halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  cmpt = 0
  nodes = np.zeros(4, dtype=np.int32)
  one_rank = (node_haloghostcenter_info.shape[0] == 0)

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

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            cmpt = cmpt + 1
            # right cell-----------------------------------
            cmpt = cmpt + 1

        if not one_rank:
          for j in range(len(node_haloghostcenter_info[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
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

        for j in range(len(node_ghostcenter_info[nod])):
          if node_ghostcenter_info[nod][j][-1] != -1:
            cmpt = cmpt + 1

        if not one_rank:
          for j in range(len(node_haloghostcenter_info[nod])):
            if node_haloghostcenter_info[nod][j][-1] != -1:
              cmpt = cmpt + 1

        for j in range(node_haloid[nod][-1]):
          cmpt = cmpt + 1

  for i in dirichletfaces:
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  return cmpt


############################################################################
# Private
_search_element = compile(_search_element)

# Public
convert_solution = compile(_convert_solution)
rhs_value_dirichlet_node = compile(_rhs_value_dirichlet_node)
rhs_value_dirichlet_face = compile(_rhs_value_dirichlet_face)
compute_P_gradient_2d_diamond = compile(_compute_P_gradient_2d_diamond)
compute_P_gradient_2d_FV4 = compile(_compute_P_gradient_2d_FV4)
get_triplet_2d = compile(_get_triplet_2d)
compute_2dmatrix_size = compile(_compute_2dmatrix_size)
compute_P_gradient_3d_diamond = compile(_compute_P_gradient_3d_diamond)
get_triplet_3d = compile(_get_triplet_3d)
compute_3dmatrix_size = compile(_compute_3dmatrix_size)
get_rhs_glob_2d = compile(_get_rhs_glob_2d)
get_rhs_glob_3d = compile(_get_rhs_glob_3d)
get_rhs_loc_2d = compile(_get_rhs_loc_2d)
get_rhs_loc_3d = compile(_get_rhs_loc_3d)
get_triplet_2d_with_contrib = compile(_get_triplet_2d_with_contrib)
get_rhs_glob_2d_with_contrib = compile(_get_rhs_glob_2d_with_contrib)
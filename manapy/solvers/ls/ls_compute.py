from manapy.backends.compile_fun import compile
import numpy as np

### UTILS
def _convert_solution(x1: 'float[:]', x1converted: 'float[:]', tc: 'int32[:]', b0Size: 'int32'):
  for i in range(b0Size):
    x1converted[i] = x1[tc[i]]

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
                                  cellidf: 'int32[:,:]',
                                  nodeidf: 'int32[:,:]', centergf: 'float[:,:]', halofid: 'int32[:]',
                                  centerc: 'float[:,:]',
                                  centerh: 'float[:,:]', oldname: 'uint32[:]', airDiamond: 'float[:]',
                                  f_1: 'float[:,:]', f_2: 'float[:,:]',
                                  f_3: 'float[:,:]', f_4: 'float[:,:]', normalf: 'float[:,:]', shift: 'float[:,:]',
                                  Pbordnode: 'float[:]',
                                  Pbordface: 'float[:]',
                                  Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]',
                                  BCdirichlet: 'uint32[:]', innerfaces: 'uint32[:]',
                                  halofaces: 'uint32[:]', neumannfaces: 'uint32[:]', dirichletfaces: 'uint32[:]',
                                  periodicfaces: 'uint32[:]'):

  for i in innerfaces:

    c_left = cellidf[i][0]
    c_right = cellidf[i][1]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = P_node[i_1]
    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      vi1 = Pbordnode[i_1]
    vi2 = P_node[i_2]
    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      vi2 = Pbordnode[i_2]

    vv1 = P_c[c_left]
    vv2 = P_c[c_right]

    Px_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    Py_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

  for i in periodicfaces:

    c_left = cellidf[i][0]
    c_right = cellidf[i][1]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = P_node[i_1]
    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      vi1 = Pbordnode[i_1]
    vi2 = P_node[i_2]
    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      vi2 = Pbordnode[i_2]

    vv1 = P_c[c_left]
    vv2 = P_c[c_right]

    Px_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    Py_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

  for i in neumannfaces:

    c_left = cellidf[i][0]
    c_right = i

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = P_node[i_1]
    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      vi1 = Pbordnode[i_1]
    vi2 = P_node[i_2]
    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      vi2 = Pbordnode[i_2]

    vv1 = P_c[c_left]
    vv2 = P_ghost[c_right]

    Px_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    Py_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

  for i in halofaces:

    c_left = cellidf[i][0]
    c_right = halofid[i]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = P_node[i_1]
    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      vi1 = Pbordnode[i_1]
    vi2 = P_node[i_2]
    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      vi2 = Pbordnode[i_2]

    vv1 = P_c[c_left]
    vv2 = P_halo[c_right]

    Px_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    Py_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

  for i in dirichletfaces:
    c_left = cellidf[i][0]
    c_right = i

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    vi1 = Pbordnode[i_1]
    vi2 = Pbordnode[i_2]
    vv1 = P_c[c_left]

    VK = Pbordface[i]
    vv2 = 2. * VK - vv1

    Px_face[i] = -1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][1] + (vv1 + vi2) * f_2[i][1] + (vi2 + vv2) * f_3[i][1] + (vv2 + vi1) * f_4[i][1])
    Py_face[i] = 1 / (2 * airDiamond[i]) * (
              (vi1 + vv1) * f_1[i][0] + (vv1 + vi2) * f_2[i][0] + (vi2 + vv2) * f_3[i][0] + (vv2 + vi1) * f_4[i][0])

def _get_triplet_2d(cellfid: 'int32[:,:]', nodeidf: 'int32[:,:]', vertexn: 'float[:,:]', halofid: 'int32[:]',
                   haloext: 'int32[:,:]', oldnamen: 'uint32[:]', volume: 'float[:]',
                   cellnid: 'int32[:,:]', centerc: 'float[:,:]', centerh: 'float[:,:]', halonid: 'int32[:,:]',
                   periodicnid: 'int32[:,:]',
                   centergn: 'float[:,:,:]', halocentergn: 'float[:,:,:]', airDiamond: 'float[:]',
                   lambda_x: 'float[:]', lambda_y: 'float[:]', lambda_z: 'float[:]', number: 'uint32[:]',
                   R_x: 'float[:]', R_y: 'float[:]',
                   R_z: 'float[:]', param1: 'float[:]', param2: 'float[:]', param3: 'float[:]', param4: 'float[:]',
                   shift: 'float[:,:]',
                   nbelements: 'int32', loctoglob: 'int32[:]', BCdirichlet: 'uint32[:]', a_loc: 'float[:]',
                   irn_loc: 'int32[:]', jcn_loc: 'int32[:]',
                   matrixinnerfaces: 'uint32[:]', halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  center = np.zeros(2)
  parameters = np.zeros(2)
  cmpt = 0

  for i in matrixinnerfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    parameters[0] = param4[i];
    parameters[1] = param2[i]

    c_right = cellfid[i][1]
    c_rightglob = loctoglob[c_right]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = param1[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    cmptparam = 0
    for nod in nodeidf[i][:nodeidf[i][-1]]:
      if _search_element(BCdirichlet, oldnamen[nod]) == 0:
        for j in range(cellnid[nod][-1]):
          center[:] = centerc[cellnid[nod][j]][0:2]
          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(len(centergn[nod])):
          if centergn[nod][j][-1] != -1:
            center[:] = centergn[nod][j][0:2]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            alpha = (1. + lambda_x[nod] * xdiff + \
                     lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
            index = np.int32(centergn[nod][j][2])
            value = alpha / volume[c_left] * parameters[cmptparam]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = loctoglob[index]
            a_loc[cmpt] = value
            cmpt = cmpt + 1
            # right cell-----------------------------------
            value = -1. * alpha / volume[c_right] * parameters[cmptparam]
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = loctoglob[index]
            a_loc[cmpt] = value
            cmpt = cmpt + 1

        for j in range(len(halocentergn[nod])):
          if halocentergn[nod][j][-1] != -1:
            center[:] = halocentergn[nod][j][0:2]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            alpha = (1. + lambda_x[nod] * xdiff + \
                     lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
            index = np.int32(halocentergn[nod][j][2])
            value = alpha / volume[c_left] * parameters[cmptparam]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = haloext[index][0]
            a_loc[cmpt] = value
            cmpt = cmpt + 1
            # right cell-----------------------------------
            value = -1. * alpha / volume[c_right] * parameters[cmptparam]
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = haloext[index][0]
            a_loc[cmpt] = value
            cmpt = cmpt + 1

        for j in range(periodicnid[nod][-1]):
          if vertexn[nod][3] == 11 or vertexn[nod][3] == 22:
            center[0] = centerc[periodicnid[nod][j]][0] + shift[periodicnid[nod][j]][0]
            center[1] = centerc[periodicnid[nod][j]][1]
          if vertexn[nod][3] == 33 or vertexn[nod][3] == 44:
            center[0] = centerc[periodicnid[nod][j]][0]
            center[1] = centerc[periodicnid[nod][j]][1] + shift[periodicnid[nod][j]][1]

          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(halonid[nod][-1]):
          center[:] = centerh[halonid[nod][j]][0:2]
          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
      cmptparam = +1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    # right cell------------------------------------------------------
    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * param1[i] / volume[c_right]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_rightglob
    value = -1. * param3[i] / volume[c_right]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

  for i in halofaces:

    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    parameters[0] = param4[i];
    parameters[1] = param2[i]

    c_rightglob = haloext[halofid[i]][0]
    c_right = halofid[i]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = param1[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    cmptparam = 0
    for nod in nodeidf[i][:nodeidf[i][-1]]:
      if _search_element(BCdirichlet, oldnamen[nod]) == 0:
        for j in range(cellnid[nod][-1]):
          center[:] = centerc[cellnid[nod][j]][0:2]
          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(len(centergn[nod])):
          if centergn[nod][j][-1] != -1:
            center[:] = centergn[nod][j][0:2]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            alpha = (1. + lambda_x[nod] * xdiff + \
                     lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
            index = np.int32(centergn[nod][j][2])
            value = alpha / volume[c_left] * parameters[cmptparam]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = loctoglob[index]
            a_loc[cmpt] = value
            cmpt = cmpt + 1

        for j in range(len(halocentergn[nod])):
          if halocentergn[nod][j][-1] != -1:
            center[:] = halocentergn[nod][j][0:2]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            alpha = (1. + lambda_x[nod] * xdiff + \
                     lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
            index = np.int32(halocentergn[nod][j][2])
            value = alpha / volume[c_left] * parameters[cmptparam]
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = haloext[index][0]
            a_loc[cmpt] = value
            cmpt = cmpt + 1

        for j in range(halonid[nod][-1]):
          center[:] = centerh[halonid[nod][j]][0:2]
          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff) / (number[nod] + lambda_x[nod] * R_x[nod] + lambda_y[nod] * R_y[nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
      cmptparam += 1

  for i in dirichletfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = param1[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

def _compute_2dmatrix_size(nodeidf: 'int32[:,:]', halofid: 'int32[:]', cellnid: 'int32[:,:]', halonid: 'int32[:,:]',
                          periodicnid: 'int32[:,:]',
                          centergn: 'float[:,:,:]', halocentergn: 'float[:,:,:]', oldnamen: 'uint32[:]',
                          BCdirichlet: 'uint32[:]',
                          matrixinnerfaces: 'uint32[:]', halofaces: 'uint32[:]',
                          dirichletfaces: 'uint32[:]'):

  cmpt = 0
  for i in matrixinnerfaces:
    cmpt = cmpt + 1

    for nod in nodeidf[i][:nodeidf[i][-1]]:
      if _search_element(BCdirichlet, oldnamen[nod]) == 0:  # and search_element(BCneumannNH, oldnamen[nod]) == 0:
        # if vertexn[nod][3] not in BCdirichlet:
        for j in range(cellnid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(len(centergn[nod])):
          if centergn[nod][j][-1] != -1:
            cmpt = cmpt + 1
            # right cell-----------------------------------
            cmpt = cmpt + 1

        for j in range(len(halocentergn[nod])):
          if halocentergn[nod][j][-1] != -1:
            cmpt = cmpt + 1
            # right cell-----------------------------------
            cmpt = cmpt + 1

        for j in range(periodicnid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(halonid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

    cmpt = cmpt + 1
    # right cell------------------------------------------------------
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  # elif namef[i] == 10:
  for i in halofaces:
    cmpt = cmpt + 1

    cmpt = cmpt + 1
    cmpt = cmpt + 1

    for nod in nodeidf[i][:nodeidf[i][-1]]:
      if _search_element(BCdirichlet, oldnamen[nod]) == 0:
        for j in range(cellnid[nod][-1]):
          cmpt = cmpt + 1

        for j in range(len(centergn[nod])):
          if centergn[nod][j][-1] != -1:
            cmpt = cmpt + 1

        for j in range(len(halocentergn[nod])):
          if halocentergn[nod][j][-1] != -1:
            cmpt = cmpt + 1

        for j in range(halonid[nod][-1]):
          cmpt = cmpt + 1

  for i in dirichletfaces:
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  return cmpt

def _get_rhs_glob_2d(cellfid: 'int32[:,:]', nodeidf: 'int32[:,:]', oldname: 'uint32[:]',
                    volume: 'float[:]', centergn: 'float[:,:,:]', loctoglob: 'int32[:]', param1: 'float[:]',
                    param2: 'float[:]',
                    param3: 'float[:]', param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'uint32[:]', centergf: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                    halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  rhs[:] = 0.
  for i in matrixinnerfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    c_right = cellfid[i][1]
    c_rightglob = loctoglob[c_right]

    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs[c_leftglob] += value_left

      value_right = V * param4[i] / volume[c_right]
      rhs[c_rightglob] += value_right

    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs[c_leftglob] += value_left

      value_right = V * param2[i] / volume[c_right]
      rhs[c_rightglob] += value_right

  for i in halofaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs[c_leftglob] += value_left

    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs[c_leftglob] += value_left

  for i in dirichletfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    if centergn[i_1][0][2] != -1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs[c_leftglob] += value_left

    if centergn[i_2][0][2] != -1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs[c_leftglob] += value_left

    V_K = Pbordface[i]
    value = -2. * param3[i] / volume[c_left] * V_K
    rhs[c_leftglob] += value

def _get_rhs_loc_2d(cellfid: 'int32[:,:]', nodeidf: 'int32[:,:]', oldname: 'uint32[:]',
                   volume: 'float[:]', centergn: 'float[:,:,:]', loctoglob: 'int32[:]', param1: 'float[:]',
                   param2: 'float[:]',
                   param3: 'float[:]', param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                   rhs_loc: 'float[:]',
                   BCdirichlet: 'uint32[:]', centergf: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                   halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):


  rhs_loc[:] = 0.
  for i in matrixinnerfaces:
    c_right = cellfid[i][1]
    c_left = cellfid[i][0]
    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs_loc[c_left] += value_left

      value_right = V * param4[i] / volume[c_right]
      rhs_loc[c_right] += value_right

    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs_loc[c_left] += value_left

      value_right = V * param2[i] / volume[c_right]
      rhs_loc[c_right] += value_right

  for i in halofaces:
    c_left = cellfid[i][0]
    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs_loc[c_left] += value_left

    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs_loc[c_left] += value_left

  # TODO verify
  for i in dirichletfaces:

    c_left = cellfid[i][0]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    if centergn[i_1][0][2] != -1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs_loc[c_left] += value_left

    if centergn[i_2][0][2] != -1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs_loc[c_left] += value_left

    V_K = Pbordface[i]
    value = -2. * param3[i] / volume[c_left] * V_K
    rhs_loc[c_left] += value

def _get_rhs_glob_2d(cellfid: 'int32[:,:]', nodeidf: 'int32[:,:]', oldname: 'uint32[:]',
                    volume: 'float[:]', centergn: 'float[:,:,:]', loctoglob: 'int32[:]', param1: 'float[:]',
                    param2: 'float[:]',
                    param3: 'float[:]', param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'uint32[:]', centergf: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                    halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  rhs[:] = 0.
  for i in matrixinnerfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    c_right = cellfid[i][1]
    c_rightglob = loctoglob[c_right]

    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs[c_leftglob] += value_left

      value_right = V * param4[i] / volume[c_right]
      rhs[c_rightglob] += value_right

    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs[c_leftglob] += value_left

      value_right = V * param2[i] / volume[c_right]
      rhs[c_rightglob] += value_right

  for i in halofaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs[c_leftglob] += value_left

    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs[c_leftglob] += value_left

  for i in dirichletfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]

    if centergn[i_1][0][2] != -1:
      V = Pbordnode[i_1]
      value_left = -1. * V * param4[i] / volume[c_left]
      rhs[c_leftglob] += value_left

    if centergn[i_2][0][2] != -1:
      V = Pbordnode[i_2]
      value_left = -1. * V * param2[i] / volume[c_left]
      rhs[c_leftglob] += value_left

    V_K = Pbordface[i]
    value = -2. * param3[i] / volume[c_left] * V_K
    rhs[c_leftglob] += value



def _compute_P_gradient_2d_FV4():
  pass

###########################################################################
def _get_rhs_glob_3d(cellfid: 'int32[:,:]', nodeidf: 'int32[:,:]', oldname: 'uint32[:]',
                    volume: 'float[:]', centergn: 'float[:,:,:]', loctoglob: 'int32[:]', param1: 'float[:]',
                    param2: 'float[:]',
                    param3: 'float[:]', param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'uint32[:]', centergf: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                    halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=np.int32)

  for i in matrixinnerfaces:

    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    c_right = cellfid[i][1]
    c_rightglob = loctoglob[c_right]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs[c_leftglob] += value_left

        value_right = V * parameters[cmpt] / volume[c_right]
        rhs[c_rightglob] += value_right

      cmpt = cmpt + 1

  for i in halofaces:

    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs[c_leftglob] += value_left
      cmpt = cmpt + 1

  for i in dirichletfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    cmpt = 0
    for nod in nodes:
      if centergn[nod][0][3] != -1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs[c_leftglob] += value_left

      cmpt = cmpt + 1

    V_K = Pbordface[i]
    value = -2. * param3[i] / volume[c_left] * V_K
    rhs[c_leftglob] += value



def _get_rhs_loc_3d(cellfid: 'int32[:,:]', nodeidf: 'int32[:,:]', oldname: 'uint32[:]',
                   volume: 'float[:]', centergn: 'float[:,:,:]', loctoglob: 'int32[:]', param1: 'float[:]',
                   param2: 'float[:]',
                   param3: 'float[:]', param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                   rhs_loc: 'float[:]',
                   BCdirichlet: 'uint32[:]', centergf: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                   halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  rhs_loc[:] = 0.
  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=np.int32)

  for i in matrixinnerfaces:

    c_left = cellfid[i][0]
    c_right = cellfid[i][1]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs_loc[c_left] += value_left

        value_right = V * parameters[cmpt] / volume[c_right]
        rhs_loc[c_right] += value_right

      cmpt = cmpt + 1

  for i in halofaces:

    c_left = cellfid[i][0]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs_loc[c_left] += value_left
      cmpt = cmpt + 1

  for i in dirichletfaces:
    c_left = cellfid[i][0]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    cmpt = 0
    for nod in nodes:
      if centergn[nod][0][3] != -1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs_loc[c_left] += value_left

      cmpt += 1

    V_K = Pbordface[i]
    value = -2. * param3[i] / volume[c_left] * V_K
    rhs_loc[c_left] += value



def _get_rhs_glob_3d(cellfid: 'int32[:,:]', nodeidf: 'int32[:,:]', oldname: 'uint32[:]',
                    volume: 'float[:]', centergn: 'float[:,:,:]', loctoglob: 'int32[:]', param1: 'float[:]',
                    param2: 'float[:]',
                    param3: 'float[:]', param4: 'float[:]', Pbordnode: 'float[:]', Pbordface: 'float[:]',
                    rhs: 'float[:]',
                    BCdirichlet: 'uint32[:]', centergf: 'float[:,:]', matrixinnerfaces: 'uint32[:]',
                    halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=np.int32)

  for i in matrixinnerfaces:

    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    c_right = cellfid[i][1]
    c_rightglob = loctoglob[c_right]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs[c_leftglob] += value_left

        value_right = V * parameters[cmpt] / volume[c_right]
        rhs[c_rightglob] += value_right

      cmpt = cmpt + 1

  for i in halofaces:

    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    cmpt = 0
    for nod in nodes:
      if _search_element(BCdirichlet, oldname[nod]) == 1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs[c_leftglob] += value_left
      cmpt = cmpt + 1

  for i in dirichletfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    cmpt = 0
    for nod in nodes:
      if centergn[nod][0][3] != -1:
        V = Pbordnode[nod]
        value_left = -1. * V * parameters[cmpt] / volume[c_left]
        rhs[c_leftglob] += value_left

      cmpt = cmpt + 1

    V_K = Pbordface[i]
    value = -2. * param3[i] / volume[c_left] * V_K
    rhs[c_leftglob] += value


def _compute_P_gradient_3d_diamond(val_c: 'float[:]', v_ghost: 'float[:]', v_halo: 'float[:]', v_node: 'float[:]',
                                  cellidf: 'int32[:,:]',
                                  nodeidf: 'int32[:,:]', centergf: 'float[:,:]', halofid: 'int32[:]',
                                  centerc: 'float[:,:]',
                                  centerh: 'float[:,:]', oldname: 'uint32[:]', airDiamond: 'float[:]',
                                  n1: 'float[:,:]', n2: 'float[:,:]',
                                  n3: 'float[:,:]', n4: 'float[:,:]', normalf: 'float[:,:]', shift: 'float[:,:]',
                                  Pbordnode: 'float[:]',
                                  Pbordface: 'float[:]',
                                  Px_face: 'float[:]', Py_face: 'float[:]', Pz_face: 'float[:]',
                                  BCdirichlet: 'uint32[:]', innerfaces: 'uint32[:]',
                                  halofaces: 'uint32[:]', neumannfaces: 'uint32[:]', dirichletfaces: 'uint32[:]',
                                  periodicfaces: 'uint32[:]'):

  for i in innerfaces:

    c_left = cellidf[i][0]
    c_right = cellidf[i][1]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]
    i_3 = nodeidf[i][2]
    i_4 = i_3
    if nodeidf[i][-1] == 4:
      i_4 = nodeidf[i][3]

    V_A = v_node[i_1]
    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V_A = Pbordnode[i_1]
    V_B = v_node[i_2]
    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V_B = Pbordnode[i_2]
    V_C = v_node[i_3]
    if _search_element(BCdirichlet, oldname[i_3]) == 1:
      V_C = Pbordnode[i_3]
    V_D = v_node[i_4]
    if _search_element(BCdirichlet, oldname[i_4]) == 1:
      V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_R = val_c[c_right]

    Px_face[i] = -1. * (n1[i][0] * (V_A - V_C) + n2[i][0] * (V_B - V_D) + normalf[i][0] * (V_R - V_L)) / airDiamond[i]
    Py_face[i] = -1. * (n1[i][1] * (V_A - V_C) + n2[i][1] * (V_B - V_D) + normalf[i][1] * (V_R - V_L)) / airDiamond[i]
    Pz_face[i] = -1. * (n1[i][2] * (V_A - V_C) + n2[i][2] * (V_B - V_D) + normalf[i][2] * (V_R - V_L)) / airDiamond[i]

  for i in periodicfaces:

    c_left = cellidf[i][0]
    c_right = cellidf[i][1]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]
    i_3 = nodeidf[i][2]
    i_4 = i_3
    if nodeidf[i][-1] == 4:
      i_4 = nodeidf[i][3]

    V_A = v_node[i_1]
    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V_A = Pbordnode[i_1]
    V_B = v_node[i_2]
    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V_B = Pbordnode[i_2]
    V_C = v_node[i_3]
    if _search_element(BCdirichlet, oldname[i_3]) == 1:
      V_C = Pbordnode[i_3]
    V_D = v_node[i_4]
    if _search_element(BCdirichlet, oldname[i_4]) == 1:
      V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_R = val_c[c_right]

    Px_face[i] = -1. * (n1[i][0] * (V_A - V_C) + n2[i][0] * (V_B - V_D) + normalf[i][0] * (V_R - V_L)) / airDiamond[i]
    Py_face[i] = -1. * (n1[i][1] * (V_A - V_C) + n2[i][1] * (V_B - V_D) + normalf[i][1] * (V_R - V_L)) / airDiamond[i]
    Pz_face[i] = -1. * (n1[i][2] * (V_A - V_C) + n2[i][2] * (V_B - V_D) + normalf[i][2] * (V_R - V_L)) / airDiamond[i]

  for i in neumannfaces:
    c_left = cellidf[i][0]
    c_right = i

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]
    i_3 = nodeidf[i][2]
    i_4 = i_3
    if nodeidf[i][-1] == 4:
      i_4 = nodeidf[i][3]

    V_A = v_node[i_1]
    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V_A = Pbordnode[i_1]
    V_B = v_node[i_2]
    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V_B = Pbordnode[i_2]
    V_C = v_node[i_3]
    if _search_element(BCdirichlet, oldname[i_3]) == 1:
      V_C = Pbordnode[i_3]
    V_D = v_node[i_4]
    if _search_element(BCdirichlet, oldname[i_4]) == 1:
      V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_R = v_ghost[c_right]

    Px_face[i] = -1. * (n1[i][0] * (V_A - V_C) + n2[i][0] * (V_B - V_D) + normalf[i][0] * (V_R - V_L)) / airDiamond[i]
    Py_face[i] = -1. * (n1[i][1] * (V_A - V_C) + n2[i][1] * (V_B - V_D) + normalf[i][1] * (V_R - V_L)) / airDiamond[i]
    Pz_face[i] = -1. * (n1[i][2] * (V_A - V_C) + n2[i][2] * (V_B - V_D) + normalf[i][2] * (V_R - V_L)) / airDiamond[i]

  for i in halofaces:
    c_left = cellidf[i][0]
    c_right = halofid[i]

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]
    i_3 = nodeidf[i][2]
    i_4 = i_3
    if nodeidf[i][-1] == 4:
      i_4 = nodeidf[i][3]

    V_A = v_node[i_1]
    if _search_element(BCdirichlet, oldname[i_1]) == 1:
      V_A = Pbordnode[i_1]
    V_B = v_node[i_2]
    if _search_element(BCdirichlet, oldname[i_2]) == 1:
      V_B = Pbordnode[i_2]
    V_C = v_node[i_3]
    if _search_element(BCdirichlet, oldname[i_3]) == 1:
      V_C = Pbordnode[i_3]
    V_D = v_node[i_4]
    if _search_element(BCdirichlet, oldname[i_4]) == 1:
      V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_R = v_halo[c_right]

    Px_face[i] = -1. * (n1[i][0] * (V_A - V_C) + n2[i][0] * (V_B - V_D) + normalf[i][0] * (V_R - V_L)) / airDiamond[i]
    Py_face[i] = -1. * (n1[i][1] * (V_A - V_C) + n2[i][1] * (V_B - V_D) + normalf[i][1] * (V_R - V_L)) / airDiamond[i]
    Pz_face[i] = -1. * (n1[i][2] * (V_A - V_C) + n2[i][2] * (V_B - V_D) + normalf[i][2] * (V_R - V_L)) / airDiamond[i]

  for i in dirichletfaces:
    c_left = cellidf[i][0]
    c_right = i

    i_1 = nodeidf[i][0]
    i_2 = nodeidf[i][1]
    i_3 = nodeidf[i][2]
    i_4 = i_3
    if nodeidf[i][-1] == 4:
      i_4 = nodeidf[i][3]

    V_A = Pbordnode[i_1]
    V_B = Pbordnode[i_2]
    V_C = Pbordnode[i_3]
    V_D = Pbordnode[i_4]

    V_L = val_c[c_left]
    V_K = Pbordface[i]
    V_R = 2. * V_K - V_L

    Px_face[i] = -1. * (n1[i][0] * (V_A - V_C) + n2[i][0] * (V_B - V_D) + normalf[i][0] * (V_R - V_L)) / airDiamond[i]
    Py_face[i] = -1. * (n1[i][1] * (V_A - V_C) + n2[i][1] * (V_B - V_D) + normalf[i][1] * (V_R - V_L)) / airDiamond[i]
    Pz_face[i] = -1. * (n1[i][2] * (V_A - V_C) + n2[i][2] * (V_B - V_D) + normalf[i][2] * (V_R - V_L)) / airDiamond[i]

def _get_triplet_3d(cellfid: 'int32[:,:]', nodeidf: 'int32[:,:]', vertexn: 'float[:,:]', halofid: 'int32[:]',
                   haloext: 'int32[:,:]', oldnamen: 'uint32[:]', volume: 'float[:]',
                   cellnid: 'int32[:,:]', centerc: 'float[:,:]', centerh: 'float[:,:]', halonid: 'int32[:,:]',
                   periodicnid: 'int32[:,:]',
                   centergn: 'float[:,:,:]', halocentergn: 'float[:,:,:]', airDiamond: 'float[:]',
                   lambda_x: 'float[:]', lambda_y: 'float[:]', lambda_z: 'float[:]', number: 'uint32[:]',
                   R_x: 'float[:]', R_y: 'float[:]',
                   R_z: 'float[:]', param1: 'float[:]', param2: 'float[:]', param3: 'float[:]', param4: 'float[:]',
                   shift: 'float[:,:]',
                   nbelements: 'intc', loctoglob: 'int32[:]', BCdirichlet: 'uint32[:]', a_loc: 'float[:]',
                   irn_loc: 'int32[:]', jcn_loc: 'int32[:]',
                   matrixinnerfaces: 'uint32[:]', halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  parameters = np.zeros(4)
  nodes = np.zeros(4, dtype=np.int32)

  cmpt = 0

  for i in matrixinnerfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    c_right = cellfid[i][1]
    c_rightglob = loctoglob[c_right]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1 * param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    cmptparam = 0
    for nod in nodes:
      if _search_element(BCdirichlet, oldnamen[nod]) == 0:
        for j in range(cellnid[nod][-1]):
          center = centerc[cellnid[nod][j]]
          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          zdiff = center[2] - vertexn[nod][2]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                     lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                       nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(halonid[nod][-1]):
          center = centerh[halonid[nod][j]]
          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          zdiff = center[2] - vertexn[nod][2]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                     lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                       nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(periodicnid[nod][-1]):
          if vertexn[nod][3] == 11 or vertexn[nod][3] == 22:
            center[0] = centerc[periodicnid[nod][j]][0] + shift[periodicnid[nod][j]][0]
            center[1] = centerc[periodicnid[nod][j]][1]
            center[2] = centerc[periodicnid[nod][j]][2]
          if vertexn[nod][3] == 33 or vertexn[nod][3] == 44:
            center[0] = centerc[periodicnid[nod][j]][0]
            center[1] = centerc[periodicnid[nod][j]][1] + shift[periodicnid[nod][j]][1]
            center[2] = centerc[periodicnid[nod][j]][2]
          if vertexn[nod][3] == 55 or vertexn[nod][3] == 66:
            center[0] = centerc[periodicnid[nod][j]][0]
            center[1] = centerc[periodicnid[nod][j]][1]
            center[2] = centerc[periodicnid[nod][j]][2] + shift[periodicnid[nod][j]][2]

          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          zdiff = center[2] - vertexn[nod][2]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                     lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                       nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1
          # right cell-----------------------------------
          value = -1. * alpha / volume[c_right] * parameters[cmptparam]
          irn_loc[cmpt] = c_rightglob
          jcn_loc[cmpt] = loctoglob[periodicnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(len(centergn[nod])):
          if centergn[nod][j][-1] != -1:
            center = centergn[nod][j][0:3]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod] * xdiff + \
                     lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                       lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                         nod])
            value = alpha / volume[c_left] * parameters[cmptparam]

            index = np.int32(centergn[nod][j][3])
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = loctoglob[index]
            a_loc[cmpt] = value
            cmpt = cmpt + 1
            # right cell-----------------------------------
            value = -1. * alpha / volume[c_right] * parameters[cmptparam]
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = loctoglob[index]
            a_loc[cmpt] = value
            cmpt = cmpt + 1

        for j in range(len(halocentergn[nod])):
          if halocentergn[nod][j][-1] != -1:
            center = halocentergn[nod][j][0:3]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod] * xdiff + \
                     lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                       lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                         nod])
            value = alpha / volume[c_left] * parameters[cmptparam]
            index = np.int32(halocentergn[nod][j][3])
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = haloext[index][0]
            a_loc[cmpt] = value
            cmpt = cmpt + 1
            # right cell-----------------------------------
            value = -1. * alpha / volume[c_right] * parameters[cmptparam]
            irn_loc[cmpt] = c_rightglob
            jcn_loc[cmpt] = haloext[index][0]
            a_loc[cmpt] = value
            cmpt = cmpt + 1
      cmptparam = cmptparam + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    # right cell------------------------------------------------------
    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_leftglob
    value = param3[i] / volume[c_right]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_rightglob
    jcn_loc[cmpt] = c_rightglob
    value = -1. * param3[i] / volume[c_right]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

  for i in halofaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    c_rightglob = haloext[halofid[i]][0]
    c_right = halofid[i]

    cmptparam = 0
    for nod in nodes:
      if _search_element(BCdirichlet, oldnamen[nod]) == 0:
        for j in range(cellnid[nod][-1]):
          center = centerc[cellnid[nod][j]]
          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          zdiff = center[2] - vertexn[nod][2]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                     lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                       nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = loctoglob[cellnid[nod][j]]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(halonid[nod][-1]):
          center = centerh[halonid[nod][j]]
          xdiff = center[0] - vertexn[nod][0]
          ydiff = center[1] - vertexn[nod][1]
          zdiff = center[2] - vertexn[nod][2]
          alpha = (1. + lambda_x[nod] * xdiff + \
                   lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                     lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                       nod])
          value = alpha / volume[c_left] * parameters[cmptparam]
          irn_loc[cmpt] = c_leftglob
          jcn_loc[cmpt] = haloext[halonid[nod][j]][0]
          a_loc[cmpt] = value
          cmpt = cmpt + 1

        for j in range(len(centergn[nod])):
          if centergn[nod][j][-1] != -1:
            center = centergn[nod][j][0:3]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod] * xdiff + \
                     lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                       lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                         nod])
            value = alpha / volume[c_left] * parameters[cmptparam]

            index = np.int32(centergn[nod][j][3])
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = loctoglob[index]
            a_loc[cmpt] = value
            cmpt = cmpt + 1

        for j in range(len(halocentergn[nod])):
          if halocentergn[nod][j][-1] != -1:
            center = halocentergn[nod][j][0:3]
            xdiff = center[0] - vertexn[nod][0]
            ydiff = center[1] - vertexn[nod][1]
            zdiff = center[2] - vertexn[nod][2]
            alpha = (1. + lambda_x[nod] * xdiff + \
                     lambda_y[nod] * ydiff + lambda_z[nod] * zdiff) / (number[nod] + lambda_x[nod] * R_x[nod] + \
                                                                       lambda_y[nod] * R_y[nod] + lambda_z[nod] * R_z[
                                                                         nod])
            value = alpha / volume[c_left] * parameters[cmptparam]
            index = np.int32(halocentergn[nod][j][3])
            irn_loc[cmpt] = c_leftglob
            jcn_loc[cmpt] = haloext[index][0]
            a_loc[cmpt] = value
            cmpt = cmpt + 1
      cmptparam = cmptparam + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1 * param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_rightglob
    value = param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

  for i in dirichletfaces:
    c_left = cellfid[i][0]
    c_leftglob = loctoglob[c_left]

    parameters[0] = param1[i];
    parameters[1] = param2[i]
    parameters[2] = -1. * param1[i];
    parameters[3] = -1. * param2[i]

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1 * param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

    irn_loc[cmpt] = c_leftglob
    jcn_loc[cmpt] = c_leftglob
    value = -1. * param3[i] / volume[c_left]
    a_loc[cmpt] = value
    cmpt = cmpt + 1

def _compute_3dmatrix_size(nodeidf: 'int32[:,:]', halofid: 'int32[:]', cellnid: 'int32[:,:]', halonid: 'int32[:,:]',
                          periodicnid: 'int32[:,:]',
                          centergn: 'float[:,:,:]', halocentergn: 'float[:,:,:]', oldnamen: 'uint32[:]',
                          BCdirichlet: 'uint32[:]',
                          matrixinnerfaces: 'uint32[:]', halofaces: 'uint32[:]', dirichletfaces: 'uint32[:]'):

  cmpt = 0
  nodes = np.zeros(4, dtype=np.int32)

  for i in matrixinnerfaces:

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    cmpt = cmpt + 1

    for nod in nodes:
      if _search_element(BCdirichlet, oldnamen[nod]) == 0:
        for j in range(cellnid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(len(centergn[nod])):
          if centergn[nod][j][-1] != -1:
            cmpt = cmpt + 1
            # right cell-----------------------------------
            cmpt = cmpt + 1

        for j in range(len(halocentergn[nod])):
          if halocentergn[nod][j][-1] != -1:
            cmpt = cmpt + 1
            # right cell-----------------------------------
            cmpt = cmpt + 1

        for j in range(periodicnid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

        for j in range(halonid[nod][-1]):
          cmpt = cmpt + 1
          # right cell-----------------------------------
          cmpt = cmpt + 1

    cmpt = cmpt + 1
    # right cell------------------------------------------------------
    cmpt = cmpt + 1
    cmpt = cmpt + 1

  for i in halofaces:

    nodes[0:3] = nodeidf[i][0:3]
    nodes[3] = nodeidf[i][2]
    if nodeidf[i][-1] == 4:
      nodes[3] = nodeidf[i][3]

    cmpt = cmpt + 1
    cmpt = cmpt + 1

    for nod in nodes:
      if _search_element(BCdirichlet, oldnamen[nod]) == 0:
        for j in range(cellnid[nod][-1]):
          cmpt = cmpt + 1

        for j in range(len(centergn[nod])):
          if centergn[nod][j][-1] != -1:
            cmpt = cmpt + 1

        for j in range(len(halocentergn[nod])):
          if halocentergn[nod][j][-1] != -1:
            cmpt = cmpt + 1

        for j in range(halonid[nod][-1]):
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
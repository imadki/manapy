import numpy as np
from manapy.backends.compile_fun import compile

def _update_SW(h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', hc_c: 'float[:]', Z_c: 'float[:]',
              rez_h: 'float[:]', rez_hu: 'float[:]', rez_hv: 'float[:]', rez_hc: 'float[:]', rez_Z: 'float[:]',
              src_h: 'float[:]', src_hu: 'float[:]', src_hv: 'float[:]', src_hc: 'float[:]', src_Z: 'float[:]',
              dissip_hc: 'float[:]', corio_hu: 'float[:]', corio_hv: 'float[:]', wind_hu: 'float', wind_hv: 'float',
              dtime: 'float', cell_volume: 'float[:]'):
  for i in range(len(h_c)):
    h_c[i] += dtime * (rez_h[i] + src_h[i]) / cell_volume[i]
    hu_c[i] += dtime * ((rez_hu[i] + src_hu[i]) / cell_volume[i] + corio_hu[i] + wind_hu)
    hv_c[i] += dtime * ((rez_hv[i] + src_hv[i]) / cell_volume[i] + corio_hv[i] + wind_hv)
    hc_c[i] += dtime * (rez_hc[i] + src_hc[i] - dissip_hc[i]) / cell_volume[i]
    Z_c[i] += dtime * (rez_Z[i] + src_Z[i]) / cell_volume[i]


def _time_step_SW(h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', cfl: 'float', face_normal: 'float[:,:]',
                 face_measure: 'float[:]', cell_volume: 'float[:]',
                 cell_faceid: 'int[:,:]', grav: 'float', Dxx: 'float', Dyy: 'float'):
  # from numpy import sqrt, fabs, zeros
  nbelement = len(cell_faceid)
  u_n = 0.
  norm = np.zeros(3)
  dt = 1e6

  for i in range(nbelement):
    velson = np.sqrt(grav * h_c[i])
    lam = 0.
    for j in range(cell_faceid[i][-1]):
      norm[:] = face_normal[cell_faceid[i][j]][:]

      # convective part
      u_n = np.fabs(hu_c[i] / h_c[i] * norm[0] + hv_c[i] / h_c[i] * norm[1])
      lam_convect = u_n / face_measure[cell_faceid[i][j]] + velson
      lam += lam_convect * face_measure[cell_faceid[i][j]]

      # diffusion part
      mes = np.sqrt(norm[0] * norm[0] + norm[1] * norm[1])
      lam_diff = Dxx * mes ** 2 + Dyy * mes ** 2
      lam += lam_diff / cell_volume[i]
    dt = min(dt, cfl * cell_volume[i] / lam)

  return dt


def _term_source_srnh_SW(src_h: 'float[:]', src_hu: 'float[:]', src_hv: 'float[:]', src_hc: 'float[:]',
                        src_Z: 'float[:]',
                        h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', hc_c: 'float[:]', Z_c: 'float[:]',
                        h_ghost: 'float[:]', hu_ghost: 'float[:]', hv_ghost: 'float[:]', hc_ghost: 'float[:]',
                        Z_ghost: 'float[:]',
                        h_halo: 'float[:]', hu_halo: 'float[:]', hv_halo: 'float[:]', hc_halo: 'float[:]',
                        Z_halo: 'float[:]',
                        h_x: 'float[:]', h_y: 'float[:]', psi: 'float[:]',
                        hx_halo: 'float[:]', hy_halo: 'float[:]', psi_halo: 'float[:]',
                        cell_nodeid: 'int[:,:]', cell_faceid: 'int[:,:]', cell_cellfid: 'int[:,:]', face_cellid: 'int[:,:]',
                        cell_center: 'float[:,:]', cell_nf: 'float[:,:,:]',
                        face_name: 'int[:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                        nodes: 'float[:,:]', face_haloid: 'int[:]', grav: 'float', order: 'intc'):
  nbelement = len(h_c)
  hi_p = np.zeros(3)
  zi_p = np.zeros(3)

  zv = np.zeros(3)

  mata = np.zeros(3)
  matb = np.zeros(3)

  ns = np.zeros((3, 3))
  ss = np.zeros((3, 3))
  s_1 = np.zeros(3)
  s_2 = np.zeros(3)
  s_3 = np.zeros(3)
  b = np.zeros(3)
  G = np.zeros(3)

  for i in range(nbelement):

    G[:] = cell_center[i]
    c_1 = 0.
    c_2 = 0.

    for j in range(3):
      f = cell_faceid[i][j]
      ss[j] = cell_nf[i][j]

      if face_name[f] == 10:

        h_1p = h_c[i]
        z_1p = Z_c[i]

        h_p1 = h_halo[face_haloid[f]]
        z_p1 = Z_halo[face_haloid[f]]

      elif face_name[f] == 0:

        h_1p = h_c[i]
        z_1p = Z_c[i]

        # Neighbour across face f must be taken from face_cellid (the same
        # neighbour the convective flux uses). cell_cellfid does NOT match the
        # flux neighbour for boundary cells in this framework, which breaks the
        # C-property on the boundary cells (interior cells are unaffected).
        if face_cellid[f][0] == i:
          nbr = face_cellid[f][1]
        else:
          nbr = face_cellid[f][0]
        h_p1 = h_c[nbr]
        z_p1 = Z_c[nbr]

      else:
        h_1p = h_c[i]
        z_1p = Z_c[i]

        h_p1 = h_ghost[f]
        z_p1 = Z_ghost[f]

      zv[j] = z_p1
      mata[j] = h_p1 * ss[j][0]
      matb[j] = h_p1 * ss[j][1]

      c_1 = c_1 + (0.5 * (h_1p + h_p1) * 0.5 * (h_1p + h_p1)) * ss[j][0]
      c_2 = c_2 + (0.5 * (h_1p + h_p1) * 0.5 * (h_1p + h_p1)) * ss[j][1]

      hi_p[j] = h_1p
      zi_p[j] = z_1p

    c_3 = 3.0 * h_1p

    delta = (mata[1] * matb[2] - mata[2] * matb[1]) - (mata[0] * matb[2] - matb[0] * mata[2]) + (
              mata[0] * matb[1] - matb[0] * mata[1])

    deltax = c_3 * (mata[1] * matb[2] - mata[2] * matb[1]) - (c_1 * matb[2] - c_2 * mata[2]) + (
              c_1 * matb[1] - c_2 * mata[1])

    deltay = (c_1 * matb[2] - c_2 * mata[2]) - c_3 * (mata[0] * matb[2] - matb[0] * mata[2]) + (
              mata[0] * c_2 - matb[0] * c_1)

    deltaz = (mata[1] * c_2 - matb[1] * c_1) - (mata[0] * c_2 - matb[0] * c_1) + c_3 * (
              mata[0] * matb[1] - matb[0] * mata[1])

    h_1 = deltax / delta
    h_2 = deltay / delta
    h_3 = deltaz / delta

    z_1 = zi_p[0] + hi_p[0] - h_1
    z_2 = zi_p[1] + hi_p[1] - h_2
    z_3 = zi_p[2] + hi_p[2] - h_3

    b[:] = nodes[cell_nodeid[i][1]][0:3]

    ns[0] = np.array([(G[1] - b[1]), -(G[0] - b[0]), 0.])
    ns[1] = ns[0] - ss[
      1]  # N23                                                                                                                                                                       
    ns[2] = ns[0] + ss[0]  # N31    

    s_1 = 0.5 * h_1 * (zv[0] * ss[0] + z_2 * ns[0] + z_3 * (-1) * ns[2])
    s_2 = 0.5 * h_2 * (zv[1] * ss[1] + z_1 * (-1) * ns[0] + z_3 * ns[1])
    s_3 = 0.5 * h_3 * (zv[2] * ss[2] + z_1 * ns[2] + z_2 * (-1) * ns[1])

    # TODO
    src_h[i] = 0
    src_hu[i] = -grav * (s_1[0] + s_2[0] + s_3[0])
    src_hv[i] = -grav * (s_1[1] + s_2[1] + s_3[1])
    src_hc[i] = 0.
    src_Z[i] = 0.



def _srnh_scheme(hu_l: 'float', hu_r: 'float', hv_l: 'float', hv_r: 'float', h_l: 'float', h_r: 'float', hc_l: 'float',
                hc_r: 'float', Z_l: 'float', Z_r: 'float', normal: 'float[:]', mesure: 'float', grav: 'float',
                flux: 'float64[:]'):
  # from numpy import zeros, np.sqrt, np.arccos, cos, fabs, pi

  ninv = np.zeros(2)
  w_dif = np.zeros(5)
  rmat = np.zeros((5, 5))

  As = 0
  p = 0.4
  xi = 1 / (1 - p)
  ninv = np.zeros(2)

  ninv[0] = -1 * normal[1]
  ninv[1] = normal[0]

  u_h = (hu_l / h_l * np.sqrt(h_l)
         + hu_r / h_r * np.sqrt(h_r)) / (np.sqrt(h_l) + np.sqrt(h_r))

  v_h = (hv_l / h_l * np.sqrt(h_l)
         + hv_r / h_r * np.sqrt(h_r)) / (np.sqrt(h_l) + np.sqrt(h_r))

  c_h = (hc_l / h_l * np.sqrt(h_l)
         + hc_r / h_r * np.sqrt(h_r)) / (np.sqrt(h_l) + np.sqrt(h_r))

  # uvh =  np.array([uh, vh])
  un_h = u_h * normal[0] + v_h * normal[1]
  un_h = un_h / mesure
  vn_h = u_h * ninv[0] + v_h * ninv[1]
  vn_h = vn_h / mesure

  hroe = (h_l + h_r) / 2
  uroe = un_h
  vroe = vn_h
  croe = c_h

  uleft = hu_l * normal[0] + hv_l * normal[1]
  uleft = uleft / mesure
  vleft = hu_l * ninv[0] + hv_l * ninv[1]
  vleft = vleft / mesure

  uright = hu_r * normal[0] + hv_r * normal[1]
  uright = uright / mesure
  vright = hu_r * ninv[0] + hv_r * ninv[1]
  vright = vright / mesure

  w_lrh = (h_l + h_r) / 2
  w_lrhu = (uleft + uright) / 2
  w_lrhv = (vleft + vright) / 2
  w_lrhc = (hc_l + hc_r) / 2
  w_lrz = (Z_l + Z_r) / 2

  w_dif[0] = h_r - h_l
  w_dif[1] = uright - uleft
  w_dif[2] = vright - vleft
  w_dif[3] = hc_r - hc_l
  w_dif[4] = Z_r - Z_l

  d = As * xi * (3 * uroe ** 2 + vroe ** 2)
  sound = np.sqrt(grav * hroe)
  Q = -(uroe ** 2 + 3 * grav * (hroe + d)) / 9
  R = uroe * (9 * grav * (2 * hroe - d) - 2 * uroe ** 2) / 54
  theta = np.arccos(R / (np.sqrt(-Q ** 3)))

  # Les valeurs propres
  lambda1 = 2 * np.sqrt(-Q) * np.cos(theta / 3) + (2 / 3) * uroe
  lambda2 = 2 * np.sqrt(-Q) * np.cos((theta + 2 * np.pi) / 3) + (2 / 3) * uroe
  lambda3 = 2 * np.sqrt(-Q) * np.cos((theta + 4 * np.pi) / 3) + (2 / 3) * uroe
  lambda4 = uroe
  lambda5 = uroe

  # définition de alpha
  alpha1 = lambda1 - uroe
  alpha2 = lambda2 - uroe
  alpha3 = lambda3 - uroe

  # définition de beta
  beta = 2 * As * xi * vroe / hroe

  # définition de gamma
  gamma1 = sound ** 2 - uroe ** 2 + lambda2 * lambda3 - beta * alpha2 * alpha3 * vroe
  gamma2 = sound ** 2 - uroe ** 2 + lambda1 * lambda3 - beta * alpha1 * alpha3 * vroe
  gamma3 = sound ** 2 - uroe ** 2 + lambda1 * lambda2 - beta * alpha1 * alpha2 * vroe

  # définition de sigma
  sigma1 = -alpha1 * alpha2 + alpha2 * alpha3 - alpha1 * alpha3 + alpha1 ** 2
  sigma2 = alpha1 * alpha2 + alpha2 * alpha3 - alpha1 * alpha3 - alpha2 ** 2
  sigma3 = alpha1 * alpha2 - alpha2 * alpha3 - alpha1 * alpha3 + alpha3 ** 2  # ici 

  epsilon = 1e-10

  if np.fabs(lambda1) < epsilon:
    sign1 = 0.
  else:
    sign1 = lambda1 / np.fabs(lambda1)

  if np.fabs(lambda2) < epsilon:
    sign2 = 0.
  else:
    sign2 = lambda2 / np.fabs(lambda2)

  if np.fabs(lambda3) < epsilon:
    sign3 = 0.
  else:
    sign3 = lambda3 / np.fabs(lambda3)

  if np.fabs(lambda4) < epsilon:
    sign4 = 0.
  else:
    sign4 = lambda4 / np.fabs(lambda4)

  if np.fabs(lambda5) < epsilon:
    sign5 = 0.
  else:
    sign5 = lambda5 / np.fabs(lambda5)

  # 1ère colonne
  rmat[0][0] = sign1 * (gamma1 / sigma1) - sign2 * (gamma2 / sigma2) + sign3 * (gamma3 / sigma3) + sign5 * (beta * vroe)
  rmat[1][0] = lambda1 * sign1 * (gamma1 / sigma1) - lambda2 * sign2 * (gamma2 / sigma2) + lambda3 * sign3 * (
            gamma3 / sigma3) + sign5 * (beta * uroe * vroe)
  rmat[2][0] = vroe * sign1 * (gamma1 / sigma1) - vroe * sign2 * (gamma2 / sigma2) + vroe * sign3 * (
            gamma3 / sigma3) - vroe * sign5 * (1 - beta * vroe)
  rmat[3][0] = croe * sign1 * (gamma1 / sigma1) - croe * sign2 * (gamma2 / sigma2) + croe * sign3 * (
            gamma3 / sigma3) - croe * sign4 + croe * sign5 * beta * vroe
  rmat[4][0] = (alpha1 ** 2 / sound ** 2 - 1) * sign1 * (gamma1 / sigma1) - (alpha2 ** 2 / sound ** 2 - 1) * sign2 * (
            gamma2 / sigma2) + (alpha3 ** 2 / sound ** 2 - 1) * sign3 * (gamma3 / sigma3) - sign5 * (beta * vroe)

  # 2ème colonne
  rmat[0][1] = - sign1 * (alpha2 + alpha3) / sigma1 + sign2 * (alpha1 + alpha3) / sigma2 - sign3 * (
            alpha1 + alpha2) / sigma3
  rmat[1][1] = - lambda1 * sign1 * (alpha2 + alpha3) / sigma1 + lambda2 * sign2 * (
            alpha1 + alpha3) / sigma2 - lambda3 * sign3 * (alpha1 + alpha2) / sigma3
  rmat[2][1] = - vroe * sign1 * (alpha2 + alpha3) / sigma1 + vroe * sign2 * (
            alpha1 + alpha3) / sigma2 - vroe * sign3 * (alpha1 + alpha2) / sigma3
  rmat[3][1] = - croe * sign1 * (alpha2 + alpha3) / sigma1 + croe * sign2 * (
            alpha1 + alpha3) / sigma2 - croe * sign3 * (alpha1 + alpha2) / sigma3
  rmat[4][1] = - (alpha1 ** 2 / sound ** 2 - 1) * sign1 * (alpha2 + alpha3) / sigma1 + (
            alpha2 ** 2 / sound ** 2 - 1) * sign2 * (alpha1 + alpha3) / sigma2 - (
                         alpha3 ** 2 / sound ** 2 - 1) * sign3 * (alpha1 + alpha2) / sigma3

  # 3ème colonne 
  rmat[0][
    2] = sign1 * beta * alpha2 * alpha3 / sigma1 - sign2 * beta * alpha1 * alpha3 / sigma2 + sign3 * beta * alpha1 * alpha2 / sigma3 - sign5 * beta
  rmat[1][
    2] = lambda1 * sign1 * beta * alpha2 * alpha3 / sigma1 - lambda2 * sign2 * beta * alpha1 * alpha3 / sigma2 + lambda3 * sign3 * beta * alpha1 * alpha2 / sigma3 - sign5 * beta * uroe
  rmat[2][
    2] = vroe * sign1 * beta * alpha2 * alpha3 / sigma1 - vroe * sign2 * beta * alpha1 * alpha3 / sigma2 + vroe * sign3 * beta * alpha1 * alpha2 / sigma3 + sign5 * (
            1 - beta * vroe)
  rmat[3][
    2] = croe * sign1 * beta * alpha2 * alpha3 / sigma1 - croe * sign2 * beta * alpha1 * alpha3 / sigma2 + croe * sign3 * beta * alpha1 * alpha2 / sigma3 - croe * sign5 * beta
  rmat[4][2] = (alpha1 ** 2 / sound ** 2 - 1) * sign1 * beta * alpha2 * alpha3 / sigma1 - (
            alpha2 ** 2 / sound ** 2 - 1) * sign2 * beta * alpha1 * alpha3 / sigma2 + (
                         alpha3 ** 2 / sound ** 2 - 1) * sign3 * beta * alpha1 * alpha2 / sigma3 + sign5 * beta

  # 4ème colonne  
  rmat[0][3] = 0.
  rmat[1][3] = 0.
  rmat[2][3] = 0.
  rmat[3][3] = sign4
  rmat[4][3] = 0.

  # 5ème colone
  rmat[0][4] = sign1 * sound ** 2 / sigma1 - sign2 * sound ** 2 / sigma2 + sign3 * sound ** 2 / sigma3
  rmat[1][
    4] = lambda1 * sign1 * sound ** 2 / sigma1 - lambda2 * sign2 * sound ** 2 / sigma2 + lambda3 * sign3 * sound ** 2 / sigma3
  rmat[2][
    4] = vroe * sign1 * sound ** 2 / sigma1 - vroe * sign2 * sound ** 2 / sigma2 + vroe * sign3 * sound ** 2 / sigma3
  rmat[3][
    4] = croe * sign1 * sound ** 2 / sigma1 - croe * sign2 * sound ** 2 / sigma2 + croe * sign3 * sound ** 2 / sigma3
  rmat[4][4] = (alpha1 ** 2 / sound ** 2 - 1) * sign1 * sound ** 2 / sigma1 - (
            alpha2 ** 2 / sound ** 2 - 1) * sign2 * sound ** 2 / sigma2 + (
                         alpha3 ** 2 / sound ** 2 - 1) * sign3 * sound ** 2 / sigma3

  hnew = sum(rmat[0][:] * w_dif[:])
  unew = sum(rmat[1][:] * w_dif[:])
  vnew = sum(rmat[2][:] * w_dif[:])
  cnew = sum(rmat[3][:] * w_dif[:])
  znew = sum(rmat[4][:] * w_dif[:])

  u_h = hnew / 2
  u_hu = unew / 2
  u_hv = vnew / 2
  u_hc = cnew / 2
  u_z = znew / 2

  w_lrh = w_lrh - u_h
  w_lrhu = w_lrhu - u_hu
  w_lrhv = w_lrhv - u_hv
  w_lrhc = w_lrhc - u_hc
  w_lrz = w_lrz - u_z

  unew = 0.
  vnew = 0.

  unew = w_lrhu * normal[0] + w_lrhv * -1 * normal[1]
  unew = unew / mesure
  vnew = w_lrhu * -1 * ninv[0] + w_lrhv * ninv[1]
  vnew = vnew / mesure

  w_lrhu = unew
  w_lrhv = vnew

  q_s = normal[0] * unew + normal[1] * vnew

  flux[0] = q_s
  flux[1] = q_s * w_lrhu / w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[0]
  flux[2] = q_s * w_lrhv / w_lrh + 0.5 * grav * w_lrh * w_lrh * normal[1]
  flux[3] = q_s * w_lrhc / w_lrh
  flux[4] = As * xi * normal[0] * unew * (unew ** 2 + vnew ** 2) / w_lrh ** 3 + As * xi * normal[1] * vnew * (
            unew ** 2 + vnew ** 2) / w_lrh ** 3


def _explicitscheme_convective_SW(rez_h: 'float[:]', rez_hu: 'float[:]', rez_hv: 'float[:]', rez_hc: 'float[:]',
                                 rez_Z: 'float[:]',
                                 h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', hc_c: 'float[:]', Z_c: 'float[:]',
                                 h_ghost: 'float[:]', hu_ghost: 'float[:]', hv_ghost: 'float[:]', hc_ghost: 'float[:]',
                                 Z_ghost: 'float[:]',
                                 h_halo: 'float[:]', hu_halo: 'float[:]', hv_halo: 'float[:]', hc_halo: 'float[:]',
                                 Z_halo: 'float[:]',
                                 h_x: 'float[:]', h_y: 'float[:]', hx_halo: 'float[:]', hy_halo: 'float[:]',
                                 hc_x: 'float[:]', hc_y: 'float[:]', hcx_halo: 'float[:]', hcy_halo: 'float[:]',
                                 psi: 'float[:]', psi_halo: 'float[:]',
                                 cell_center: 'float[:,:]', face_center: 'float[:,:]', halo_centvol: 'float[:,:]',
                                 face_ghost_id: 'int[:]', ghost_info_flt: 'float[:,:]',
                                 face_cellid: 'int[:,:]', face_measure: 'float[:]', face_normal: 'float[:,:]', face_haloid: 'int[:]',
                                 d_innerfaces: 'int[:]', d_halofaces: 'int[:]', d_boundaryfaces: 'int[:]',
                                 grav: 'float', order: 'int'):
  rez_h[:] = 0.
  rez_hu[:] = 0.
  rez_hv[:] = 0.
  rez_hc[:] = 0.
  rez_Z[:] = 0.

  # from numpy import zeros

  flux = np.zeros(5)
  r_l = np.zeros(2)
  r_r = np.zeros(2)

  for i in d_innerfaces:
    h_l = h_c[face_cellid[i][0]]
    hu_l = hu_c[face_cellid[i][0]]
    hv_l = hv_c[face_cellid[i][0]]
    hc_l = hc_c[face_cellid[i][0]]
    Z_l = Z_c[face_cellid[i][0]]

    normal = face_normal[i]
    mesure = face_measure[i]

    h_r = h_c[face_cellid[i][1]]
    hu_r = hu_c[face_cellid[i][1]]
    hv_r = hv_c[face_cellid[i][1]]
    hc_r = hc_c[face_cellid[i][1]]
    Z_r = Z_c[face_cellid[i][1]]

    center_left = cell_center[face_cellid[i][0]]
    center_right = cell_center[face_cellid[i][1]]

    h_x_left = h_x[face_cellid[i][0]]
    h_x_right = h_x[face_cellid[i][1]]
    h_y_left = h_y[face_cellid[i][0]]
    h_y_right = h_y[face_cellid[i][1]]
    hc_x_left = hc_x[face_cellid[i][0]]
    hc_x_right = hc_x[face_cellid[i][1]]
    hc_y_left = hc_y[face_cellid[i][0]]
    hc_y_right = hc_y[face_cellid[i][1]]

    psi_left = psi[face_cellid[i][0]]
    psi_right = psi[face_cellid[i][1]]

    r_l[0] = face_center[i][0] - center_left[0]
    r_r[0] = face_center[i][0] - center_right[0]
    r_l[1] = face_center[i][1] - center_left[1]
    r_r[1] = face_center[i][1] - center_right[1]

    h_l = h_l + (order - 1) * psi_left * (h_x_left * r_l[0] + h_y_left * r_l[1])
    h_r = h_r + (order - 1) * psi_right * (h_x_right * r_r[0] + h_y_right * r_r[1])

    hc_l = hc_l + (order - 1) * psi_left * (hc_x_left * r_l[0] + hc_y_left * r_l[1])
    hc_r = hc_r + (order - 1) * psi_right * (hc_x_right * r_r[0] + hc_y_right * r_r[1])

    _srnh_scheme(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hc_l, hc_r, Z_l, Z_r, normal, mesure, grav, flux)

    rez_h[face_cellid[i][0]] -= flux[0]
    rez_hu[face_cellid[i][0]] -= flux[1]
    rez_hv[face_cellid[i][0]] -= flux[2]
    rez_hc[face_cellid[i][0]] -= flux[3]
    rez_Z[face_cellid[i][0]] -= flux[4]

    rez_h[face_cellid[i][1]] += flux[0]
    rez_hu[face_cellid[i][1]] += flux[1]
    rez_hv[face_cellid[i][1]] += flux[2]
    rez_hc[face_cellid[i][1]] += flux[3]
    rez_Z[face_cellid[i][1]] += flux[4]

  for i in d_halofaces:
    h_l = h_c[face_cellid[i][0]]
    hu_l = hu_c[face_cellid[i][0]]
    hv_l = hv_c[face_cellid[i][0]]
    hc_l = hc_c[face_cellid[i][0]]
    Z_l = Z_c[face_cellid[i][0]]

    normal = face_normal[i]
    mesure = face_measure[i]
    h_r = h_halo[face_haloid[i]]
    hu_r = hu_halo[face_haloid[i]]
    hv_r = hv_halo[face_haloid[i]]
    hc_r = hc_halo[face_haloid[i]]
    Z_r = Z_halo[face_haloid[i]]

    center_left = cell_center[face_cellid[i][0]]
    center_right = halo_centvol[face_haloid[i]]

    h_x_left = h_x[face_cellid[i][0]]
    h_x_right = hx_halo[face_haloid[i]]
    h_y_left = h_y[face_cellid[i][0]]
    h_y_right = hy_halo[face_haloid[i]]
    hc_x_left = hc_x[face_cellid[i][0]]
    hc_x_right = hcx_halo[face_haloid[i]]
    hc_y_left = hc_y[face_cellid[i][0]]
    hc_y_right = hcy_halo[face_haloid[i]]

    psi_left = psi[face_cellid[i][0]];
    psi_right = psi_halo[face_haloid[i]]

    r_l[0] = face_center[i][0] - center_left[0]
    r_r[0] = face_center[i][0] - center_right[0]
    r_l[1] = face_center[i][1] - center_left[1]
    r_r[1] = face_center[i][1] - center_right[1]

    h_l = h_l + (order - 1) * psi_left * (h_x_left * r_l[0] + h_y_left * r_l[1])
    h_r = h_r + (order - 1) * psi_right * (h_x_right * r_r[0] + h_y_right * r_r[1])

    hc_l = hc_l + (order - 1) * psi_left * (hc_x_left * r_l[0] + hc_y_left * r_l[1])
    hc_r = hc_r + (order - 1) * psi_right * (hc_x_right * r_r[0] + hc_y_right * r_r[1])

    _srnh_scheme(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hc_l, hc_r, Z_l, Z_r, normal, mesure, grav, flux)

    rez_h[face_cellid[i][0]] -= flux[0]
    rez_hu[face_cellid[i][0]] -= flux[1]
    rez_hv[face_cellid[i][0]] -= flux[2]
    rez_hc[face_cellid[i][0]] -= flux[3]
    rez_Z[face_cellid[i][0]] -= flux[4]

  for i in d_boundaryfaces:
    h_l = h_c[face_cellid[i][0]]
    hu_l = hu_c[face_cellid[i][0]]
    hv_l = hv_c[face_cellid[i][0]]
    hc_l = hc_c[face_cellid[i][0]]
    Z_l = Z_c[face_cellid[i][0]]

    normal = face_normal[i]
    mesure = face_measure[i]
    h_r = h_ghost[i]
    hu_r = hu_ghost[i]
    hv_r = hv_ghost[i]
    hc_r = hc_ghost[i]
    Z_r = Z_ghost[i]

    center_left = cell_center[face_cellid[i][0]]
    ghost_id = face_ghost_id[i]
    center_right = ghost_info_flt[ghost_id]

    h_x_left = h_x[face_cellid[i][0]]
    h_y_left = h_y[face_cellid[i][0]]
    hc_x_left = hc_x[face_cellid[i][0]]
    hc_y_left = hc_y[face_cellid[i][0]]

    psi_left = psi[face_cellid[i][0]]

    r_l[0] = face_center[i][0] - center_left[0]
    r_r[0] = face_center[i][0] - center_right[0]
    r_l[1] = face_center[i][1] - center_left[1]
    r_r[1] = face_center[i][1] - center_right[1]

    h_l = h_l + (order - 1) * psi_left * (h_x_left * r_l[0] + h_y_left * r_l[1])
    h_r = h_r

    hc_l = hc_l + (order - 1) * psi_left * (hc_x_left * r_l[0] + hc_y_left * r_l[1])
    hc_r = hc_r

    _srnh_scheme(hu_l, hu_r, hv_l, hv_r, h_l, h_r, hc_l, hc_r, Z_l, Z_r, normal, mesure, grav, flux)

    rez_h[face_cellid[i][0]] -= flux[0]
    rez_hu[face_cellid[i][0]] -= flux[1]
    rez_hv[face_cellid[i][0]] -= flux[2]
    rez_hc[face_cellid[i][0]] -= flux[3]
    rez_Z[face_cellid[i][0]] -= flux[4]


def _term_coriolis_SW(hu_c: 'float[:]', hv_c: 'float[:]', corio_hu: 'float[:]', corio_hv: 'float[:]', f_c: 'float[:]'):
  for i in range(len(hu_c)):
    corio_hu[i] = f_c[i] * hu_c[i]
    corio_hv[i] = -f_c[i] * hv_c[i]


def _term_friction_SW(h_c: 'float[:]', hu_c: 'float[:]', hv_c: 'float[:]', grav: 'float', eta: 'float', time: 'float'):
  nbelement = len(h_c)

  for i in range(nbelement):
    ufric = hu_c[i] / h_c[i]
    vfric = hv_c[i] / h_c[i]
    hfric = h_c[i]

    A = 1 + time * grav * (eta ** 2) * (np.sqrt(ufric ** 2 + vfric ** 2) / (hfric ** (4 / 3)))
    hutild = hu_c[i] / A
    hvtild = hv_c[i] / A

    hu_c[i] = hutild
    hv_c[i] = hvtild


def _term_wind_SW(Tx_wind: 'float[:]', Ty_wind: 'float[:]', wind_wx: 'float[:]', wind_wy: 'float[:]'):
  # Tx_wind -> self.domain.cells.center
  ro_a = 1.25
  nbelements = len(Tx_wind)

  for i in range(nbelements):
    u_wind, v_wind = wind_wx[i], wind_wy[i]
    nor_v_wind = np.sqrt(u_wind ** 2 + v_wind ** 2)
    C_wind = ro_a * (0.75 + 0.067 * nor_v_wind) * 1e-03

    Tx_wind[i] = C_wind * u_wind * nor_v_wind
    Ty_wind[i] = C_wind * v_wind * nor_v_wind

    Tx_wind[i] = C_wind * u_wind * nor_v_wind
    Ty_wind[i] = C_wind * v_wind * nor_v_wind




############################################################################
# FVC (Finite Volume Characteristics) scheme -- Benkhaldoun-Seaid family.
#
# Eigenstructure-FREE flux: instead of a Riemann solver (SRNH's Roe/Cardano,
# which arccos-NaNs at Froude=1), the interface state is obtained by the method
# of characteristics -- a semi-Lagrangian "departure point" back-trace along the
# flow + a half-step predictor carrying the pressure-gradient (acoustic) coupling.
# The physical flux is then evaluated at that predicted state. Robust across the
# sonic point, low-diffusion, and generic (same idea for SW / Euler / MHD).
#
# WELL-BALANCING (C-property): the pressure + bed source reuse the proven Audusse
# hydrostatic reconstruction (hLs) + per-side correction `corr` from the HLLC path
# (machine-zero here), which is INDEPENDENT of the predicted state: at rest u=0 =>
# the advective flux vanishes and only the balanced pressure/corr remains => rest
# is preserved to machine precision. The predictor's pressure forcing uses the
# free-surface gradient d(h+Z)/dn (=0 at rest), so no spurious rest momentum.
#
# Pipeline per step: cell->node interp of (h,hu,hv,hc) -> ValForInterp stencil;
# face gradients of (u,v,eta) via the Diamond scheme; departure() -> predictor()
# -> explicitscheme_convective_SW_fvc(). Stencil geometry is static (built once).
# --------------------------------------------------------------------------- #

def _node_value_for_interpolation_2d(ValForInterp: 'float64[:,:]', w_cell: 'float64[:]', w_node: 'float64[:]', w_ghost: 'float64[:]', w_halo: 'float64[:]', nodefid: 'int32[:,:]', cellfid: 'int32[:,:]', halofid: 'int32[:]', name: 'int32[:]'):
  # 4-point interpolation stencil per face: [node0, node1, cellL, (cellR|halo|ghost)].
  nbfaces = len(nodefid)
  for i in range(nbfaces):
    ValForInterp[i][0] = w_node[nodefid[i][0]]
    ValForInterp[i][1] = w_node[nodefid[i][1]]
    ValForInterp[i][2] = w_cell[cellfid[i][0]]
    if name[i] == 0:
      ValForInterp[i][3] = w_cell[cellfid[i][1]]
    elif name[i] == 10:
      ValForInterp[i][3] = w_halo[halofid[i]]
    else:
      ValForInterp[i][3] = w_ghost[i]


def _weight_parameters_carac_2d(xCenterForInterp: 'float64[:]', yCenterForInterp: 'float64[:]', X0: 'float64', Y0: 'float64'):
  # least-squares weights for a linear-exact interpolation at (X0,Y0) over the 4 points.
  I_xx = 0.; I_yy = 0.; I_xy = 0.; R_x = 0.; R_y = 0.
  for i in range(0, 4):
    Rx = xCenterForInterp[i] - X0
    Ry = yCenterForInterp[i] - Y0
    I_xx += (Rx * Rx)
    I_yy += (Ry * Ry)
    I_xy += (Rx * Ry)
    R_x += Rx
    R_y += Ry
  D = I_xx * I_yy - I_xy * I_xy
  if np.fabs(D) < 1e-30:
    D = 1e-30
  lambda_x = (I_xy * R_y - I_yy * R_x) / D
  lambda_y = (I_xy * R_x - I_xx * R_y) / D
  return R_x, R_y, lambda_x, lambda_y


def _set_carac_field_2d(ValForInterp: 'float64[:]', xCenterForInterp: 'float64[:]', yCenterForInterp: 'float64[:]', X0: 'float64', Y0: 'float64'):
  # linearity-preserving interpolation of a field at the departure point (X0,Y0).
  R_x, R_y, lambda_x, lambda_y = _weight_parameters_carac_2d(xCenterForInterp, yCenterForInterp, X0, Y0)
  w_carac = 0.
  for i in range(0, 4):
    xdiff = xCenterForInterp[i] - X0
    ydiff = yCenterForInterp[i] - Y0
    denom = (4. + lambda_x * R_x + lambda_y * R_y)
    alpha_interp = (1. + lambda_x * xdiff + lambda_y * ydiff) / denom
    w_carac += alpha_interp * ValForInterp[i]
  return w_carac


def _departure_SW_2d(X0: 'float64[:]', Y0: 'float64[:]',
                     hValForInterp: 'float64[:,:]', huValForInterp: 'float64[:,:]', hvValForInterp: 'float64[:,:]',
                     xCenterForInterp: 'float64[:,:]', yCenterForInterp: 'float64[:,:]',
                     centerf: 'float64[:,:]', normal: 'float64[:,:]', mesure: 'float64[:]',
                     dt: 'float64', alphaf: 'float64'):
  # Foot of the characteristic reaching each face: back-trace from the face centre
  # along the (interpolated) normal velocity over alpha*dt.
  nbfaces = len(centerf)
  for i in range(nbfaces):
    X0[i] = centerf[i][0]
    Y0[i] = centerf[i][1]
    h_ed = _set_carac_field_2d(hValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
    if h_ed < 1e-10:
      h_ed = 1e-10
    u_ed = _set_carac_field_2d(huValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i]) / h_ed
    v_ed = _set_carac_field_2d(hvValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i]) / h_ed
    nnx = normal[i][0] / mesure[i]
    nny = normal[i][1] / mesure[i]
    u_n = u_ed * nnx + v_ed * nny
    X0[i] = X0[i] - alphaf * dt * u_n * nnx
    Y0[i] = Y0[i] - alphaf * dt * u_n * nny


def _predictor_SW_2d(h_p: 'float64[:]', hu_p: 'float64[:]', hv_p: 'float64[:]', hc_p: 'float64[:]',
                     hValForInterp: 'float64[:,:]', huValForInterp: 'float64[:,:]', hvValForInterp: 'float64[:,:]', hcValForInterp: 'float64[:,:]',
                     xCenterForInterp: 'float64[:,:]', yCenterForInterp: 'float64[:,:]', X0: 'float64[:]', Y0: 'float64[:]',
                     ugradfacex: 'float64[:]', ugradfacey: 'float64[:]', vgradfacex: 'float64[:]', vgradfacey: 'float64[:]',
                     etagradfacex: 'float64[:]', etagradfacey: 'float64[:]',
                     grav: 'float64', dt: 'float64', alphaf: 'float64', normal: 'float64[:,:]', mesure: 'float64[:]'):
  # Interface state at t^{n+1/2}: material transport (interp at the departure point)
  # + a half characteristic step for the pressure/acoustic coupling. The pressure
  # forcing uses the FREE-SURFACE gradient d(eta)/dn = d(h+Z)/dn (0 at rest -> WB).
  nbfaces = len(h_p)
  for i in range(nbfaces):
    h_ed = _set_carac_field_2d(hValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
    hu_ed = _set_carac_field_2d(huValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
    hv_ed = _set_carac_field_2d(hvValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
    hc_ed = _set_carac_field_2d(hcValForInterp[i], xCenterForInterp[i], yCenterForInterp[i], X0[i], Y0[i])
    if h_ed < 1e-10:
      h_ed = 1e-10
    up = hu_ed / h_ed
    vp = hv_ed / h_ed
    nnx = normal[i][0] / mesure[i]
    nny = normal[i][1] / mesure[i]
    # normal derivative of the normal velocity (compressibility term)
    unx = ugradfacex[i] * nnx + vgradfacex[i] * nny
    uny = ugradfacey[i] * nnx + vgradfacey[i] * nny
    Un_grad = unx * nnx + uny * nny
    # normal derivative of the free surface (pressure-gradient forcing, WB)
    En_grad = etagradfacex[i] * nnx + etagradfacey[i] * nny
    # normal / tangential momentum at the departure state
    hu_n = hu_ed * nnx + hv_ed * nny
    hu_t = hv_ed * nnx - hu_ed * nny
    # half-step characteristic update
    h_p[i] = h_ed * (1.0 - alphaf * dt * Un_grad)
    hu_np = hu_n - alphaf * dt * (hu_n * Un_grad + grav * h_ed * En_grad)
    hu_tp = hu_t - alphaf * dt * (hu_t * Un_grad)
    hc_p[i] = hc_ed * (1.0 - alphaf * dt * Un_grad)
    # rotate the momentum back to (x, y)
    hu_p[i] = hu_np * nnx - hu_tp * nny
    hv_p[i] = hu_np * nny + hu_tp * nnx


def _explicitscheme_convective_SW_fvc(rez_h: 'float[:]', rez_hu: 'float[:]', rez_hv: 'float[:]', rez_hc: 'float[:]',
                                      rez_Z: 'float[:]',
                                      h_p: 'float[:]', hu_p: 'float[:]', hv_p: 'float[:]', hc_p: 'float[:]',
                                      h_c: 'float[:]', Z_c: 'float[:]',
                                      h_ghost: 'float[:]', Z_ghost: 'float[:]',
                                      h_halo: 'float[:]', Z_halo: 'float[:]',
                                      face_cellid: 'int[:,:]', face_measure: 'float[:]', face_normal: 'float[:,:]',
                                      face_haloid: 'int[:]',
                                      d_innerfaces: 'int[:]', d_halofaces: 'int[:]', d_boundaryfaces: 'int[:]',
                                      grav: 'float'):
  # Corrector: physical flux at the predicted interface state (advection) + the
  # Audusse well-balanced hydrostatic pressure/correction (from the CELL states).
  rez_h[:] = 0.
  rez_hu[:] = 0.
  rez_hv[:] = 0.
  rez_hc[:] = 0.
  rez_Z[:] = 0.

  for idx in range(len(d_innerfaces)):
    i = d_innerfaces[idx]
    L = face_cellid[i][0]
    R = face_cellid[i][1]
    nx = face_normal[i][0]
    ny = face_normal[i][1]
    mesure = face_measure[i]
    nnx = nx / mesure
    nny = ny / mesure
    hp = h_p[i]
    if hp < 1e-10:
      hp = 1e-10
    up = hu_p[i] / hp
    vp = hv_p[i] / hp
    cp = hc_p[i] / hp
    qn = hu_p[i] * nnx + hv_p[i] * nny            # normal mass flux (per length)
    # Audusse hydrostatic reconstruction (well-balancing), from the cell states
    dz = Z_c[L] if Z_c[L] > Z_c[R] else Z_c[R]
    hLs = h_c[L] + Z_c[L] - dz
    if hLs < 0.:
      hLs = 0.
    hRs = h_c[R] + Z_c[R] - dz
    if hRs < 0.:
      hRs = 0.
    P = 0.5 * grav * hLs * hLs
    F_h = qn * mesure
    F_hu = (qn * up + P * nnx) * mesure
    F_hv = (qn * vp + P * nny) * mesure
    F_hc = qn * cp * mesure
    corrL = 0.5 * grav * (h_c[L] * h_c[L] - hLs * hLs)
    corrR = 0.5 * grav * (h_c[R] * h_c[R] - hRs * hRs)
    rez_h[L] -= F_h
    rez_h[R] += F_h
    rez_hu[L] -= F_hu + corrL * nx
    rez_hu[R] += F_hu + corrR * nx
    rez_hv[L] -= F_hv + corrL * ny
    rez_hv[R] += F_hv + corrR * ny
    rez_hc[L] -= F_hc
    rez_hc[R] += F_hc

  for idx in range(len(d_halofaces)):
    i = d_halofaces[idx]
    L = face_cellid[i][0]
    k = face_haloid[i]
    nx = face_normal[i][0]
    ny = face_normal[i][1]
    mesure = face_measure[i]
    nnx = nx / mesure
    nny = ny / mesure
    hp = h_p[i]
    if hp < 1e-10:
      hp = 1e-10
    up = hu_p[i] / hp
    vp = hv_p[i] / hp
    cp = hc_p[i] / hp
    qn = hu_p[i] * nnx + hv_p[i] * nny
    dz = Z_c[L] if Z_c[L] > Z_halo[k] else Z_halo[k]
    hLs = h_c[L] + Z_c[L] - dz
    if hLs < 0.:
      hLs = 0.
    P = 0.5 * grav * hLs * hLs
    F_h = qn * mesure
    F_hu = (qn * up + P * nnx) * mesure
    F_hv = (qn * vp + P * nny) * mesure
    F_hc = qn * cp * mesure
    corrL = 0.5 * grav * (h_c[L] * h_c[L] - hLs * hLs)
    rez_h[L] -= F_h
    rez_hu[L] -= F_hu + corrL * nx
    rez_hv[L] -= F_hv + corrL * ny
    rez_hc[L] -= F_hc

  for idx in range(len(d_boundaryfaces)):
    i = d_boundaryfaces[idx]
    L = face_cellid[i][0]
    nx = face_normal[i][0]
    ny = face_normal[i][1]
    mesure = face_measure[i]
    nnx = nx / mesure
    nny = ny / mesure
    hp = h_p[i]
    if hp < 1e-10:
      hp = 1e-10
    up = hu_p[i] / hp
    vp = hv_p[i] / hp
    cp = hc_p[i] / hp
    qn = hu_p[i] * nnx + hv_p[i] * nny
    dz = Z_c[L] if Z_c[L] > Z_ghost[i] else Z_ghost[i]
    hLs = h_c[L] + Z_c[L] - dz
    if hLs < 0.:
      hLs = 0.
    P = 0.5 * grav * hLs * hLs
    F_h = qn * mesure
    F_hu = (qn * up + P * nnx) * mesure
    F_hv = (qn * vp + P * nny) * mesure
    F_hc = qn * cp * mesure
    corrL = 0.5 * grav * (h_c[L] * h_c[L] - hLs * hLs)
    rez_h[L] -= F_h
    rez_hu[L] -= F_hu + corrL * nx
    rez_hv[L] -= F_hv + corrL * ny
    rez_hc[L] -= F_hc


############################################################################
# NOTHING is compiled at import. Call setup(dim) once (uniformly on all MPI
# ranks) before using any kernel below; ShallowWaterSolver does this in __init__.
# The shallow-water kernels are dimension-agnostic, so they are compiled once.
# The nested helper _srnh_scheme is compiled (and rebound to the module global)
# before the kernel that calls it, so numba can resolve the njit->njit call.
_agnostic_done = False

def setup(dim):
  global _agnostic_done
  if dim not in (2, 3):
    raise ValueError(f"Unsupported dimension: {dim}")
  if not _agnostic_done:
    global _srnh_scheme  # nested helper first
    global _weight_parameters_carac_2d, _set_carac_field_2d  # FVC nested helpers first
    global update_SW, time_step_SW, term_source_srnh_SW, explicitscheme_convective_SW
    global term_coriolis_SW, term_friction_SW, term_wind_SW
    global node_value_for_interpolation_2d, departure_SW_2d, predictor_SW_2d, explicitscheme_convective_SW_fvc
    _srnh_scheme = compile(_srnh_scheme)
    update_SW = compile(_update_SW)
    time_step_SW = compile(_time_step_SW)
    term_source_srnh_SW = compile(_term_source_srnh_SW)
    explicitscheme_convective_SW = compile(_explicitscheme_convective_SW)
    term_coriolis_SW = compile(_term_coriolis_SW)
    term_friction_SW = compile(_term_friction_SW)
    term_wind_SW = compile(_term_wind_SW)
    # FVC pipeline (nested helpers compiled and rebound before their callers)
    _weight_parameters_carac_2d = compile(_weight_parameters_carac_2d)
    _set_carac_field_2d = compile(_set_carac_field_2d)
    node_value_for_interpolation_2d = compile(_node_value_for_interpolation_2d)
    departure_SW_2d = compile(_departure_SW_2d)
    predictor_SW_2d = compile(_predictor_SW_2d)
    explicitscheme_convective_SW_fvc = compile(_explicitscheme_convective_SW_fvc)
    _agnostic_done = True

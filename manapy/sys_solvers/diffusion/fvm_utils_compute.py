from manapy.backends.compile_fun import compile
import numpy as np

def _explicitscheme_dissipative(wx_face: 'float[:]', wy_face: 'float[:]', wz_face: 'float[:]',
                               face_cellid: 'int[:,:]', face_normal: 'float[:,:]', face_name: 'int[:]',
                               dissip_w: 'float[:]', Dxx: 'float', Dyy: 'float', Dzz: 'float'):
  nbface = len(face_cellid)
  norm = np.zeros(3)
  dissip_w[:] = 0.

  for i in range(nbface):

    norm[:] = face_normal[i][:]
    q = Dxx * wx_face[i] * norm[0] + Dyy * wy_face[i] * norm[1] + Dzz * wz_face[i] * norm[2]

    flux_w = q

    if face_name[i] == 0:

      dissip_w[face_cellid[i][0]] += flux_w
      dissip_w[face_cellid[i][1]] -= flux_w

    else:
      dissip_w[face_cellid[i][0]] += flux_w

def _time_step(u: 'float[:]', v: 'float[:]', w: 'float[:]', cfl: 'float', face_normal: 'float[:,:]',
               face_measure: 'float[:]', cell_volume: 'float[:]', cell_faceid: 'int[:,:]', dim: 'int',
               Dxx: 'float', Dyy: 'float', Dzz: 'float'):
  nbelement = len(cell_faceid)
  norm = np.zeros(3)
  dt = 1e6
  for i in range(nbelement):
    lam = 0.

    for j in range(cell_faceid[i][-1]):
      norm[:] = face_normal[cell_faceid[i][j]][:]

      mes = np.sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2])
      lam_diff = Dxx * mes ** 2 + Dyy * mes ** 2 + Dzz * mes ** 2
      lam += lam_diff / cell_volume[i]

    if lam != 0:
      dt = min(dt, cfl * cell_volume[i] / lam)

  return dt

def _update_new_value(ne_c: 'float[:]', rez_ne: 'float[:]', dissip_ne: 'float[:]', src_ne: 'float[:]',
                      dtime: 'float', cell_volume: 'float[:]'):
  nbelements = len(ne_c)
  for i in range(nbelements):
    ne_c[i] += dtime * ((rez_ne[i] + dissip_ne[i]) / cell_volume[i] + src_ne[i])


############################################################################
# Public
explicitscheme_dissipative = compile(_explicitscheme_dissipative)
time_step = compile(_time_step)
update_new_value = compile(_update_new_value)
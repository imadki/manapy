from manapy.backends.compile_fun import FunObj
import numpy as np

def _initialisation_gaussian_2d(ne: 'float[:]', u: 'float[:]', v: 'float[:]', P: 'float[:]', cell_center: 'float[:, :]', Pinit: 'float'):
  nbelements = len(cell_center)

  sigma = 0.05
  for i in range(nbelements):
    xcent = cell_center[i][0]
    ycent = cell_center[i][1]

    ne[i] = 5 * np.exp(-1. * ((xcent - 0.) ** 2 + (ycent - 0.2) ** 2) / sigma ** 2) + 1
    u[i] = 0.
    v[i] = 0.
    P[i] = Pinit * (.5 - xcent)


def _initialisation_gaussian_3d(ne: 'float[:]', u: 'float[:]', v: 'float[:]', w: 'float[:]', P: 'float[:]', cell_center: 'float[:, :]', Pinit: 'float'):
  nbelements = len(cell_center)

  sigma = 0.05
  for i in range(nbelements):
    xcent = cell_center[i][0]
    ycent = cell_center[i][1]
    zcent = cell_center[i][2]

    ne[i] = 5 * np.exp(-1. * ((xcent - 0.2) ** 2 + (ycent - 0.25) ** 2 + (zcent - 0.45) ** 2) / sigma ** 2) + 1
    u[i] = 0.
    v[i] = 0.
    w[i] = 0.
    P[i] = Pinit * (.5 - xcent)

############################################################################
# Public: compiled lazily on first call (only the dimension actually used).
initialisation_gaussian_2d = FunObj(_initialisation_gaussian_2d)
initialisation_gaussian_3d = FunObj(_initialisation_gaussian_3d)



import numpy as np
from manapy.backends.compile_fun import FunObj


def _initialisation_streamer_2d(ne: 'float[:]', ni: 'float[:]', u: 'float[:]', v: 'float[:]', Ex: 'float[:]', Ey: 'float[:]',
                                P: 'float[:]', cell_center: 'float[:,:]', Pinit: 'float'):
  nbelements = len(cell_center)
  sigma = 0.01

  for i in range(nbelements):
    xcent = cell_center[i][0]
    ycent = cell_center[i][1]

    ne[i] = 1e16 * np.exp(-1. * ((xcent - 0.2) ** 2 + (ycent - 0.25) ** 2) / sigma ** 2) + 1e09
    ni[i] = 1e16 * np.exp(-1. * ((xcent - 0.2) ** 2 + (ycent - 0.25) ** 2) / sigma ** 2) + 1e09
    u[i] = 0.
    v[i] = 0.
    P[i] = Pinit * (1. - xcent)
    Ex[i] = 0.
    Ey[i] = 0.


def _initialisation_streamer_3d(ne: 'float[:]', ni: 'float[:]', u: 'float[:]', v: 'float[:]', w: 'float[:]', Ex: 'float[:]',
                                Ey: 'float[:]', Ez: 'float[:]', P: 'float[:]', cell_center: 'float[:,:]', Pinit: 'float'):
  nbelements = len(cell_center)
  sigma = 0.01

  for i in range(nbelements):
    xcent = cell_center[i][0]
    ycent = cell_center[i][1]
    zcent = cell_center[i][2]

    ne[i] = 1e16 * np.exp(-1. * ((xcent - 0.2) ** 2 + (ycent - 0.25) ** 2 + (zcent - 0.25) ** 2) / sigma ** 2) + 1e12
    ni[i] = 1e16 * np.exp(-1. * ((xcent - 0.2) ** 2 + (ycent - 0.25) ** 2 + (zcent - 0.25) ** 2) / sigma ** 2) + 1e12
    u[i] = 0.
    v[i] = 0.
    w[i] = 0.
    P[i] = Pinit * (.5 - xcent)
    Ex[i] = 0.
    Ey[i] = 0.
    Ez[i] = 0.


############################################################################
# Public: compiled lazily on first call (only the dimension actually used).
initialisation_streamer_2d = FunObj(_initialisation_streamer_2d)
initialisation_streamer_3d = FunObj(_initialisation_streamer_3d)

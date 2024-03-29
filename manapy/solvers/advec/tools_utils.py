from numpy import sqrt, fabs, zeros, exp
from numba import njit

#@njit
def initialisation_gaussian_2d(ne,  u, v, P, center, Pinit):

    nbelements = len(center)
    
    sigma = 0.05
    for i in range(nbelements):
        xcent = center[i][0]
        ycent = center[i][1]
        
        ne[i] = 5 * exp(-1.*((xcent-0.2)**2 + (ycent-0.2)**2) / sigma**2) + 1
        u[i]  = 0.
        v[i]  = 0.
        P[i]  = Pinit * (.5 - xcent)
#@njit
def initialisation_gaussian_3d(ne,  u, v, w, P, center, Pinit): 


    nbelements = len(center)
    
    sigma = 0.05
    for i in range(nbelements):
        xcent = center[i][0]
        ycent = center[i][1]
        zcent = center[i][2]
        
        ne[i] = 5 * exp(-1.*((xcent-0.2)**2 + (ycent-0.25)**2 +  (zcent-0.45)**2) / sigma**2) + 1
        u[i]  = 0.
        v[i]  = 0.
        w[i]  = 0.
        P[i]  = Pinit * (.5 - xcent)



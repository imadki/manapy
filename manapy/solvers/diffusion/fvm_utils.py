from numpy import int32,  uint32
from numba import njit, prange
import numpy as np

def explicitscheme_dissipative(wx_face:'float[:]',  wy_face:'float[:]', wz_face:'float[:]', 
                               cellidf:'int32[:,:]', normalf:'float[:,:]', namef:'uint32[:]', 
                               dissip_w:'float[:]', Dxx:'float', Dyy:'float', Dzz:'float'):
     
    nbface = len(cellidf)
    norm = np.zeros(3)
    dissip_w[:] = 0.

    for i in range(nbface):
        
        norm[:] = normalf[i][:]
        q = Dxx * wx_face[i] * norm[0] + Dyy * wy_face[i] * norm[1] + Dzz * wz_face[i] * norm[2]

        flux_w = q

        if namef[i] == 0:

            dissip_w[cellidf[i][0]] += flux_w
            dissip_w[cellidf[i][1]] -= flux_w

        else:
            dissip_w[cellidf[i][0]] += flux_w


  
def time_step(u:'float[:]', v:'float[:]', w:'float[:]',  cfl:'float', normal:'float[:,:]', 
              mesure:'float[:]', volume:'float[:]', faceid:'int32[:,:]', dim:'int32',
              Dxx:'float', Dyy:'float', Dzz:'float'):
   
    nbelement =  len(faceid)
    norm = np.zeros(3)
    dt = 1e6
    for i in range(nbelement):
        lam = 0.
       
        for j in range(faceid[i][-1]):
            norm[:] = normal[faceid[i][j]][:]
            
            mes = np.sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2])
            lam_diff = Dxx * mes**2 + Dyy * mes**2 + Dzz * mes**2
            lam += lam_diff/volume[i]
        
        dt  = min(dt, cfl * volume[i]/lam)
     
    return dt

def update_new_value(ne_c:'float[:]', rez_ne:'float[:]', dissip_ne:'float[:]',  src_ne:'float[:]',
                     dtime:'float', vol:'float[:]'):
    nbelements = len(ne_c)
    for i in range(nbelements):
        ne_c[i]  += dtime  * ((rez_ne[i]  +  dissip_ne[i]) /vol[i] + src_ne[i] )

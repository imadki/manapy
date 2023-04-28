from numba import njit

@njit
def initialisation_SW(h:'float[:]', hu:'float[:]', hv:'float[:]', hc:'float[:]', Z:'float[:]', center:'float[:,:]'):
   
    nbelements = len(center)
    
    for i in range(nbelements):
        xcent = center[i][0]
        h[i] = 2
        Z[i]  = 0.
        
        if xcent < .5:
            h[i] = 5.
            
        hu[i] = 0.
        hv[i] = 0.
        hc[i] = 0.
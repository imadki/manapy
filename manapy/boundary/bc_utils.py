#from numpy import np.zeros, int32, np.array, np.sqrt, float, uint32
from numba import njit
import numpy as np

@njit("float64(float64[:], float64[:])", nogil=True)
def distance(x:'float[:]', y:'float[:]'):
   
    z = np.zeros(2)
    z[0] = x[0] - y[0]
    z[1] = x[1] - y[1]
    
    return np.sqrt(z[0]**2 + z[1]**2) 

def rhs_value_dirichlet_node(Pbordnode:'float[:]', nodes:'uint32[:]', value:'float[:]'):
    
    for i in nodes:
        Pbordnode[i] = value[i]
    
def rhs_value_dirichlet_face(Pbordface:'float[:]', faces:'uint32[:]', value:'float[:]'):
    
    for i in faces:
        Pbordface[i] = value[i]
        
def rhs_value_neumannNH_face(w_c:'float[:]', Pbordface:'float[:]', cellid:'int32[:,:]', faces:'uint32[:]',
                          cst:'float[:]', dist:'float[:]'):
    for i in faces:
        val = w_c[cellid[i][0]] + cst[i]*dist[i]
        Pbordface[i]  = (val + w_c[cellid[i][0]])/2.

#def ghost_value_slip(u_c:'float[:]', v_c:'float[:]', w_ghost:'float[:]', 
#                     cellid:'int32[:,:]', faces:'int32[:]', normal:'float[:,:]', mesure:'float[:]'):
#    
#    s_n = np.zeros(3)
#   
#    for i in faces:
#        
#        u_i = u_c[cellid[i][0]]
#        v_i = v_c[cellid[i][0]]
#        
#        s_n[:] = normal[i][:] / mesure[i]
#        u_g = u_i*(s_n[1]*s_n[1] - s_n[0]*s_n[0]) - 2.0*v_i*s_n[0]*s_n[1]
#        
#        w_ghost[i] = u_c[cellid[i][0]] * u_g

def ghost_value_nonslip(w_c:'float[:]', w_ghost:'float[:]', cellid:'int32[:,:]', faces:'uint32[:]',
                        cst:'float[:]', dist:'float[:]'):
    
    for i in faces:
        w_ghost[i]  = -1*w_c[cellid[i][0]]

def ghost_value_neumann(w_c:'float[:]', w_ghost:'float[:]', cellid:'int32[:,:]', faces:'uint32[:]',
                        cst:'float[:]', dist:'float[:]'):
    
    for i in faces:
        w_ghost[i]  = w_c[cellid[i][0]]
        
        
def ghost_value_neumannNH(w_c:'float[:]', w_ghost:'float[:]', cellid:'int32[:,:]', faces:'uint32[:]',
                          cst:'float[:]', dist:'float[:]'):
    
    for i in faces:
        w_ghost[i] = w_c[cellid[i][0]] + cst[i]*dist[i]

def ghost_value_dirichlet(value:'float[:]', w_ghost:'float[:]', cellid:'int32[:,:]', faces:'uint32[:]',
                          cst:'float[:]', dist:'float[:]'):
    
    for i in faces:
        w_ghost[i]  = value[i]

def haloghost_value_neumann(w_halo:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                            BCindex: 'int32', halonodes:'uint32[:]',  cst:'float[:]'):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellhalo  = np.int32(haloghostcenter[i][j][-3])
                    cellghost = np.int32(haloghostcenter[i][j][-1])
    
                    w_haloghost[cellghost]   = w_halo[cellhalo]

def haloghost_value_neumannNH(w_halo:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                            BCindex: 'int32', halonodes:'uint32[:]',  cst:'float[:]'):
    
    #TODO dist is not well computed (work only if NH is in the infaces)
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellhalo  = np.int32(haloghostcenter[i][j][-3])
                    cellghost = np.int32(haloghostcenter[i][j][-1])
                    dist = 2*distance(haloghostcenter[i][j][0:2], np.array([0., haloghostcenter[i][j][1]]))
    
                    w_haloghost[cellghost]   = w_halo[cellhalo] + cst[i]*dist

def haloghost_value_dirichlet(value:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                              BCindex: 'int32', halonodes:'uint32[:]',  cst:'float[:]'):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellghost = np.int32(haloghostcenter[i][j][-1])
                    w_haloghost[cellghost]   = value[cellghost]

def haloghost_value_nonslip(w_halo:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
                           BCindex: 'int32', halonodes:'uint32[:]',  cst:'float[:]'):
    
    for i in halonodes:
        for j in range(len(haloghostcenter[i])):
            if haloghostcenter[i][j][-1] != -1:
                if haloghostcenter[i][j][-2] == BCindex:
                    cellghost = np.int32(haloghostcenter[i][j][-1])
                    w_haloghost[cellghost]   = -1*w_halo[cellghost]


#def haloghost_value_slip(u_halo:'float[:]', v_halo:'float[:]', w_haloghost:'float[:]', haloghostcenter:'float[:,:,:]',
#                         BCindex: 'int32', halonodes:'int32[:]', haloghostfaceinfo:'float[:,:,:]'):
#    
#    from numpy import np.sqrt, np.zeros
#
#    s_n = np.zeros(2)
#    for i in halonodes:
#        for j in range(len(haloghostcenter[i])):
#            if haloghostcenter[i][j][-1] != -1:
#                if haloghostcenter[i][j][-2] == BCindex:
#                    cellghost = int32(haloghostcenter[i][j][-1])
#
#                    u_i = u_halo[cellghost]
#                    v_i = v_halo[cellghost]
#                    
#                    mesure = np.sqrt(haloghostfaceinfo[i][j][2]**2 + haloghostfaceinfo[i][j][3]**2)# + haloghostfaceinfo[i][5]**2)
#                    
#                    s_n[0] = haloghostfaceinfo[i][j][2] / mesure
#                    s_n[1] = haloghostfaceinfo[i][j][3] / mesure
#                    
#                    u_g = u_i*(s_n[1]*s_n[1] - s_n[0]*s_n[0]) - 2.0*v_i*s_n[0]*s_n[1]
#                        
#                    w_haloghost[i] = u_halo[cellghost] * u_g



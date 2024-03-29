from numpy import float, int32
from numba import njit

def convert_solution(x1:'float[:]', x1converted:'float[:]', tc:'int32[:]', b0Size:'int32'):
    for i in range(b0Size):
        x1converted[i] = x1[tc[i]]
        
def facetocell(u_face:'float[:]', u_c:'float[:]', faceidc:'int32[:,:]', dim:'int32'):
  
    nbelements = len(u_c)
    u_c[:] = 0.
    
    for i in range(nbelements):
        for j in range(faceidc[i][-1]):
            u_c[i]  += u_face[faceidc[i][j]]
    
    for i in range(nbelements):
        u_c[i]  /= faceidc[i][-1]

def celltoface(u_cell:'float[:]', u_face:'float[:]', u_ghost:'float[:]', u_halo:'float[:]',
               cellid:'int32[:,:]', halofid:'int32[:]',
               innerfaces:'int32[:]', boundaryfaces:'int32[:]', halofaces:'int32[:]'):
    
    for i in innerfaces:
        c1 = cellid[i][0]
        c2 = cellid[i][1]
        u_face[i] = .5*(u_cell[c1] + u_cell[c2])
    
    for i in halofaces:
        c1 = cellid[i][0]
        u_face[i] = .5*(u_cell[c1] + u_halo[halofid[i]])
        
    for i in boundaryfaces:
        c1 = cellid[i][0]
        u_face[i] = .5*(u_cell[c1] + u_ghost[i])
        
        
@njit#('int32(int32[:], int32)')
def search_element(a, target_value):
    find = 0
    for val in a:
        if val == target_value:
            find = 1
            break
    return find

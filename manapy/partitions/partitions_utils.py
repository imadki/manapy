import numpy as np
from numpy import uint32, int32

from manapy.ddm.ddm_utils3d import split_to_tetra
from manapy.ddm.ddm_utils2d import split_to_triangle


def define_ghost_node(mesh, periodic, nbnodes, dim):
    ghost_nodes = np.zeros(nbnodes, dtype=np.int32)
    
    if dim == 2:
        typefaces = ["line"]
    elif dim == 3:
        typefaces = ["quad", "triangle"]
    
    ghost = {}
    if type(mesh.cells) == dict:
        for i, j in mesh.cell_data.items():
            if i in typefaces:
                ghost[i] = j.get('gmsh:physical')

        for i, j in mesh.cells.items():
            if i in typefaces:
                for k in range(len(ghost[i])):
                    for index in range(len(j[k])):
                        if ghost[i][k] == 1 or ghost[i][k] == 2 :#or ghost[k] == 5 or ghost[k] == 6 :
                            ghost_nodes[j[k][index]] = int(ghost[i][k])
                            
            if i in typefaces:
                for k in range(len(ghost[i])):
                    for index in range(len(j[k])):
                        if ghost_nodes[j[k][index]] != 1 and ghost_nodes[j[k][index]] !=2:
                            if ghost[i][k] == 3 or ghost[i][k] == 4 :#or ghost[k] == 5 or ghost[k] == 6 :
                                ghost_nodes[j[k][index]] = int(ghost[i][k])
            
        for i, j in mesh.cells.items():
            if i in typefaces:
                for k in range(len(ghost[i])):
                    for index in range(len(j[k])):
                        if ghost_nodes[j[k][index]] == 0:
                            ghost_nodes[j[k][index]] = int(ghost[i][k])
    
    if periodic[0] == 1:
        for i in range(len(ghost_nodes)):
            if ghost_nodes[i] == 1:
                ghost_nodes[i] = 11
            elif ghost_nodes[i] == 2:
                ghost_nodes[i] = 22
                
    if periodic[1] == 1:
        for i in range(len(ghost_nodes)):
            if ghost_nodes[i] == 3:
                ghost_nodes[i] = 33
            elif ghost_nodes[i] == 4:
                ghost_nodes[i] = 44
                
    if periodic[2] == 1:
        for i in range(len(ghost_nodes)):
            if ghost_nodes[i] == 5:
                ghost_nodes[i] = 55
            elif ghost_nodes[i] == 6:
                ghost_nodes[i] = 66

    return ghost_nodes
        
def convert_2d_cons_to_array(ele1:'uint32[:,:]', ele2:'uint32[:,:]'):
    nbelements = 0
    if len(ele1) > 1:
        nbelements += len(ele1)
        l = 3
        
    if len(ele2) > 1:
        nbelements += len(ele2)
        l = 4
    
    padded_l = np.zeros((nbelements, l+1), dtype=np.int32)
    
    if len(ele1) > 1:
        for i in range(len(ele1)):
            padded_l[i][0:3] = ele1[i][0:3]
            padded_l[i][-1]  = 3
            
        for i in range(len(ele1), nbelements):
            padded_l[i][0:4] = ele2[i-len(ele1)]
            padded_l[i][-1]  = 4
            
    else:
         for i in range(nbelements):
            padded_l[i][0:4] = ele2[i]
            padded_l[i][-1]  = 4
        
    return padded_l


def convert_3d_cons_to_array(ele1:'uint32[:,:]', ele2:'uint32[:,:]', ele3:'uint32[:,:]'):
    nbelements = 0
    
    #tetra
    if len(ele1) > 1:
        nbelements += len(ele1)
        l = 4
    
    #pyramid
    if len(ele2) > 1:
        nbelements += len(ele2)
        l = 5
    
    #hexa
    if len(ele3) > 1:
        nbelements += len(ele3)
        l = 8
    
    padded_l = np.zeros((nbelements, l+1), dtype=np.int32)
    
    if len(ele1) > 1:
        for i in range(len(ele1)):
            padded_l[i][0:4] = ele1[i][0:4]
            padded_l[i][-1]  = 4
        
        for i in range(len(ele1), nbelements-len(ele3)):
            padded_l[i][0:5] = ele2[i-len(ele1)]
            padded_l[i][-1]  = 5
            
        for i in range(len(ele1)+len(ele2), nbelements):
            padded_l[i][0:8] = ele3[i-(len(ele1)+len(ele2))]
            padded_l[i][-1]  = 8
            
    elif len(ele2) > 1:
         for i in range(nbelements):
            padded_l[i][0:5] = ele2[i]
            padded_l[i][-1]  = 5
            
    elif len(ele3) > 1:
         for i in range(nbelements):
            padded_l[i][0:8] = ele3[i]
            padded_l[i][-1]  = 8
        
    return padded_l

def create_npart_cpart(cell_nodeid:'uint32[:,:]', npart:'uint32[:]', 
                       epart:'uint32[:]', nbnodes:'int32', nbelements:'int32',
                       SIZE:'int32'):
    
    npart         = [ [i ] for i in npart ]
    cpart         = [ [i ] for i in epart ]
    neighsub      = [[i for i in range(0)]  for i in range(SIZE)]
    globcelltoloc = [[i for i in range(0)]  for i in range(SIZE)]
    locnodetoglob = [[i for i in range(0)]  for i in range(SIZE)]
    halo_cellid   = [[i for i in range(0)]  for i in range(SIZE)]
    
    for i in range(nbelements):
        for j in range(cell_nodeid[i][-1]):
            k = cell_nodeid[i][j]
            if epart[i] not in npart[k]:
                npart[k].append(epart[i])
            locnodetoglob[epart[i]].append(k)
        globcelltoloc[epart[i]].append(i)
    
    for i in range(nbelements):
        for j in range(cell_nodeid[i][-1]):
            for k in range(len(npart[cell_nodeid[i][j]])):
                if npart[cell_nodeid[i][j]][k] not in cpart[i]:
                    cpart[i].append(npart[cell_nodeid[i][j]][k])
                    
    maxnpart = 0
    for i in range(nbnodes):
        maxnpart = max(maxnpart, len(npart[i]))
        for j in range(len(npart[i])):
            neighsub[npart[i][j]].extend(npart[i])
    
    
    for i in range(SIZE):
        for cell in globcelltoloc[i]:
            # check if the cell's partition belong to two subdomains
            if len(cpart[cell]) > 1:
                # append the cell's partition to halo_cellid list
                halo_cellid[i].append(cell)
    
    
    npartnew = -1*np.ones((nbnodes, maxnpart+2), dtype=np.int32)
    #convert npart to array                    
    for i in range(nbnodes):
        for j in range(len(npart[i])):
            npartnew[i][j] = npart[i][j]
        npartnew[i][-2] = len(npart[i])
        npartnew[i][-1] = i
        
    tc = np.zeros(nbelements, dtype=np.uint32)
    cmpt = 0
    for i in range(SIZE):
        for j in range(len(globcelltoloc[i])):
            tc[cmpt] = globcelltoloc[i][j]
            cmpt += 1
    
    return npartnew, cpart, neighsub, halo_cellid, globcelltoloc, locnodetoglob, tc


def compute_halocell(halo_cellid, cpart, cell_nodeid, nodes, neighsub, SIZE, dim, precision ):
    haloint = {}
    for i in range(SIZE):
        for cell in halo_cellid[i]:
            for k in range(len(cpart[cell])):
                if i != cpart[cell][k]:
                    haloint.setdefault((i, cpart[cell][k]), []).append(cell)
    centvol = [[] for i in range(SIZE)]
    haloextloc = [[] for i in range(SIZE)]
    halointloc = [[] for i in range(SIZE)]
    halointlen = [[] for i in range(SIZE)]
    
    if dim == 2:
        vertices = np.zeros((4, 2))
        triangles = np.zeros((4, 3, 2))
        for i in range(SIZE):
            for j in range(len(neighsub[i])):
                halointloc[i].extend(haloint[(i, neighsub[i][j])])
                halointlen[i].append(len(haloint[(i, neighsub[i][j])]))
                for k in range(len(haloint[(neighsub[i][j], i)])):
                    
                    if cell_nodeid[haloint[(neighsub[i][j], i)][k]][-1] == 3:
                        
                        s_1 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][0]
                        s_2 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][1]
                        s_3 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][2]
                        
                        x_1 = nodes[s_1][0]; y_1 = nodes[s_1][1]
                        x_2 = nodes[s_2][0]; y_2 = nodes[s_2][1]
                        x_3 = nodes[s_3][0]; y_3 = nodes[s_3][1]
            
                        centvol[i].append([1./3 * (x_1 + x_2 + x_3), 1./3*(y_1 + y_2 + y_3), 0.,
                                           (1./2) * abs((x_1-x_2)*(y_1-y_3)-(x_1-x_3)*(y_1-y_2))])
                        
                        haloextloc[i].append([haloint[(neighsub[i][j], i)][k], s_1, s_2, s_3, -1, 4])
                        
                    elif cell_nodeid[haloint[(neighsub[i][j], i)][k]][-1] == 4:
                        
                        s_1 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][0]
                        s_2 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][1]
                        s_3 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][2]
                        s_4 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][3]
                        
                        vertices[:,:] = np.vstack((nodes[s_1][:dim], nodes[s_2][:dim], 
                                                   nodes[s_3][:dim], nodes[s_4][:dim]))
                        split_to_triangle(vertices, triangles)
                        
                        center = triangles[0][2]
                        volume = 0.
                        for triangle in triangles:
                            x1 = triangle[0][0]; y1 = triangle[0][1]
                            x2 = triangle[1][0]; y2 = triangle[1][1]
                            x3 = triangle[2][0]; y3 = triangle[2][1]
                            volume += (1./2) * abs((x1-x2)*(y1-y3)-(x1-x3)*(y1-y2))
            
                        centvol[i].append([center[0], center[1], 0., volume])
                        haloextloc[i].append([haloint[(neighsub[i][j], i)][k], s_1, s_2, s_3, s_4, 5])
                        
                        
    if dim == 3:
        wedge = np.zeros(3, dtype=precision)
        u = np.zeros(3, dtype=precision)
        v = np.zeros(3, dtype=precision)
        w = np.zeros(3, dtype=precision)
        
        #arrays for split tetra
        vertices1 = np.zeros((8, 3))
        vertices2 = np.zeros((5, 3))
        tetrahedra1 = np.zeros((8, 4, 3))
        tetrahedra2 = np.zeros((5, 4, 3))
        
        for i in range(SIZE):
            for j in range(len(neighsub[i])):
                halointloc[i].extend(haloint[(i, neighsub[i][j])])
                halointlen[i].append(len(haloint[(i, neighsub[i][j])]))
                for k in range(len(haloint[(neighsub[i][j], i)])):
                    if cell_nodeid[haloint[(neighsub[i][j], i)][k]][-1] == 4:
                    
                        s_1 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][0]
                        s_2 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][1]
                        s_3 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][2]
                        s_4 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][3]
                        
                        x_1 = nodes[s_1][0]; y_1 = nodes[s_1][1]; z_1 = nodes[s_1][2]
                        x_2 = nodes[s_2][0]; y_2 = nodes[s_2][1]; z_2 = nodes[s_2][2] 
                        x_3 = nodes[s_3][0]; y_3 = nodes[s_3][1]; z_3 = nodes[s_3][2] 
                        x_4 = nodes[s_4][0]; y_4 = nodes[s_4][1]; z_4 = nodes[s_4][2]
            
                        u[:] = nodes[s_2][0:3]- nodes[s_1][0:3]
                        v[:] = nodes[s_3][0:3]- nodes[s_1][0:3]
                        w[:] = nodes[s_4][0:3]- nodes[s_1][0:3]
                        
                        wedge[0] = v[1]*w[2] - v[2]*w[1]
                        wedge[1] = v[2]*w[0] - v[0]*w[2]
                        wedge[2] = v[0]*w[1] - v[1]*w[0]
                        
                        volume = 1./6*np.fabs(u[0]*wedge[0] + u[1]*wedge[1] + u[2]*wedge[2]) 
                        
                        centvol[i].append([1./4 * (x_1 + x_2 + x_3 + x_4), 1./4*(y_1 + y_2 + y_3 + y_4), 
                                           1./4*(z_1 + z_2 + z_3 + z_4), 
                                           volume])
    
                        haloextloc[i].append([haloint[(neighsub[i][j], i)][k], s_1, s_2, s_3, s_4, -1, -1, -1, -1,  5])
                        
                    elif cell_nodeid[haloint[(neighsub[i][j], i)][k]][-1] == 5:
            
                        s_1 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][0]
                        s_2 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][1]
                        s_3 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][2]
                        s_4 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][3]
                        s_5 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][4]
                        
                        vertices2[:,:] = np.vstack((nodes[s_1][:dim], nodes[s_2][:dim], 
                                                    nodes[s_3][:dim], nodes[s_4][:dim],
                                                    nodes[s_5][:dim]))
                        
                        split_to_tetra(vertices2, tetrahedra2)
                        
                        volume = 0.
                        for tetrahedron in tetrahedra2:
                            u[:] = tetrahedron[1]-tetrahedron[0]
                            v[:] = tetrahedron[2]-tetrahedron[0]
                            w[:] = tetrahedron[3]-tetrahedron[0]
                            
                            wedge[0] = v[1]*w[2] - v[2]*w[1]
                            wedge[1] = v[2]*w[0] - v[0]*w[2]
                            wedge[2] = v[0]*w[1] - v[1]*w[0]
                            
                            center = tetrahedron[-1]
                            volume += 1./6*np.fabs(u[0]*wedge[0] + u[1]*wedge[1] + u[2]*wedge[2]) 
                            
                        centvol[i].append([center[0], center[1], center[2], volume])
                        haloextloc[i].append([haloint[(neighsub[i][j], i)][k], s_1, s_2, s_3, s_4, s_5, -1, -1, -1,  6])
                        
                    elif cell_nodeid[haloint[(neighsub[i][j], i)][k]][-1] == 8:
                        
                        s_1 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][0]
                        s_2 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][1]
                        s_3 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][2]
                        s_4 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][3]
                        s_5 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][4]
                        s_6 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][5]
                        s_7 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][6]
                        s_8 = cell_nodeid[haloint[(neighsub[i][j], i)][k]][7]
                        
                        vertices1[:,:] = np.vstack((nodes[s_1][:dim], nodes[s_2][:dim], 
                                                    nodes[s_3][:dim], nodes[s_4][:dim],
                                                    nodes[s_5][:dim], nodes[s_6][:dim],
                                                    nodes[s_7][:dim], nodes[s_8][:dim]))
                        
                        split_to_tetra(vertices1, tetrahedra1)
                        
                        volume = 0.
                        for tetrahedron in tetrahedra1:
                            u[:] = tetrahedron[1]-tetrahedron[0]
                            v[:] = tetrahedron[2]-tetrahedron[0]
                            w[:] = tetrahedron[3]-tetrahedron[0]
                            
                            wedge[0] = v[1]*w[2] - v[2]*w[1]
                            wedge[1] = v[2]*w[0] - v[0]*w[2]
                            wedge[2] = v[0]*w[1] - v[1]*w[0]
                            
                            center = tetrahedron[-1]
                            volume += 1./6*np.fabs(u[0]*wedge[0] + u[1]*wedge[1] + u[2]*wedge[2]) 
                        
                        centvol[i].append([center[0], center[1], center[2], volume])
                        haloextloc[i].append([haloint[(neighsub[i][j], i)][k], s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, 9])
    
    return centvol, haloextloc, halointloc, halointlen
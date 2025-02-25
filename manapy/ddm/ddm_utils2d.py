import numpy as np
from numpy import int32, uint32, int64
from numba import njit

from mpi4py import MPI

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()

###############################################################################
def create_cellsOfFace(faceid:'int32[:,:]', nbelements:'int32', nbfaces:'int32', cellid:'int32[:,:]', maxcellfid:'int32'):
    for i in range(nbelements):
        for j in range(faceid[i][-1]):
            if cellid[faceid[i][j]][0] == -1 :
                cellid[faceid[i][j]][0] = i

            if cellid[faceid[i][j]][0] != i:
                cellid[faceid[i][j]][0] = cellid[faceid[i][j]][0]
                cellid[faceid[i][j]][1] = i

def create_cell_faceid(nbelements:'int32', oldTonewIndex:'int64[:]', cellf:'int32[:,:]', faceid:'int32[:,:]', maxcellfid:'int32'):

    for i in range(nbelements):
        for j in range(maxcellfid):
            if cellf[i][j] != -1:
                faceid[i][j] = oldTonewIndex[cellf[i][j]]

        faceid[i][-1] = len(faceid[i][faceid[i] !=-1])


def create_NeighborCellByFace(faceid:'int32[:,:]', cellid:'int32[:,:]', nbelements:'int32', maxcellfid:'int32'):
    
    cellfid = [[i for i in range(0)] for i in range(nbelements)]
    #Création des 3/4 triangles voisins par face
    for i in range(nbelements):
        for j in range(faceid[i][-1]):
            f = faceid[i][j]
            if cellid[f][1] != -1:
                if i == cellid[f][0]:
                    cellfid[i].append(cellid[f][1])
                else:
                    cellfid[i].append(cellid[f][0])
                    
    for i in range(nbelements):
        numb = len(cellfid[i])
        iterator = maxcellfid - numb
        for k in range(iterator):
             cellfid[i].append(-1)
        cellfid[i].append(numb)
    cellfid = np.asarray(cellfid, dtype=np.int32)
    
    return cellfid

def create_node_cellid(nodeid:'uint32[:,:]', vertex:'float[:,:]', nbelements:'int32', nbnodes:'int32'):
    
    tmp = [[i for i in range(0)] for i in range(nbnodes)]
    longn = np.zeros(nbnodes, dtype=np.uint32)
    
    for i in range(nbelements):
        for j in range(nodeid[i][-1]):
            tmp[nodeid[i][j]].append(i)
            longn[nodeid[i][j]] = longn[nodeid[i][j]] + 1
    
    maxlongn = int(max(longn))
    cellid = -1*np.ones((nbnodes, maxlongn+1), dtype=np.int32)
    
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            cellid[i][j] = tmp[i][j]
        cellid[i][-1] = longn[i]
    
    cellid = cellid.astype(np.int32)
    #######################################
    longc = np.zeros(nbelements, dtype=np.uint32)
    tmp2 = [[i for i in range(0)] for i in range(nbelements)]
    
    for i in range(nbelements):
        for j in range(nodeid[i][-1]):
            for k in range(len(tmp[nodeid[i][j]])):
                if (tmp[nodeid[i][j]][k] not in tmp2[i] and  tmp[nodeid[i][j]][k] != i):
                    tmp2[i].append(tmp[nodeid[i][j]][k])
                    longc[i] = longc[i] + 1
        tmp2[i].sort()
        
    
    maxlongc = int(max(longc))
    cellnid = -1*np.ones((nbelements, maxlongc+1), dtype=np.int32)
    
    for i in range(len(tmp2)):
        for j in range(len(tmp2[i])):
            cellnid[i][j] = tmp2[i][j]
        cellnid[i][-1] = longc[i]
        
    cellnid = cellnid.astype(np.int32)

    return cellid, cellnid

def create_NormalFacesOfCell(centerc:'float[:,:]', centerf:'float[:,:]', faceid:'int32[:,:]', 
                             normal:'float[:,:]',  nbelements:'int32', nf:'float[:,:,:]',  maxcellfid:'int32'):

    #from numpy import np.zeros                                                                                                                                                                               
    ss = np.zeros(3, dtype=np.float64)
    G  = np.zeros(3, dtype=np.float64)
    c  = np.zeros(3, dtype=np.float64)

    #compute the outgoing normal faces for each cell                                                                                                                                                       
    for i in range(nbelements):
        G[:] = centerc[i][:]

        for j in range(faceid[i][-1]):
            f = faceid[i][j]
            c[:] = centerf[f][:]

            if ((G[0]-c[0])*normal[f][0] + (G[1]-c[1])*normal[f][1] + (G[2]-c[2])*normal[f][2]) < 0.:
                ss[:] = normal[f][:]
            else:
                ss[:] = -1.0*normal[f][:]

            nf[i][j][:] = ss[:]

def create_node_ghostid(centergn:'float[:,:,:]', nodeid:'uint32[:,:]'):
    
    nbnodes = len(centergn)
    ghostid = [[i for i in range(0)]  for i in range(nbnodes)]
    cmpt = np.zeros(nbnodes, dtype=np.int32)
    
    for i in range(nbnodes):
        for j in range(len(centergn[i])):
            if centergn[i][j][-1] !=-1:
                ghostid[i].append(np.int32(centergn[i][j][-1]))
                cmpt[i] = cmpt[i]+1
            else:
                ghostid[i].append(-1)
#                cmpt[i] += 0
        ghostid[i].append(int(cmpt[i]))
        
    ghostid = np.asarray(ghostid, dtype=np.int32)
    
    nbelements = len(nodeid)
    ghostnid = [[i for i in range(0)]  for i in range(nbelements)]
    
    for i in range(nbelements):
        for j in range(nodeid[i][-1]):
            for k in range(ghostid[nodeid[i][j]][-1]):
                if ghostid[nodeid[i][j]][k] not in ghostnid[i]:
                    ghostnid[i].append(ghostid[nodeid[i][j]][k])
        ghostnid[i].sort()
    
    maxghostnid = max([len(i) for i in ghostnid])
    #TODO update nodeid
    for i in range(nbelements):
        numb = len(ghostnid[i])
        iterator = maxghostnid - len(ghostnid[i])
        for k in range(iterator):
             ghostnid[i].append(-1)
        ghostnid[i].append(numb)
        
    ghostnid = np.asarray(ghostnid, dtype=np.int32)
        
    return ghostid, ghostnid

def face_info_2d(cellidf, centerc, nodeidc, nodeidf, boundaryfaces, facenameoldf, centerf, normalf, vertexn, halonidn,
                 nbcells, nbfaces, nbnodes, size, namen, precision ):
    # A vérifier !!!!!!
    ghostcenterf = -1*np.ones((nbfaces, 3), dtype=precision)
    ghostcentern = [[1. for i in range(0)]  for i in range(nbnodes)]
    ghostfaceinfon   = [[1. for i in range(0)]  for i in range(nbnodes)]
    
    #compute the ghostcenter for each face and each node
    
    for i in boundaryfaces:
        nod1 = nodeidf[i][1]
        nod2 = nodeidf[i][0]
            
        x_1 = vertexn[nod1]
        x_2 = vertexn[nod2]
        
        c_left = cellidf[i][0]
        v_1 = centerc[c_left]
        gamma = ((v_1[0] - x_2[0])*(x_1[0]-x_2[0]) + (v_1[1]-x_2[1])*(x_1[1]-x_2[1]))/((x_1[0]-x_2[0])**2 + (x_1[1]-x_2[1])**2)
       
        kk = np.array([gamma * x_1[0] + (1 - gamma) * x_2[0], gamma * x_1[1] + (1 - gamma) * x_2[1]])
        
        v_2 = np.array([2 * kk[0] + ( -1 * v_1[0]), 2 * kk[1] + ( -1 * v_1[1])])

        ghostcenterf[i] = np.array([v_2[0], v_2[1], gamma])
        
        ll = [centerf[i][0], centerf[i][1], normalf[i][0], normalf[i][1]]
    
        ghostcentern[nod1].append([v_2[0], v_2[1], cellidf[i][0], facenameoldf[i], i])
        ghostcentern[nod2].append([v_2[0], v_2[1], cellidf[i][0], facenameoldf[i], i])

        ll = [centerf[i][0], centerf[i][1], normalf[i][0], normalf[i][1]]
        ghostfaceinfon[nod1].append(ll)
        ghostfaceinfon[nod2].append(ll)

            
    #define halo cells neighbor by nodes
    maxhalonid = 0
    halonidc = [[0 for i in range(0)] for i in range(nbcells)]
    if size > 1:
        for i in range(nbcells):
            for j in range(nodeidc[i][-1]):
                nod = nodeidc[i][j]
                k = halonidn[nod][-1]
                halonidc[i].extend(halonidn[nod][:k])
            halonidc[i] = list(set(halonidc[i]))
            maxhalonid = max(maxhalonid, len(halonidc[i]))
        
        for i in range(nbcells):
            numb = len(halonidc[i])
            iterator = maxhalonid - len(halonidc[i])
            for k in range(iterator):
                 halonidc[i].append(-1)
            halonidc[i].append(numb)
            
    else:
        halonidc = np.zeros((nbcells,2), dtype=np.int32)
        
    halonidc = np.asarray(halonidc, dtype=np.int32)
        
    return halonidc, ghostcenterf, ghostcentern, ghostfaceinfon

def create_2d_halo_structure(halosext:'int32[:,:]', nodeidf:'int32[:,:]', cellidf:'int32[:,:]', namef:'uint32[:]', namen:'uint32[:]',
                             loctoglobn:'int32[:]', size:'int32', nbcells:'int32', nbfaces:'int32', nbnodes:'int32'):

    def find_tuple(lst:'int32[:,:]', num1:'int32', num2:'int32'):
        for i in range(len(lst)):
            if (num1 in lst[i][1:-1] and num2 in lst[i][1:-1]):
                return i
        return None

    facenameoldf = namef
    #TODO change adding cell index
    halofid    = np.zeros(nbfaces, dtype=np.int32)
    
    # first value of halos._halosext is the halo cell index 
    if size > 1:
        for i in range(nbfaces):
            if cellidf[i][1] == -1:
                n1 = loctoglobn[nodeidf[i][0]]
                n2 = loctoglobn[nodeidf[i][1]]
                
                index = find_tuple(halosext, n1, n2)
                if index is not None:
                    cellidf[i] = [cellidf[i][0], -10]
                    namef[i] = 10
                    namen[nodeidf[i][0]] = 10
                    namen[nodeidf[i][1]] = 10
                    halofid[i] = index

        longueur = 0
        longh = np.zeros(nbnodes, dtype=np.int32)
        
        tmp = [[i for i in range(0)] for i in range(nbnodes)]
        for i in range(nbnodes):
            if namen[i] == 10:
                arg = np.where(halosext[:,1:-1] == loctoglobn[i])
                for ar in arg[0]:
                    tmp[i].append(ar)
                longueur = max(longueur, len(arg[0]))
                longh[i] = len(arg[0])
            else:
                tmp[i].append(-1)
                tmp[i].append(-1)

        halonid = -1*np.ones((nbnodes, longueur+1), dtype=np.int32)
        
        for i in range(len(tmp)):
            for j in range(len(tmp[i])):
                halonid[i][j] = tmp[i][j]
            halonid[i][-1] = longh[i]

        halonid = halonid.astype(np.int32)

    if size == 1 : 
        halonid = np.zeros((nbnodes,2), dtype=np.int32)

    return halonid, halofid, namen, namef, facenameoldf


def update_pediodic_info_2d(centerf, cellidf, cellnidc, centerc, vertexn, cellidn, nbnodes, nbcells, periodicinfaces, periodicoutfaces, 
                            periodicupperfaces, periodicbottomfaces, periodicinnodes, periodicoutnodes, periodicuppernodes, periodicbottomnodes):
    
    
    ##########################################################################
    #TODO periodic test
    periodicidn = [[i for i in range(0)]  for i in range(nbnodes)]
    periodicnidc = [[i for i in range(0)] for i in range(nbcells)]
    periodicfidc = np.zeros(nbcells)
    shiftc = np.zeros((nbcells, 3))
    
    #  TODO Periodic boundary (left and right)
    leftb = {}
    rightb = {}
    for i in periodicinfaces:
        leftb[tuple([np.float64(centerf[i][0]), np.float64(centerf[i][1]), np.float64(centerf[i][2])])] = cellidf[i][0]
    for i in periodicoutfaces:
        rightb[tuple([np.float64(centerf[i][0]), np.float64(centerf[i][1]), np.float64(centerf[i][2])])] = cellidf[i][0]
    
    shiftx = max(vertexn[:,0])
    maxcoordx = max(centerf[:,0])
    longper = np.zeros(len(centerc), dtype=np.int32)
    
    for i in periodicinfaces:
        cellidf[i][1] = rightb[tuple([np.float64(centerf[i][0] + maxcoordx ), np.float64(centerf[i][1]), np.float64(centerf[i][2])])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#rightb[np.float64(centerf[i][1])]
        for j in range(cellnidc[cellidf[i][1]][-1]):
            periodicnidc[cellidf[i][0]].append(cellnidc[cellidf[i][1]][j])
            longper[cellidf[i][0]] +=1
        for cell in periodicnidc[cellidf[i][0]]:
            if i != -1:
                shiftc[cell][0] = -1*shiftx
    
    for i in periodicoutfaces:
        cellidf[i][1] = leftb[tuple([np.float64(centerf[i][0] - maxcoordx), np.float64(centerf[i][1]), np.float64(centerf[i][2])])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#leftb[np.float64(centerf[i][1])]
        for j in range(cellnidc[cellidf[i][1]][-1]):
            periodicnidc[cellidf[i][0]].append(cellnidc[cellidf[i][1]][j])
            longper[cellidf[i][0]] +=1
        for cell in periodicnidc[cellidf[i][0]]:
            if i != -1:
                shiftc[cell][0] = shiftx
    
    leftb = {}
    rightb = {}
    
    for i in periodicinnodes:
        for j in range(cellidn[i][-1]):
            leftb.setdefault(tuple([np.float64(vertexn[i][0]), np.float64(vertexn[i][1]) , np.float64(vertexn[i][2])]), []).append(cellidn[i][j])
    for i in periodicoutnodes:
        for j in range(cellidn[i][-1]):
            rightb.setdefault(tuple([np.float64(vertexn[i][0]), np.float64(vertexn[i][1]) , np.float64(vertexn[i][2])]), []).append(cellidn[i][j])
    
    for i in periodicinnodes:
        periodicidn[i].extend(rightb[tuple([np.float64(vertexn[i][0]) + maxcoordx, np.float64(vertexn[i][1]) , np.float64(vertexn[i][2])])])
    for i in periodicoutnodes:    
        periodicidn[i].extend(leftb[tuple([np.float64(vertexn[i][0]) - maxcoordx, np.float64(vertexn[i][1]) , np.float64(vertexn[i][2])])])
    
    ########################################################################################################
    #  TODO Periodic boundary (bottom and upper)
    leftb = {}
    rightb = {}
    for i in periodicupperfaces:
        leftb[tuple([np.float64(centerf[i][0]), np.float64(centerf[i][1]), np.float64(centerf[i][2])])] = cellidf[i][0]
    for i in periodicbottomfaces:
        rightb[tuple([np.float64(centerf[i][0]), np.float64(centerf[i][1]), np.float64(centerf[i][2])])] = cellidf[i][0]
    
    shifty = max(vertexn[:,1])
    maxcoordy = max(centerf[:,1])
    
    for i in periodicupperfaces:
        cellidf[i][1] = rightb[tuple([np.float64(centerf[i][0]), np.float64(centerf[i][1]) - maxcoordy, np.float64(centerf[i][2])])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#rightb[np.float64(centerf[i][1])]
        for j in range(cellnidc[cellidf[i][1]][-1]):
            periodicnidc[cellidf[i][0]].append(cellnidc[cellidf[i][1]][j])
            longper[cellidf[i][0]] +=1
        for cell in periodicnidc[cellidf[i][0]]:
            if i != -1:
                shiftc[cell][1] = -1*shifty
    for i in periodicbottomfaces:        
        cellidf[i][1] = leftb[tuple([np.float64(centerf[i][0]), np.float64(centerf[i][1])  + maxcoordy, np.float64(centerf[i][2])])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#leftb[np.float64(centerf[i][1])]
        for j in range(cellnidc[cellidf[i][1]][-1]):
            periodicnidc[cellidf[i][0]].append(cellnidc[cellidf[i][1]][j])
            longper[cellidf[i][0]] +=1
        for cell in periodicnidc[cellidf[i][0]]:
            if i != -1:
                shiftc[cell][1] = shifty
                    
    leftb = {}
    rightb = {}
    
    for i in periodicuppernodes:
        for j in range(cellidn[i][-1]):
            leftb.setdefault(tuple([np.float64(vertexn[i][0]), np.float64(vertexn[i][1]) , np.float64(vertexn[i][2])]), []).append(cellidn[i][j])
    for i in periodicbottomnodes:
        rightb.setdefault(tuple([np.float64(vertexn[i][0]), np.float64(vertexn[i][1]) , np.float64(vertexn[i][2])]), []).append(cellidn[i][j])
    
    for i in periodicuppernodes:
        periodicidn[i].extend(rightb[tuple([np.float64(vertexn[i][0]), np.float64(vertexn[i][1]) - maxcoordy, np.float64(vertexn[i][2])])])
    for i in periodicbottomnodes:    
        periodicidn[i].extend(leftb[tuple([np.float64(vertexn[i][0]), np.float64(vertexn[i][1]) + maxcoordy , np.float64(vertexn[i][2])])])
            
    ###########################################################################################    
    maxperiodiccell = 0
    for i in range(nbcells):
        periodicnidc[i] = np.unique(periodicnidc[i])
        maxperiodiccell = max(maxperiodiccell, len(periodicnidc[i]))
    
    for i in range(nbcells):
        iterator  = maxperiodiccell - len(periodicnidc[i])
        for j in range(iterator):
            periodicnidc[i] = np.append(periodicnidc[i], -1)

        if periodicfidc[i] == 0:
            periodicfidc[i] = -1
    for i in range(nbcells):
          periodicnidc[i] = np.append(periodicnidc[i], len(periodicnidc[i][periodicnidc[i] !=-1]))
    
    maxperiodicnode = 0
    for i in range(nbnodes):
        periodicidn[i] = np.unique(periodicidn[i])
        maxperiodicnode = max(maxperiodicnode, len(periodicidn[i]))
    
    for i in range(nbnodes):
        iterator  = maxperiodicnode - len(periodicidn[i])
        for j in range(iterator):
            periodicidn[i] = np.append(periodicidn[i], -1) 
    
    for i in range(nbnodes):
        periodicidn[i] = np.append(periodicidn[i], len(periodicidn[i][periodicidn[i] !=-1]))
            
    periodicnidc = np.asarray(periodicnidc, dtype=np.int32)
    periodicfidc = np.asarray(periodicfidc, dtype=np.int32)
    periodicidn = np.asarray(periodicidn, dtype=np.int32)

###############################################################################
@njit("void(float64[:,:], float64[:,:,:])", fastmath=True, cache=True)
def split_to_triangle(vertices, triangles):
    center = np.zeros(2)
    lv = vertices.shape[0]
    
    center[0] = sum(vertices[:,0])/lv
    center[1] = sum(vertices[:,1])/lv
    
    for i in range(lv):
        triangles[i][0] = vertices[i]
        triangles[i][1] = vertices[(i + 1) % lv]
        triangles[i][2] = center

def Compute_2dcentervolumeOfCell(nodeid:'uint32[:,:]', vertex:'float[:,:]', nbcells:'int32',
                                 center:'float[:,:]', volume:'float[:]'):
    
    
    def split_to_triangle(vertices:'float64[:,:]', triangles:'float64[:,:,:]'):
        ctr = np.zeros(2)
        lv = vertices.shape[0]
        
        ctr[0] = sum(vertices[:,0])/lv
        ctr[1] = sum(vertices[:,1])/lv
        
        for i in range(lv):
            triangles[i][0] = vertices[i]
            triangles[i][1] = vertices[(i + 1) % lv]
            triangles[i][2] = ctr
    
    vertices = np.zeros((4,2))
    triangles = np.zeros((4, 3,2))
    #calcul du barycentre et volume
    for i in range(nbcells):

        if nodeid[i][-1] == 3:
            s_1 = nodeid[i][0]
            s_2 = nodeid[i][1]
            s_3 = nodeid[i][2]
            
            x_1 = vertex[s_1][0]; y_1 = vertex[s_1][1]
            x_2 = vertex[s_2][0]; y_2 = vertex[s_2][1]
            x_3 = vertex[s_3][0]; y_3 = vertex[s_3][1]
    
            center[i][0] = 1./3 * (x_1 + x_2 + x_3); center[i][1] = 1./3*(y_1 + y_2 + y_3); center[i][2] =  0.
            volume[i] = (1./2) * abs((x_1-x_2)*(y_1-y_3)-(x_1-x_3)*(y_1-y_2))
          
        elif nodeid[i][-1] == 4:
            s_1 = nodeid[i][0]
            s_2 = nodeid[i][1]
            s_3 = nodeid[i][2]
            s_4 = nodeid[i][3]
            
            x_1 = vertex[s_1][0]; y_1 = vertex[s_1][1]
            x_2 = vertex[s_2][0]; y_2 = vertex[s_2][1]
            x_3 = vertex[s_3][0]; y_3 = vertex[s_3][1]
            x_4 = vertex[s_4][0]; y_4 = vertex[s_4][1]
            
            vertices[:,:] = np.array([
                                [x_1, y_1],
                                [x_2, y_2],
                                [x_3, y_3],
                                [x_4, y_4],
                                ])
    
            split_to_triangle(vertices, triangles)
            
            center[i][0:2] = triangles[0][2]
            for triangle in triangles:
                x1 = triangle[0][0]; y1 = triangle[0][1]
                x2 = triangle[1][0]; y2 = triangle[1][1]
                x3 = triangle[2][0]; y3 = triangle[2][1]
                volume[i] += (1./2) * abs((x1-x2)*(y1-y3)-(x1-x3)*(y1-y2))
            

def create_2dfaces(nodeidc:'uint32[:,:]', nbelements:'int32', faces:'int32[:,:]',
                   cellf:'int32[:,:]'):
   
    #_petype_fnmap = {
    #    'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
    #    'quad': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]},
    #Create 2d faces
    k = 0
    faces[:,-1] = 2
    for i in range(nbelements):
        if nodeidc[i][-1] == 3: 
            faces[k][0]   = nodeidc[i][0]; faces[k][1]   = nodeidc[i][1]
            faces[k+1][0] = nodeidc[i][1]; faces[k+1][1] = nodeidc[i][2]
            faces[k+2][0] = nodeidc[i][2]; faces[k+2][1] = nodeidc[i][0]
            cellf[i][0] = k; cellf[i][1] = k+1; cellf[i][2] = k+2
            k = k+3
        
        elif nodeidc[i][-1] == 4:
            faces[k][0]   = nodeidc[i][0]; faces[k][1]   = nodeidc[i][1]
            faces[k+1][0] = nodeidc[i][1]; faces[k+1][1] = nodeidc[i][2]
            faces[k+2][0] = nodeidc[i][2]; faces[k+2][1] = nodeidc[i][3]
            faces[k+3][0] = nodeidc[i][3]; faces[k+3][1] = nodeidc[i][0]
            cellf[i][0] = k; cellf[i][1] = k+1; cellf[i][2] = k+2; cellf[i][3] = k+3; 
            k = k+4

def create_info_2dfaces(cellid:'int32[:,:]', nodeid:'int32[:,:]', namen:'uint32[:]', vertex:'float[:,:]', 
                        centerc:'float[:,:]', nbfaces:'int32', normalf:'float[:,:]', mesuref:'float[:]',
                        centerf:'float[:,:]', namef:'uint32[:]'):
    
    norm   = np.zeros(3, dtype=np.float64)
    snorm  = np.zeros(3, dtype=np.float64)
    
    
    #Faces aux bords (1,2,3,4), Faces à l'interieur 0    A VOIR !!!!!
    for i in range(nbfaces):
        if (cellid[i][1] == -1 and cellid[i][1] != -10):
           
            if namen[nodeid[i][0]] == namen[nodeid[i][1]]:
                namef[i] = namen[nodeid[i][0]]
          
            elif ((namen[nodeid[i][0]] == 3 and namen[nodeid[i][1]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 3)):
                namef[i] = 3
            elif ((namen[nodeid[i][0]] == 4 and namen[nodeid[i][1]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 4)):
                namef[i] = 4
                
            elif ((namen[nodeid[i][0]] == 33 and namen[nodeid[i][1]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 33)):
                namef[i] = 33
            elif ((namen[nodeid[i][0]] == 44 and namen[nodeid[i][1]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 44)):
                namef[i] = 44  
            
            else:
                namef[i] = 100
        
        norm[0] = vertex[nodeid[i][0]][1] - vertex[nodeid[i][1]][1]
        norm[1] = vertex[nodeid[i][1]][0] - vertex[nodeid[i][0]][0]
    
        centerf[i][:] = 0.5 * (vertex[nodeid[i][0]][0:3] + vertex[nodeid[i][1]][0:3])
    
        snorm[:] = centerc[cellid[i][0]][:] - centerf[i][:]
    
        if (snorm[0] * norm[0] + snorm[1] * norm[1]) > 0:
            normalf[i][:] = -1*norm[:]
        else:
            normalf[i][:] = norm[:]

        mesuref[i] = np.sqrt(normalf[i][0]**2 + normalf[i][1]**2)

def face_gradient_info_2d(cellidf:'int32[:,:]', nodeidf:'int32[:,:]', centergf:'float[:,:]', namef:'uint32[:]', normalf:'float[:,:]', 
                          centerc:'float[:,:]',  centerh:'float[:,:]', halofid:'int32[:]', vertexn:'float[:,:]', 
                          airDiamond:'float[:]', param1:'float[:]', param2:'float[:]', param3:'float[:]', param4:'float[:]', 
                          f_1:'float[:,:]', f_2:'float[:,:]', f_3:'float[:,:]', f_4:'float[:,:]', shift:'float[:,:]', 
                          dim:'int32'):

    nbface = len(cellidf)
    
    xy_1 = np.zeros(dim)
    xy_2 = np.zeros(dim)
    v_1  = np.zeros(dim)
    v_2  = np.zeros(dim)
    
    for i in range(nbface):
        
        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]       
        
        xy_1[:] = vertexn[i_1][0:dim]
        xy_2[:] = vertexn[i_2][0:dim]
        
        v_1[:] = centerc[c_left][0:dim]
        
        if namef[i] == 0:
            v_2[:] = centerc[c_right][0:dim]
        elif namef[i] == 11 or namef[i] == 22 :
            v_2[0] = centerc[c_right][0] + shift[c_right][0]
            v_2[1] = centerc[c_right][1] 
        elif namef[i] == 33 or namef[i] == 44:
            v_2[0] = centerc[c_right][0]
            v_2[1] = centerc[c_right][1] + shift[c_right][1]
        elif namef[i] == 10:
            v_2[:] = centerh[halofid[i]][0:dim]
        else :
            v_2[:] = centergf[i][0:dim]

        f_1[i][:] = v_1[:] - xy_1[:]
        f_2[i][:] = xy_2[:] - v_1[:]
        f_3[i][:] = v_2[:] - xy_2[:]
        f_4[i][:] = xy_1[:] - v_2[:]
        
        n1 = normalf[i][0]
        n2 = normalf[i][1]
        
        airDiamond[i] = 0.5 *((xy_2[0] - xy_1[0]) * (v_2[1]-v_1[1]) + (v_1[0]-v_2[0]) * (xy_2[1] - xy_1[1]))
        
        param1[i] = 1./(2.*airDiamond[i]) * ((f_1[i][1]+f_2[i][1])*n1 - (f_1[i][0]+f_2[i][0])*n2)
        param2[i] = 1./(2.*airDiamond[i]) * ((f_2[i][1]+f_3[i][1])*n1 - (f_2[i][0]+f_3[i][0])*n2)
        param3[i] = 1./(2.*airDiamond[i]) * ((f_3[i][1]+f_4[i][1])*n1 - (f_3[i][0]+f_4[i][0])*n2)
        param4[i] = 1./(2.*airDiamond[i]) * ((f_4[i][1]+f_1[i][1])*n1 - (f_4[i][0]+f_1[i][0])*n2)


def variables_2d(centerc:'float[:,:]', cellid:'int32[:,:]', haloid:'int32[:,:]', ghostid:'int32[:,:]', haloghostid:'int32[:,:]',
                 periodicid:'int32[:,:]', vertexn:'float[:,:]', centergf:'float[:,:]',
                 halocenterg:'float[:,:]', centerh:'float[:,:]',  
                 R_x:'float[:]', R_y:'float[:]', lambda_x:'float[:]', 
                 lambda_y:'float[:]', number:'uint32[:]', shift:'float[:,:]'):
    
    nbnode = len(R_x)
        
    I_xx = np.zeros(nbnode, dtype=np.float64)
    I_yy = np.zeros(nbnode, dtype=np.float64)
    I_xy = np.zeros(nbnode, dtype=np.float64)
    center = np.zeros(3, dtype=np.float64)
   
    for i in range(nbnode):
        for j in range(cellid[i][-1]):
            center[:] = centerc[cellid[i][j]][0:3]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_xy[i] += (Rx * Ry)
            R_x[i] += Rx
            R_y[i] += Ry
            number[i] += 1
            
        for j in range(ghostid[i][-1]):
            center[:] = centergf[ghostid[i][j]][0:3]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_xy[i] += (Rx * Ry)
            R_x[i] += Rx
            R_y[i] += Ry
            number[i] += 1
       
        #periodic boundary old vertex names)
        if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
            for j in range(periodicid[i][-1]):
                cell = periodicid[i][j]
                center[0] = centerc[cell][0] + shift[cell][0]
                center[1] = centerc[cell][1]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_xy[i] += (Rx * Ry)
                R_x[i] += Rx
                R_y[i] += Ry
                number[i] += 1
                    
        elif vertexn[i][3] == 33 or vertexn[i][3] == 44:
            for j in range(periodicid[i][-1]):
                cell = periodicid[i][j]
                center[0] = centerc[cell][0]
                center[1] = centerc[cell][1] + shift[cell][1]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_xy[i] += (Rx * Ry)
                R_x[i] += Rx
                R_y[i] += Ry
                number[i] += 1
            
        for j in range(haloghostid[i][-1]):
            cell = haloghostid[i][j]

            center[:] = halocenterg[cell]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_xy[i] += (Rx * Ry)
            R_x[i] += Rx
            R_y[i] += Ry
            number[i] = number[i] + 1
            
            # if haloidn[i][-1] > 0:
        for j in range(haloid[i][-1]):
            cell = haloid[i][j]
            center[:] = centerh[cell][0:3]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_xy[i] += (Rx * Ry)
            R_x[i] += Rx
            R_y[i] += Ry
            number[i] = number[i] + 1
        
        
        D = I_xx[i]*I_yy[i] - I_xy[i]*I_xy[i]
        lambda_x[i] = (I_xy[i]*R_y[i] - I_yy[i]*R_x[i]) / D
        lambda_y[i] = (I_xy[i]*R_x[i] - I_xx[i]*R_y[i]) / D

@njit("float64(float64[:], float64[:])", nogil=True)
def distance(x:'float64[:]', y:'float64[:]'):
   
    z = np.zeros(2)
    z[0] = x[0] - y[0]
    z[1] = x[1] - y[1]
    
    return np.sqrt(z[0]**2 + z[1]**2) 

def  dist_ortho_function(innerfaces:'uint32[:]', boundaryfaces:'uint32[:]', infaces:'uint32[:]',
                         cellid:'int32[:,:]', centerc:'float[:,:]', 
                         dist_ortho:'float[:]', centerf:'float[:,:]', 
                         normalf:'float[:,:]', mesuref:'float[:]'):
  
    projection = np.zeros(3)
    projection_bis = np.zeros(3)
    u = np.zeros(3)
    v = np.zeros(3)
    
    for i in boundaryfaces:
        K = cellid[i][0]
        v[:] = centerc[K] - centerf[i]
        u[:] = normalf[i]#/mesuref[i]
        projection[:] = centerc[K] - (v[0]*u[0]+v[1]*u[1]) * u
        dist_ortho[i] = 2*distance(centerc[K].astype('float64'), projection.astype('float64')) #+  distance(ghostcenter[i], projection_bis)
        
    for i in innerfaces:
         K = cellid[i][0]
         L = cellid[i][1]
         u[:] = normalf[i]#/mesuref[i]
         
         v[:] = centerc[K] - centerf[i]
         projection[:] = centerc[K] - (v[0]*u[0]+v[1]*u[1]) * u
         
         v[:] = centerc[L] - centerf[i]
         projection_bis[:] = centerc[L] - (v[0]*u[0]+v[1]*u[1]) * u
         dist_ortho[i] = distance(centerc[K].astype('float64'), projection.astype('float64')) \
                         + distance(centerc[L].astype('float64'), projection_bis.astype('float64'))

import numpy as np
from numpy import int32, uint32, float64
from numba import njit

def create_3d_halo_structure(halosext:'int32[:,:]', nodeidf:'int32[:,:]', cellidf:'int32[:,:]', namef:'uint32[:]', namen:'uint32[:]',
                             loctoglobn:'int32[:]', size:'int32', nbcells:'int32', nbfaces:'int32', nbnodes:'int32'):

    def find_triple(lst, num1, num2, num3):
        for i in range(len(lst)):
            if (num1 in lst[i][1:-1] and num2 in lst[i][1:-1] and num3 in lst[i][1:-1]):
                return i
        return None
    
    def find_quadruple(lst, num1, num2, num3, num4):
        for i in range(len(lst)):
            if (num1 in lst[i][1:-1] and num2 in lst[i][1:-1] and num3 in lst[i][1:-1] and num4 in lst[i][1:-1]):
                return i
        return None

    facenameoldf = namef
    #TODO change adding cell index
    halofid    = np.zeros(nbfaces, dtype=np.int32)
    
    # first value of halos._halosext is the halo cell index 
    if size > 1:
        for i in range(nbfaces):
            if cellidf[i][1] == -1:
                if nodeidf[i][-1] == 3:
                    n1 = loctoglobn[nodeidf[i][0]]
                    n2 = loctoglobn[nodeidf[i][1]]
                    n3 = loctoglobn[nodeidf[i][2]]
                
                    index = find_triple(halosext, n1, n2, n3)
                    if index is not None:
                        cellidf[i] = [cellidf[i][0], -10]
                        namef[i] = 10
                        namen[nodeidf[i][0]] = 10
                        namen[nodeidf[i][1]] = 10
                        namen[nodeidf[i][2]] = 10
                        halofid[i] = index
                
                if nodeidf[i][-1] == 4:
                    n1 = loctoglobn[nodeidf[i][0]]
                    n2 = loctoglobn[nodeidf[i][1]]
                    n3 = loctoglobn[nodeidf[i][2]]
                    n4 = loctoglobn[nodeidf[i][3]]
                    
                    index = find_quadruple(halosext, n1, n2, n3, n4)
                    if index is not None:
                        cellidf[i] = [cellidf[i][0], -10]
                        namef[i] = 10
                        namen[nodeidf[i][0]] = 10
                        namen[nodeidf[i][1]] = 10
                        namen[nodeidf[i][2]] = 10
                        namen[nodeidf[i][3]] = 10
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

def face_info_3d(cellidf, centerc, nodeidc, nodeidf, boundaryfaces, facenameoldf, centerf, normalf, mesuref, vertexn, halonidn,
                 nbcells, nbfaces, nbnodes, size, namen, precision  ):
    # A vérifier !!!!!!
    ghostcenterf     = -1*np.ones((nbfaces, 4), dtype=precision)
    ghostcentern     = [[1. for i in range(0)]  for i in range(nbnodes)]
    ghostfaceinfon   = [[1. for i in range(0)]  for i in range(nbnodes)]
    
    #compute the ghostcenter for each face and each node
    kk = np.zeros(3)
    for i in boundaryfaces:
        #TODO ghost center à verifier
        nod1 = nodeidf[i][1]
        nod2 = nodeidf[i][0]
        nod3 = nodeidf[i][2]
        
        n = normalf[i]/mesuref[i]
        
        c_left = cellidf[i][0]
        v_1 = centerc[c_left]
        u = centerf[i][:] - v_1[:]
        gamma = np.dot(u, n)
        
        kk[0] = v_1[0] + gamma*n[0]
        kk[1] = v_1[1] + gamma*n[1]
        kk[2] = v_1[2] + gamma*n[2]
        
        v_2 = np.array([2 * kk[0] + ( -1 * v_1[0]), 2 * kk[1] + ( -1 * v_1[1]), 2 * kk[2] + ( -1 * v_1[2])])
    
        ll = [centerf[i][0], centerf[i][1], centerf[i][2], normalf[i][0], 
              normalf[i][1], normalf[i][2]]
        
        ghostcenterf[i] = [v_2[0], v_2[1], v_2[2], gamma]
        ghostcentern[nod1].append([v_2[0], v_2[1], v_2[2], cellidf[i][0], facenameoldf[i], i])
        ghostcentern[nod2].append([v_2[0], v_2[1], v_2[2], cellidf[i][0], facenameoldf[i], i])
        ghostcentern[nod3].append([v_2[0], v_2[1], v_2[2], cellidf[i][0], facenameoldf[i], i])
        
        ghostfaceinfon[nod1].append(ll)
        ghostfaceinfon[nod2].append(ll)
        ghostfaceinfon[nod3].append(ll)
        
        if nodeidf[i][-1] == 4:
            nod4 = nodeidf[i][3]
            ghostcentern[nod4].append([v_2[0], v_2[1], v_2[2], cellidf[i][0], facenameoldf[i], i])
            ghostfaceinfon[nod4].append(ll)
        
    #define halo cells neighbor by nodes
    maxhalonid = 0
    if size > 1:
        halonidc = [[] for i in range(nbcells)]
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
             
def update_pediodic_info_3d(centerf, cellidf, cellnidc, centerc, vertexn, cellidn, nbnodes, nbcells, periodicinfaces, periodicoutfaces, 
                            periodicupperfaces, periodicbottomfaces, periodicinnodes, periodicoutnodes, periodicuppernodes, periodicbottomnodes,
                            periodicfrontfaces, periodicbackfaces, periodicfrontnodes, periodicbacknodes):

    
    ##########################################################################
    #TODO periodic test
    periodicidn = [[] for i in range(nbnodes)]
    periodicnidc = [[] for i in range(nbcells)]
    periodicfidc = np.zeros(nbcells)
    shiftc = np.zeros((nbcells, 3))
    
    #  TODO Periodic boundary (left and right)
    leftb = {}
    rightb = {}
    for i in periodicinfaces:
        leftb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]), np.float(centerf[i][2])])] = cellidf[i][0]
    for i in periodicoutfaces:
        rightb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]), np.float(centerf[i][2])])] = cellidf[i][0]
    
    shiftx = max(vertexn[:,0])
    maxcoordx = max(centerf[:,0])
    longper = np.zeros(len(centerc), dtype=np.int32)
    
    for i in periodicinfaces:
        cellidf[i][1] = rightb[tuple([np.float(centerf[i][0] + maxcoordx ), np.float(centerf[i][1]), np.float(centerf[i][2])])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#rightb[np.float(centerf[i][1])]
        for j in range(cellnidc[cellidf[i][1]][-1]):
            periodicnidc[cellidf[i][0]].append(cellnidc[cellidf[i][1]][j])
            longper[cellidf[i][0]] +=1
        for cell in periodicnidc[cellidf[i][0]]:
            if i != -1:
                shiftc[cell][0] = -1*shiftx
   
    for i in periodicoutfaces:        
        cellidf[i][1] = leftb[tuple([np.float(centerf[i][0] - maxcoordx), np.float(centerf[i][1]), np.float(centerf[i][2])])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#leftb[np.float(centerf[i][1])]
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
            leftb.setdefault(tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) , np.float(vertexn[i][2])]), []).append(cellidn[i][j])
    for i in periodicoutnodes:
        for j in range(cellidn[i][-1]):
            rightb.setdefault(tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) , np.float(vertexn[i][2])]), []).append(cellidn[i][j])
    
    for i in periodicinnodes:
        periodicidn[i].extend(rightb[tuple([np.float(vertexn[i][0]) + maxcoordx, np.float(vertexn[i][1]) , np.float(vertexn[i][2])])])
    for i in periodicoutnodes:        
        periodicidn[i].extend(leftb[tuple([np.float(vertexn[i][0]) - maxcoordx, np.float(vertexn[i][1]) , np.float(vertexn[i][2])])])
    
    ########################################################################################################
    #  TODO Periodic boundary (bottom and upper)
    leftb = {}
    rightb = {}
    for i in periodicupperfaces:
        leftb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]), np.float(centerf[i][2])])] = cellidf[i][0]
    for i in periodicbottomfaces:
        rightb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]), np.float(centerf[i][2])])] = cellidf[i][0]
    
    shifty = max(vertexn[:,1])
    maxcoordy = max(centerf[:,1])
    
    for i in periodicupperfaces:
        cellidf[i][1] = rightb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]) - maxcoordy, np.float(centerf[i][2])])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#rightb[np.float(centerf[i][1])]
        for j in range(cellnidc[cellidf[i][1]][-1]):
            periodicnidc[cellidf[i][0]].append(cellnidc[cellidf[i][1]][j])
            longper[cellidf[i][0]] +=1
        for cell in periodicnidc[cellidf[i][0]]:
            if i != -1:
                shiftc[cell][1] = -1*shifty
    for i in periodicbottomfaces:    
        cellidf[i][1] = leftb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1])  + maxcoordy, np.float(centerf[i][2])])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#leftb[np.float(centerf[i][1])]
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
            leftb.setdefault(tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) , np.float(vertexn[i][2])]), []).append(cellidn[i][j])
    for i in periodicbottomnodes:
        for j in range(cellidn[i][-1]):
            rightb.setdefault(tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) , np.float(vertexn[i][2])]), []).append(cellidn[i][j])
    
    for i in periodicuppernodes:
        periodicidn[i].extend(rightb[tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) - maxcoordy, np.float(vertexn[i][2])])])
    
    for i in periodicbottomnodes:         
        periodicidn[i].extend(leftb[tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) + maxcoordy , np.float(vertexn[i][2])])])
            
            
   ########################################################################################################
    #  TODO Periodic boundary (front and back)
    leftb = {}
    rightb = {}

    for i in periodicfrontfaces:
        leftb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]), np.float(centerf[i][2])])] = cellidf[i][0]
    for i in periodicbackfaces:
        rightb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]), np.float(centerf[i][2])])] = cellidf[i][0]
    
    shiftz = max(vertexn[:,2])
    maxcoordz = max(centerf[:,2])
    
    for i in periodicfrontfaces:
        cellidf[i][1] = rightb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]) , np.float(centerf[i][2])- maxcoordz])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#rightb[np.float(centerf[i][1])]
        for j in range(cellnidc[cellidf[i][1]][-1]):
            periodicnidc[cellidf[i][0]].append(cellnidc[cellidf[i][1]][j])
            longper[cellidf[i][0]] +=1
        for cell in periodicnidc[cellidf[i][0]]:
            if i != -1:
                shiftc[cell][2] = -1*shiftz
    for i in periodicbackfaces:        
        cellidf[i][1] = leftb[tuple([np.float(centerf[i][0]), np.float(centerf[i][1]) , np.float(centerf[i][2]) + maxcoordz])]
        periodicfidc[cellidf[i][0]] = cellidf[i][1]#leftb[np.float(centerf[i][1])]
        for j in range(cellnidc[cellidf[i][1]][-1]):
            periodicnidc[cellidf[i][0]].append(cellnidc[cellidf[i][1]][j])
            longper[cellidf[i][0]] +=1
        for cell in periodicnidc[cellidf[i][0]]:
            if i != -1:
                shiftc[cell][2] = shiftz
                
    leftb = {}
    rightb = {}
    
    for i in periodicfrontnodes:
        for j in range(cellidn[i][-1]):
            leftb.setdefault(tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) , np.float(vertexn[i][2])]), []).append(cellidn[i][j])
    for i in periodicbacknodes:
        for j in range(cellidn[i][-1]):
            rightb.setdefault(tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) , np.float(vertexn[i][2])]), []).append(cellidn[i][j])
    
    for i in periodicfrontnodes:
        periodicidn[i].extend(rightb[tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) , np.float(vertexn[i][2])- maxcoordz])])
    for i in periodicbacknodes:        
        periodicidn[i].extend(leftb[tuple([np.float(vertexn[i][0]), np.float(vertexn[i][1]) , np.float(vertexn[i][2]) + maxcoordz ])])

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


###################################################################################################
@njit("intc(float64[:], float64[:])", nogil=True)
def CrossProductZ(a:'float64[:]', b:'float64[:]'):
    return a[0] * b[1] - a[1] * b[0]

@njit("float64(intc, intc, intc, float64[:,:])", nogil=True)
def Orientation(s_1:'int32', s_2:'int32', s_3:'int32', vertex:'float[:,:]'):
    a = vertex[s_1][0:2]
    b = vertex[s_2][0:2]
    c = vertex[s_3][0:2]
    
    return CrossProductZ(a, b) + CrossProductZ(b, c) + CrossProductZ(c, a);


def oriente_3dfacenodeid(nodeid:'int32[:,:]', normal:'float[:,:]', vertex:'float[:,:]'):
    
    nbfaces = len(nodeid)
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    
    for i in range(nbfaces):
        if nodeid[i][-1] == 3:
            n1 = nodeid[i][0]; n2 = nodeid[i][1]; n3 = nodeid[i][2]
            s1 = vertex[n1][0:3]; s2 = vertex[n2][0:3]; s3 = vertex[n3][0:3];
            
            v1[:] = s2[:] - s1[:] 
            v2[:] = s3[:] - s1[:]
            
            if np.dot(np.ascontiguousarray(np.cross(v1, v2)), np.ascontiguousarray(normal[i].astype(np.float64))) < 0:
                nodeid[i][1] = n3; nodeid[i][2] = n2
        
        elif nodeid[i][-1] == 4:
            s_1 = nodeid[i][0]
            s_2 = nodeid[i][1]
            s_3 = nodeid[i][2]
            s_4 = nodeid[i][3]
        
            if (Orientation(s_1, s_2, s_3, vertex.astype(np.float64)) < 0.0):
            # Triangle abc is already clockwise.  np.where does d fit?
                if (Orientation(s_1, s_3, s_4, vertex.astype(np.float64)) < 0.0):
                    continue;           # Cool!
                if (Orientation(s_1, s_2, s_4, vertex.astype(np.float64)) < 0.0):
                    nodeid[i][2] = s_4;   nodeid[i][3] = s_3;
                else:
                    nodeid[i][0] = s_4;   nodeid[i][3] = s_1;
            elif (Orientation(s_1, s_3, s_4, vertex.astype(np.float64)) < 0.0):
                # Triangle abc is counterclockwise, i.e. acb is clockwise.
                # Also, acd is clockwise.
                if (Orientation(s_1, s_2, s_4, vertex.astype(np.float64)) < 0.0):
                    nodeid[i][1] = s_3;   nodeid[i][2] = s_2;
                else:
                    nodeid[i][0] = s_2;   nodeid[i][1] = s_1;
            else:
                # Triangle abc is counterclockwise, and acd is counterclockwise.
                # Therefore, abcd is counterclockwise.
                nodeid[i][0] = s_3;   nodeid[i][2] = s_1;
                
                
def create_3doppNodeOfFaces(nodeidc:'uint32[:,:]', faceidc:'uint32[:,:]', nodeidf:'uint32[:,:]', nbelements:'int32', nbfaces:'int32'):
    
    #TODO improve the opp node creation
    oppnodeid = [[i for i in range(0)] for i in range(nbfaces)]
    for i in range(nbelements):
        f1 = faceidc[i][0]; f2 = faceidc[i][1]; f3 = faceidc[i][2]; f4 = faceidc[i][3]; 
        n1 = nodeidc[i][0]; n2 = nodeidc[i][1]; n3 = nodeidc[i][2]; n4 = nodeidc[i][3]; 
        
        if n1 not in nodeidf[f1] :
            oppnodeid[f1].append(n1)
        if n1 not in nodeidf[f2] :
            oppnodeid[f2].append(n1)
        if n1 not in nodeidf[f3] :
            oppnodeid[f3].append(n1)
        if n1 not in nodeidf[f4] :
            oppnodeid[f4].append(n1)
        
        if n2 not in nodeidf[f1] :
            oppnodeid[f1].append(n2)
        if n2 not in nodeidf[f2] :
            oppnodeid[f2].append(n2)
        if n2 not in nodeidf[f3] :
            oppnodeid[f3].append(n2)
        if n2 not in nodeidf[f4] :
            oppnodeid[f4].append(n2)
        
        if n3 not in nodeidf[f1] :
            oppnodeid[f1].append(n3)
        if n3 not in nodeidf[f2] :
            oppnodeid[f2].append(n3)
        if n3 not in nodeidf[f3] :
            oppnodeid[f3].append(n3)
        if n3 not in nodeidf[f4] :
            oppnodeid[f4].append(n3)
        
        if n4 not in nodeidf[f1] :
            oppnodeid[f1].append(n4)
        if n4 not in nodeidf[f2] :
            oppnodeid[f2].append(n4)
        if n4 not in nodeidf[f3] :
            oppnodeid[f3].append(n4)
        if n4 not in nodeidf[f4] :
            oppnodeid[f4].append(n4)
                    
    for i in range(nbfaces):
        if len(oppnodeid[i]) < 2:
            oppnodeid[i].append(-1)
            
    oppnodeid = np.asarray(oppnodeid, dtype=np.int32)   
    
    return oppnodeid


def face_gradient_info_3d(cellidf:'int32[:,:]', nodeidf:'int32[:,:]', centergf:'float[:,:]', namef:'uint32[:]', normalf:'float[:,:]', 
                          centerc:'float[:,:]',  centerh:'float[:,:]', halofid:'int32[:]', vertexn:'float[:,:]', 
                          airDiamond:'float[:]', param1:'float[:]', param2:'float[:]', param3:'float[:]',
                          n1:'float[:,:]', n2:'float[:,:]', shift:'float[:,:]',  dim:'int32'):

    nbfaces = len(cellidf)
    
    v_1 = np.zeros(dim)
    v_2 = np.zeros(dim)
    s1 = np.zeros(3)
    s2 = np.zeros(3)
    s3 = np.zeros(3)
    s4 = np.zeros(3)
    s5 = np.zeros(3)
    s6 = np.zeros(3)
    s7 = np.zeros(3)
    
    for i in range(nbfaces):

        c_left = cellidf[i][0]
        c_right = cellidf[i][1]
        
        i_1 = nodeidf[i][0]
        i_2 = nodeidf[i][1]   
        i_3 = nodeidf[i][2] 
        i_4 = i_3
        if nodeidf[i][-1] == 4:
            i_4 = nodeidf[i][3] 
        
        v_1[:] = centerc[c_left][0:dim]
        
        if namef[i] == 0:
            v_2[:] = centerc[c_right][0:dim]
        
        elif namef[i] == 11 or namef[i] == 22 :
            v_2[0] = centerc[c_right][0] + shift[c_right][0]
            v_2[1] = centerc[c_right][1] 
            v_2[2] = centerc[c_right][2]
        elif namef[i] == 33 or namef[i] == 44:
            v_2[0] = centerc[c_right][0]
            v_2[1] = centerc[c_right][1] + shift[c_right][1]
            v_2[2] = centerc[c_right][2]
        elif namef[i] == 55 or namef[i] == 66:
            v_2[0] = centerc[c_right][0]
            v_2[1] = centerc[c_right][1] 
            v_2[2] = centerc[c_right][2] + shift[c_right][2]
            
        elif namef[i] == 10:
            v_2[:] = centerh[halofid[i]][0:dim]
        else :
            v_2[:] = centergf[i][0:dim]
        
        s1[:] = v_2                 - vertexn[i_2][0:dim]
        s2[:] = vertexn[i_4][0:dim] - vertexn[i_2][0:dim]
        s3[:] = v_1                 - vertexn[i_2][0:dim]
        n1[i][:] = (0.5 * np.cross(s1, s2)) + (0.5 * np.cross(s2, s3))
        
        s4[:] = v_2                 - vertexn[i_3][0:dim]
        s5[:] = vertexn[i_1][0:dim] - vertexn[i_3][0:dim]
        s6[:] = v_1                 - vertexn[i_3][0:dim]
        n2[i][:] = (0.5 * np.cross(s4, s5)) + (0.5 * np.cross(s5, s6))
        
        s7[:] = v_2 - v_1
        airDiamond[i] = np.dot(np.ascontiguousarray(normalf[i].astype(np.float64)), np.ascontiguousarray(s7))
        
        
        param1[i] = (np.dot(np.ascontiguousarray(n1[i]), np.ascontiguousarray(normalf[i]))) / airDiamond[i]
        param2[i] = (np.dot(np.ascontiguousarray(n2[i]), np.ascontiguousarray(normalf[i]))) / airDiamond[i]
        param3[i] = (np.dot(np.ascontiguousarray(normalf[i]), np.ascontiguousarray(normalf[i]))) / airDiamond[i]
     

def create_info_3dfaces(cellid:'int32[:,:]', nodeid:'int32[:,:]', namen:'uint32[:]', vertex:'float[:,:]', 
                        centerc:'float[:,:]', nbfaces:'int32', normalf:'float[:,:]', tangentf:'float[:,:]', binormalf:'float[:,:]', mesuref:'float[:]',
                        centerf:'float[:,:]', namef:'uint32[:]'):
    
    norm   = np.zeros(3)
    snorm  = np.zeros(3)
    u      = np.zeros(3)
    v      = np.zeros(3)
    tangent      = np.zeros(3)
    binormal      = np.zeros(3)    
    
    for i in range(nbfaces):
             
        if nodeid[i][-1] == 3:
            centerf[i][:] = 1./3 * (vertex[nodeid[i][0]][:3] + vertex[nodeid[i][1]][:3] + vertex[nodeid[i][2]][:3])
            if (cellid[i][1] == -1 ):      
                if namen[nodeid[i][0]] == namen[nodeid[i][1]] and namen[nodeid[i][0]] == namen[nodeid[i][2]] :
                    namef[i] = namen[nodeid[i][0]]
           
                elif ((namen[nodeid[i][0]] == 5 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 5 and namen[nodeid[i][2]] !=0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 5)):
                        namef[i] = 5
                
                elif ((namen[nodeid[i][0]] == 6 and namen[nodeid[i][1]] !=0 and namen[nodeid[i][2]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 6 and namen[nodeid[i][2]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 6)):
                    namef[i] = 6
                    
                elif ((namen[nodeid[i][0]] == 3 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 3 and namen[nodeid[i][2]] != 0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 3)):
                        namef[i] = 3
                
                elif ((namen[nodeid[i][0]] == 4 and namen[nodeid[i][1]] !=0 and namen[nodeid[i][2]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 4 and namen[nodeid[i][2]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 4)):
                    namef[i] = 4
                
                elif ((namen[nodeid[i][0]] == 55 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 55 and namen[nodeid[i][2]] != 0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 55)):
                        namef[i] = 55
                
                elif ((namen[nodeid[i][0]] == 66 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 66 and namen[nodeid[i][2]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 66)):
                    namef[i] = 66
                    
                elif ((namen[nodeid[i][0]] == 33 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 33 and namen[nodeid[i][2]] != 0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 33)):
                        namef[i] = 33
                
                elif ((namen[nodeid[i][0]] == 44 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 44 and namen[nodeid[i][2]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 44)):
                    namef[i] = 44
                else:
                    namef[i] = 100
                
                    
        elif nodeid[i][-1] == 4:
            centerf[i][:] = 1./4 * (vertex[nodeid[i][0]][:3] + vertex[nodeid[i][1]][:3] + vertex[nodeid[i][2]][:3] + vertex[nodeid[i][3]][:3])
            if (cellid[i][1] == -1 ):          
                if (namen[nodeid[i][0]] == namen[nodeid[i][1]] and namen[nodeid[i][0]] == namen[nodeid[i][2]] and
                    namen[nodeid[i][0]] == namen[nodeid[i][3]]) :
                    namef[i] = namen[nodeid[i][0]]
               
                elif ((namen[nodeid[i][0]] == 5 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 5 and namen[nodeid[i][2]] !=0  and namen[nodeid[i][3]] != 0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 5 and namen[nodeid[i][3]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] == 5)):
                        namef[i] = 5
                
                elif ((namen[nodeid[i][0]] == 6 and namen[nodeid[i][1]] !=0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 6 and namen[nodeid[i][2]] != 0  and namen[nodeid[i][3]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 6  and namen[nodeid[i][3]] != 0)or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] == 6)):
                    namef[i] = 6
                    
                 
                elif ((namen[nodeid[i][0]] == 3 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 3 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 3 and namen[nodeid[i][3]] != 0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] == 3)):
                        namef[i] = 3
                
                elif ((namen[nodeid[i][0]] == 4 and namen[nodeid[i][1]] !=0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 4 and namen[nodeid[i][2]] != 0  and namen[nodeid[i][3]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 4  and namen[nodeid[i][3]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0  and namen[nodeid[i][3]] == 4)):
                    namef[i] = 4
                
                elif ((namen[nodeid[i][0]] == 55 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 55 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 55 and namen[nodeid[i][3]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0  and namen[nodeid[i][3]] == 55)):
                        namef[i] = 55
                
                elif ((namen[nodeid[i][0]] == 66 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 66 and namen[nodeid[i][2]] != 0   and namen[nodeid[i][3]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 66   and namen[nodeid[i][3]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0  and namen[nodeid[i][3]] == 66)):
                    namef[i] = 66
                    
                elif ((namen[nodeid[i][0]] == 33 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 33 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or 
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 33 and namen[nodeid[i][3]] != 0) or
                      (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0  and namen[nodeid[i][3]] == 33)):
                        namef[i] = 33
                
                elif ((namen[nodeid[i][0]] == 44 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0 and namen[nodeid[i][3]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] == 44 and namen[nodeid[i][2]] != 0   and namen[nodeid[i][3]] != 0) or 
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] == 44   and namen[nodeid[i][3]] != 0) or
                    (namen[nodeid[i][0]] != 0 and namen[nodeid[i][1]] != 0 and namen[nodeid[i][2]] != 0  and namen[nodeid[i][3]] == 44)):
                    namef[i] = 44
                else:
                    namef[i] = 200
              
        u[:] = vertex[nodeid[i][1]][0:3]-vertex[nodeid[i][0]][0:3]
        v[:] = vertex[nodeid[i][2]][0:3]-vertex[nodeid[i][0]][0:3]
        
        norm[0] = 0.5*(u[1]*v[2] - u[2]*v[1])
        norm[1] = 0.5*(u[2]*v[0] - u[0]*v[2])
        norm[2] = 0.5*(u[0]*v[1] - u[1]*v[0])
    
        tangent[:]=u[:]
        
        binormal[0] = 0.5*(u[1]*norm[2] - u[2]*norm[1])
        binormal[1] = 0.5*(u[2]*norm[0] - u[0]*norm[2])
        binormal[2] = 0.5*(u[0]*norm[1] - u[1]*norm[0])
                
        snorm[:] = centerc[cellid[i][0]][:] - centerf[i][:]
    
        if (snorm[0] * norm[0] + snorm[1] * norm[1] + snorm[2] * norm[2]) > 0:
            normalf[i][:] = -1*norm[:]
        else:
            normalf[i][:] = norm[:]
    
        mesuref[i] = np.sqrt(normalf[i][0]**2 + normalf[i][1]**2 + normalf[i][2]**2)
        binormalf[i][:] = binormal[:]
        tangentf[i][:] = tangent[:]
        
        
@njit("void(float64[:,:], float64[:,:,:])", fastmath=True)
def split_to_tetra(vertices, tetrahedra):
    center = np.zeros(3)
    lv = vertices.shape[0]
    
    center[0] = sum(vertices[:,0])/lv
    center[1] = sum(vertices[:,1])/lv
    center[2] = sum(vertices[:,2])/lv
    
    for i in range(lv):
        tetrahedra[i][0] = vertices[i]
        tetrahedra[i][1] = vertices[(i + 1) % lv]
        tetrahedra[i][2] = vertices[(i + 2) % lv]
        tetrahedra[i][3] = center
    

def Compute_3dcentervolumeOfCell(nodeid:'uint32[:,:]', vertex:'float[:,:]', nbcells:'int32',
                                 center:'float[:,:]', volume:'float[:]'):
    
    wedge = np.zeros(3)
    u = np.zeros(3)
    v = np.zeros(3)
    w = np.zeros(3)
    vertices1 = np.zeros((8,3))
    vertices2 = np.zeros((5,3))
    
    tetrahedra1 = np.zeros((8, 4, 3))
    tetrahedra2 = np.zeros((5, 4, 3))
    
    
    #calcul du barycentre et volume
    for i in range(nbcells):
        if nodeid[i][-1] == 4:
            s_1 = nodeid[i][0]
            s_2 = nodeid[i][1]
            s_3 = nodeid[i][2]
            s_4 = nodeid[i][3]
            
            x_1 = vertex[s_1][0]; y_1 = vertex[s_1][1]; z_1 = vertex[s_1][2]
            x_2 = vertex[s_2][0]; y_2 = vertex[s_2][1]; z_2 = vertex[s_2][2]
            x_3 = vertex[s_3][0]; y_3 = vertex[s_3][1]; z_3 = vertex[s_3][2]
            x_4 = vertex[s_4][0]; y_4 = vertex[s_4][1]; z_4 = vertex[s_4][2]
            
            center[i][0] = 1./4*(x_1 + x_2 + x_3 + x_4) 
            center[i][1] = 1./4*(y_1 + y_2 + y_3 + y_4)
            center[i][2] = 1./4*(z_1 + z_2 + z_3 + z_4)
            
            u[:] = vertex[s_2][0:3]-vertex[s_1][0:3]
            v[:] = vertex[s_3][0:3]-vertex[s_1][0:3]
            w[:] = vertex[s_4][0:3]-vertex[s_1][0:3]
            
            wedge[0] = v[1]*w[2] - v[2]*w[1]
            wedge[1] = v[2]*w[0] - v[0]*w[2]
            wedge[2] = v[0]*w[1] - v[1]*w[0]
            
            volume[i] = 1./6*np.fabs(u[0]*wedge[0] + u[1]*wedge[1] + u[2]*wedge[2]) 
            
        elif nodeid[i][-1] == 8:
            
            s_1 = nodeid[i][0]
            s_2 = nodeid[i][1]
            s_3 = nodeid[i][2]
            s_4 = nodeid[i][3]
            s_5 = nodeid[i][4]
            s_6 = nodeid[i][5]
            s_7 = nodeid[i][6]
            s_8 = nodeid[i][7]
            
            x_1 = vertex[s_1][0]; y_1 = vertex[s_1][1]; z_1 = vertex[s_1][2]
            x_2 = vertex[s_2][0]; y_2 = vertex[s_2][1]; z_2 = vertex[s_2][2] 
            x_3 = vertex[s_3][0]; y_3 = vertex[s_3][1]; z_3 = vertex[s_3][2] 
            x_4 = vertex[s_4][0]; y_4 = vertex[s_4][1]; z_4 = vertex[s_4][2]
            x_5 = vertex[s_5][0]; y_5 = vertex[s_5][1]; z_5 = vertex[s_5][2]
            x_6 = vertex[s_6][0]; y_6 = vertex[s_6][1]; z_6 = vertex[s_6][2] 
            x_7 = vertex[s_7][0]; y_7 = vertex[s_7][1]; z_7 = vertex[s_7][2] 
            x_8 = vertex[s_8][0]; y_8 = vertex[s_8][1]; z_8 = vertex[s_8][2]
            
            vertices1[:,:] = np.array([
                                [x_1, y_1, z_1],
                                [x_2, y_2, z_2],
                                [x_3, y_3, z_3],
                                [x_4, y_4, z_4],
                                [x_5, y_5, z_5],
                                [x_6, y_6, z_6],
                                [x_7, y_7, z_7],
                                [x_8, y_8, z_8]])
    
            split_to_tetra(vertices1, tetrahedra1)
            
            for tetrahedron in tetrahedra1:
                u[:] = tetrahedron[1]-tetrahedron[0]
                v[:] = tetrahedron[2]-tetrahedron[0]
                w[:] = tetrahedron[3]-tetrahedron[0]
                
                wedge[0] = v[1]*w[2] - v[2]*w[1]
                wedge[1] = v[2]*w[0] - v[0]*w[2]
                wedge[2] = v[0]*w[1] - v[1]*w[0]
                
                center[i] = tetrahedron[-1]
                volume[i] += 1./6*np.fabs(u[0]*wedge[0] + u[1]*wedge[1] + u[2]*wedge[2]) 
                
        elif nodeid[i][-1] == 5:
            
            s_1 = nodeid[i][0]
            s_2 = nodeid[i][1]
            s_3 = nodeid[i][2]
            s_4 = nodeid[i][3]
            s_5 = nodeid[i][4]
            
            x_1 = vertex[s_1][0]; y_1 = vertex[s_1][1]; z_1 = vertex[s_1][2]
            x_2 = vertex[s_2][0]; y_2 = vertex[s_2][1]; z_2 = vertex[s_2][2] 
            x_3 = vertex[s_3][0]; y_3 = vertex[s_3][1]; z_3 = vertex[s_3][2] 
            x_4 = vertex[s_4][0]; y_4 = vertex[s_4][1]; z_4 = vertex[s_4][2]
            x_5 = vertex[s_5][0]; y_5 = vertex[s_5][1]; z_5 = vertex[s_5][2]
            
            vertices2[:,:] = np.array([
                                [x_1, y_1, z_1],
                                [x_2, y_2, z_2],
                                [x_3, y_3, z_3],
                                [x_4, y_4, z_4],
                                [x_5, y_5, z_5]])
    
            split_to_tetra(vertices2, tetrahedra2)
            
            for tetrahedron in tetrahedra2:
                u[:] = tetrahedron[1]-tetrahedron[0]
                v[:] = tetrahedron[2]-tetrahedron[0]
                w[:] = tetrahedron[3]-tetrahedron[0]
                
                wedge[0] = v[1]*w[2] - v[2]*w[1]
                wedge[1] = v[2]*w[0] - v[0]*w[2]
                wedge[2] = v[0]*w[1] - v[1]*w[0]
                
                center[i] = tetrahedron[-1]
                volume[i] += 1./6*np.fabs(u[0]*wedge[0] + u[1]*wedge[1] + u[2]*wedge[2]) 

def create_3dfaces(nodeidc:'uint32[:,:]', nbelements:'int32', faces:'int32[:,:]',
                   cellf:'int32[:,:]'):
    
    #_petype_fnmap = {
    #    'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
    
    #    'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
    #                     [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
    
    #    'pri': {'quad': [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
    #            'tri': [[0, 1, 2], [3, 4, 5]]},
    #    'pyr': {'quad': [[0, 1, 2, 3]],
    #            'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]}
    #}

    #Create 3d faces
    k = 0
    faces[:,-1] = 3
    for i in range(nbelements):
        if nodeidc[i][-1] == 4:
            faces[k][0]   = nodeidc[i][0]; faces[k][1]   = nodeidc[i][1]; faces[k][2]   = nodeidc[i][2]
            faces[k+1][0] = nodeidc[i][2]; faces[k+1][1] = nodeidc[i][3]; faces[k+1][2] = nodeidc[i][0]
            faces[k+2][0] = nodeidc[i][0]; faces[k+2][1] = nodeidc[i][1]; faces[k+2][2] = nodeidc[i][3]
            faces[k+3][0] = nodeidc[i][3]; faces[k+3][1] = nodeidc[i][1]; faces[k+3][2] = nodeidc[i][2]
            cellf[i][0]  = k; cellf[i][1] = k+1; cellf[i][2] = k+2; cellf[i][3] = k+3
            k = k+4
        
        elif nodeidc[i][-1] == 5:
            faces[k][0]   = nodeidc[i][0]; faces[k][1]   = nodeidc[i][1]; faces[k][2]   = nodeidc[i][4]; faces[k][-1] = 3
            faces[k+1][0] = nodeidc[i][1]; faces[k+1][1] = nodeidc[i][2]; faces[k+1][2] = nodeidc[i][4]; faces[k+1][-1] = 3
            faces[k+2][0] = nodeidc[i][2]; faces[k+2][1] = nodeidc[i][3]; faces[k+2][2] = nodeidc[i][4]; faces[k+1][-1] = 3
            faces[k+3][0] = nodeidc[i][0]; faces[k+3][1] = nodeidc[i][3]; faces[k+3][2] = nodeidc[i][4]; faces[k+1][-1] = 3
            faces[k+4][0] = nodeidc[i][0]; faces[k+4][1] = nodeidc[i][1]; faces[k+4][2] = nodeidc[i][2]; faces[k+4][3] = nodeidc[i][3]; faces[k+4][-1] = 4
            cellf[i][0]  = k; cellf[i][1] = k+1; cellf[i][2] = k+2; cellf[i][3] = k+3; cellf[i][4] = k+4
            k = k+5
            
        elif nodeidc[i][-1] == 8:
            faces[k][0]   = nodeidc[i][0]; faces[k][1]   = nodeidc[i][1]; faces[k][2]   = nodeidc[i][2]; faces[k][3]   = nodeidc[i][3]; faces[k][-1]   = 4
            faces[k+1][0] = nodeidc[i][0]; faces[k+1][1] = nodeidc[i][1]; faces[k+1][2] = nodeidc[i][4]; faces[k+1][3] = nodeidc[i][5]; faces[k+1][-1] = 4
            faces[k+2][0] = nodeidc[i][1]; faces[k+2][1] = nodeidc[i][2]; faces[k+2][2] = nodeidc[i][5]; faces[k+2][3] = nodeidc[i][6]; faces[k+2][-1] = 4
            faces[k+3][0] = nodeidc[i][2]; faces[k+3][1] = nodeidc[i][3]; faces[k+3][2] = nodeidc[i][6]; faces[k+3][3] = nodeidc[i][7]; faces[k+3][-1] = 4
            faces[k+4][0] = nodeidc[i][0]; faces[k+4][1] = nodeidc[i][3]; faces[k+4][2] = nodeidc[i][4]; faces[k+4][3] = nodeidc[i][7]; faces[k+4][-1] = 4
            faces[k+5][0] = nodeidc[i][4]; faces[k+5][1] = nodeidc[i][5]; faces[k+5][2] = nodeidc[i][6]; faces[k+5][3] = nodeidc[i][7]; faces[k+5][-1] = 4
            cellf[i][0]  = k; cellf[i][1] = k+1; cellf[i][2] = k+2; cellf[i][3] = k+3; cellf[i][4] = k+4; cellf[i][5] = k+5;  
            k = k+6
            
#TODO periodic checked ok 
def variables_3d(centerc:'float[:,:]', cellid:'int32[:,:]', haloid:'int32[:,:]', ghostid:'int32[:,:]', haloghostid:'int32[:,:]',
                 periodicid:'int32[:,:]', vertexn:'float[:,:]', centergf:'float[:,:]',
                 halocenterg:'float[:,:]', centerh:'float[:,:]',  
                 R_x:'float[:]', R_y:'float[:]', R_z:'float[:]', lambda_x:'float[:]', 
                 lambda_y:'float[:]', lambda_z:'float[:]', number:'uint32[:]', shift:'float[:,:]'):

    nbnode = len(R_x)
     
    I_xx = np.zeros(nbnode)
    I_yy = np.zeros(nbnode)
    I_zz = np.zeros(nbnode)
    I_xy = np.zeros(nbnode)
    I_xz = np.zeros(nbnode)
    I_yz = np.zeros(nbnode)
    center = np.zeros(3)
  

    for i in range(nbnode):
        for j in range(cellid[i][-1]):
            center[:] = centerc[cellid[i][j]][0:3]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            Rz = center[2] - vertexn[i][2]
           
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_zz[i] += (Rz * Rz)
            I_xy[i] += (Rx * Ry)
            I_xz[i] += (Rx * Rz)
            I_yz[i] += (Ry * Rz)
           
            R_x[i] += Rx
            R_y[i] += Ry
            R_z[i] += Rz
           
            number[i] += 1
            
        for j in range(ghostid[i][-1]):
            center[:] = centergf[ghostid[i][j]][0:3]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            Rz = center[2] - vertexn[i][2]
           
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_zz[i] += (Rz * Rz)
            I_xy[i] += (Rx * Ry)
            I_xz[i] += (Rx * Rz)
            I_yz[i] += (Ry * Rz)
           
            R_x[i] += Rx
            R_y[i] += Ry
            R_z[i] += Rz
           
            number[i] += 1

        #periodic boundary old vertex names)
        if vertexn[i][3] == 11 or vertexn[i][3] == 22 :
            for j in range(periodicid[i][-1]):
                cell = periodicid[i][j]
                center[0] = centerc[cell][0] + shift[cell][0]
                center[1] = centerc[cell][1]
                center[2] = centerc[cell][2]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                Rz = center[2] - vertexn[i][2]
               
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_zz[i] += (Rz * Rz)
                I_xy[i] += (Rx * Ry)
                I_xz[i] += (Rx * Rz)
                I_yz[i] += (Ry * Rz)
               
                R_x[i] += Rx
                R_y[i] += Ry
                R_z[i] += Rz
                number[i] = number[i] + 1
                    
        elif vertexn[i][3] == 33 or vertexn[i][3] == 44:
            for j in range(periodicid[i][-1]):
                cell = periodicid[i][j]
                center[0] = centerc[cell][0]
                center[1] = centerc[cell][1] + shift[cell][1]
                center[2] = centerc[cell][2]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                Rz = center[2] - vertexn[i][2]
               
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_zz[i] += (Rz * Rz)
                I_xy[i] += (Rx * Ry)
                I_xz[i] += (Rx * Rz)
                I_yz[i] += (Ry * Rz)
               
                R_x[i] += Rx
                R_y[i] += Ry
                R_z[i] += Rz
                number[i] = number[i] + 1
                
        elif vertexn[i][3] == 55 or vertexn[i][3] == 66:
            for j in range(periodicid[i][-1]):
                cell = periodicid[i][j]
                center[0] = centerc[cell][0]
                center[1] = centerc[cell][1] 
                center[2] = centerc[cell][2] + shift[cell][2]
                
                Rx = center[0] - vertexn[i][0]
                Ry = center[1] - vertexn[i][1]
                Rz = center[2] - vertexn[i][2]
               
                I_xx[i] += (Rx * Rx)
                I_yy[i] += (Ry * Ry)
                I_zz[i] += (Rz * Rz)
                I_xy[i] += (Rx * Ry)
                I_xz[i] += (Rx * Rz)
                I_yz[i] += (Ry * Rz)
               
                R_x[i] += Rx
                R_y[i] += Ry
                R_z[i] += Rz
                number[i] = number[i] + 1
     
        for j in range(haloid[i][-1]):
            cell = haloid[i][j]
            center[:] = centerh[cell][0:3]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            Rz = center[2] - vertexn[i][2]
           
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_zz[i] += (Rz * Rz)
            I_xy[i] += (Rx * Ry)
            I_xz[i] += (Rx * Rz)
            I_yz[i] += (Ry * Rz)
           
            R_x[i] += Rx
            R_y[i] += Ry
            R_z[i] += Rz
            number[i] = number[i] + 1
   
        for j in range(haloghostid[i][-1]):
            cell = haloghostid[i][j]
            center[:] = halocenterg[cell]
            Rx = center[0] - vertexn[i][0]
            Ry = center[1] - vertexn[i][1]
            Rz = center[2] - vertexn[i][2]
           
            I_xx[i] += (Rx * Rx)
            I_yy[i] += (Ry * Ry)
            I_zz[i] += (Rz * Rz)
            I_xy[i] += (Rx * Ry)
            I_xz[i] += (Rx * Rz)
            I_yz[i] += (Ry * Rz)
           
            R_x[i] += Rx
            R_y[i] += Ry
            R_z[i] += Rz
            number[i] = number[i] + 1
 
        D = I_xx[i]*I_yy[i]*I_zz[i] + 2*I_xy[i]*I_xz[i]*I_yz[i] - I_xx[i]*I_yz[i]*I_yz[i] - I_yy[i]*I_xz[i]*I_xz[i] - I_zz[i]*I_xy[i]*I_xy[i]
       
        lambda_x[i] = ((I_yz[i]*I_yz[i] - I_yy[i]*I_zz[i])*R_x[i] + (I_xy[i]*I_zz[i] - I_xz[i]*I_yz[i])*R_y[i] + (I_xz[i]*I_yy[i] - I_xy[i]*I_yz[i])*R_z[i]) / D
        lambda_y[i] = ((I_xy[i]*I_zz[i] - I_xz[i]*I_yz[i])*R_x[i] + (I_xz[i]*I_xz[i] - I_xx[i]*I_zz[i])*R_y[i] + (I_yz[i]*I_xx[i] - I_xz[i]*I_xy[i])*R_z[i]) / D
        lambda_z[i] = ((I_xz[i]*I_yy[i] - I_xy[i]*I_yz[i])*R_x[i] + (I_yz[i]*I_xx[i] - I_xz[i]*I_xy[i])*R_y[i] + (I_xy[i]*I_xy[i] - I_xx[i]*I_yy[i])*R_z[i]) / D
  

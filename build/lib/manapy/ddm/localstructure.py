#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:10:58 2020

@author: kissami
"""
from collections import OrderedDict
from mpi4py import MPI
import numpy as np
from numba import njit, literal_unroll
from numba.typed import  Dict

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

__all__ = ["generate_mesh"]

class Cells:
    nodeid = []
    faceid = []
    center = []
    volume = []
    cellfid = []
    nf = [] #normal sortante des faces
    cellnid = []
    globalindex = OrderedDict()
    father = Dict()
    son = Dict()
    iadiv = Dict()

    def __init__(self, nodeid, faceid, center, volume, father, son, iadiv, globalindex, cellfid, cellnid,
                 nf):
        self.nodeid = nodeid    # instance variable unique to each instance
        self.faceid = faceid
        self.center = center
        self.volume = volume
        self.father = father
        self.son = son
        self.iadiv = iadiv
        self.globalindex = globalindex
        self.cellfid = cellfid
        self.cellnid = cellnid
        self.nf = nf

class Nodes:
    vertex = []
    name = []
    cellid = []
    globalindex = OrderedDict()
    halonid = []
    ghostcenter = []

    def __init__(self, vertex, name, cellid, globalindex, halonid, ghostcenter):
        self.vertex = vertex    # instance variable unique to each instance
        self.name = name
        self.cellid = cellid
        self.globalindex = globalindex
        self.halonid = halonid
        self.ghostcenter = ghostcenter

class Faces:
    nodeid = []
    cellid = []
    name = []
    halofid = []
    normal = []
    mesure = []
    bound = 0
    center = []
    ghostcenter = []

    def __init__(self, nodeid, cellid, name, normal, mesure, center, bound, halofid, ghostcenter):
        self.nodeid = nodeid    # instance variable unique to each instance
        self.cellid = cellid
        self.name = name
        self.normal = normal
        self.mesure = mesure
        self.bound = bound
        self.halofid = halofid
        self.center = center
        self.ghostcenter = ghostcenter

class Halo:
    halosint = []
    halosext = []
    neigh = []
    centvol = []
    faces = OrderedDict()
    nodes = OrderedDict()

    def __init__(self, halosint, halosext, centvol, neigh, faces, nodes):
        self.halosint = halosint    # instance variable unique to each instance
        self.halosext = halosext
        self.neigh = neigh
        self.centvol = centvol
        self.faces = faces
        self.nodes = nodes

@njit
def normalvector(node_a, node_b, bary):

    norm = np.zeros(2)#[None]*2
    snorm = np.zeros(2)#[None]*2
    center = np.zeros(2)
    normal = np.zeros(2)#[None]*2

    norm[0] = node_a[1] - node_b[1]
    norm[1] = node_b[0] - node_a[0]

    center[0] = 0.5 * (node_a[0] + node_b[0])
    center[1] = 0.5 * (node_a[1] + node_b[1])

    snorm[0] = bary[0] - center[0]
    snorm[1] = bary[1] - center[1]

    if (snorm[0] * norm[0] + snorm[1] * norm[1]) > 0:
        normal[0] = -1*norm[0]
        normal[1] = -1*norm[1]
    else:
        normal[0] = norm[0]
        normal[1] = norm[1]

    normal[0] = normal[0]
    normal[1] = normal[1]
    
    mesure = np.sqrt(normal[0]**2 + normal[1]**2)

    return normal, mesure

def create_local_mesh(file):

    clear_class(Cells, Nodes, Faces)
    #Lecture des cellules à partir du fichier mesh..txt
    #file = open("mesh"+str(RANK)+".txt","r")
    for line in file:
        #read elements
        if line == "elements\n":
            continue
        if line == "endelements\n":
            continue
        if line == "nodes\n":
            break
        Cells.nodeid.append([int(x) for x in line.split()])
    #Lecture des coordonnées des noeuds à partir du fichier mesh..txt
    for line in file:
        #read Nodes
        if line == "nodes\n":
            continue
        if line == "endnodes\n":
            continue
        if line == "halosint\n":
            break
        Nodes.vertex.append([np.double(x) for x in line.split()])
        Nodes.name = [0]*len(Nodes.vertex)

    for i in range(len(Nodes.vertex)):
        Nodes.name[i] = int(Nodes.vertex[i][3])

    #calcul du barycentre
    for i in range(len(Cells.nodeid)):
        s_1 = Cells.nodeid[i][0]
        s_2 = Cells.nodeid[i][1]
        s_3 = Cells.nodeid[i][2]

        x_1 = Nodes.vertex[s_1][0]
        y_1 = Nodes.vertex[s_1][1]
        x_2 = Nodes.vertex[s_2][0]
        y_2 = Nodes.vertex[s_2][1]
        x_3 = Nodes.vertex[s_3][0]
        y_3 = Nodes.vertex[s_3][1]

        Cells.center.append((1./3 * (x_1 + x_2 + x_3), 1./3*(y_1 + y_2 + y_3)))
        Cells.volume.append((1./2) * abs((x_1-x_2)*(y_1-y_3)-(x_1-x_3)*(y_1-y_2)))
        var1 = (x_2-x_1)*(y_3-y_1)-(y_2-y_1)*(x_3-x_1)
        if var1 < 0:
            Cells.nodeid[i][0] = s_1;   Cells.nodeid[i][1] = s_3; Cells.nodeid[i][2] = s_2

    tmp = [[] for i in range(len(Nodes.vertex))]
    longn = [0]*len(Nodes.vertex)
    for i in range(len(Cells.nodeid)):
        for j in range(3):
            tmp[Cells.nodeid[i][j]].append(i)
            longn[Cells.nodeid[i][j]] = longn[Cells.nodeid[i][j]] + 1
    
    longc = [0]*len(Cells.nodeid)
    tmp2 = [[] for i in range(len(Cells.nodeid))]
    for i in range(len(Cells.nodeid)):
        for j in range(3):
            for k in range(len(tmp[Cells.nodeid[i][j]])):
                if (tmp[Cells.nodeid[i][j]][k] not in tmp2[i] and  tmp[Cells.nodeid[i][j]][k] != i):
                    tmp2[i].append(tmp[Cells.nodeid[i][j]][k])
                    longc[i] = longc[i] + 1

    maxlongn = max(longn)
    Nodes.cellid = [[-1]*maxlongn for i in range(len(Nodes.vertex))]

    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            Nodes.cellid[i][j] = tmp[i][j]
    
    maxlongc = max(longc)
    Cells.cellnid = [[-1]*maxlongc for i in range(len(Cells.nodeid))]
    
    for i in range(len(tmp2)):
        for j in range(len(tmp2[i])):
            Cells.cellnid[i][j] = tmp2[i][j]

    #Création des faces
    cellule = Cells.nodeid
    cellf = []
    faces = []
    k = 0
    for i in range(len(cellule)):
        faces.append([cellule[i][0], cellule[i][1]])
        faces.append([cellule[i][1], cellule[i][2]])
        faces.append([cellule[i][2], cellule[i][0]])
        cellf.append([faces[k], faces[k+1], faces[k+2]])
        k = k+3
    for i in range(len(faces)):
        faces[i].sort()
    faces = set(tuple(x) for x in faces)
    faces = list(faces)


    facesdict = OrderedDict()
    for i in range(len(faces)):
        facesdict[faces[i]] = i
        Faces.nodeid.append(faces[i])
        Faces.cellid.append((-1, -1))

    #Création des 3 faces de chaque cellule
    for i in range(len(cellf)):
        Cells.faceid.append([facesdict.get(tuple(cellf[i][0])), facesdict.get(tuple(cellf[i][1])),
                             facesdict.get(tuple(cellf[i][2]))])

    #Faces.cellid = [(-1, -1)]*len(faces)
    for i in range(len(Cells.faceid)):
        for j in range(3):
            if Faces.cellid[Cells.faceid[i][j]] == (-1, -1):
                Faces.cellid[Cells.faceid[i][j]] = (i, -1)

            if Faces.cellid[Cells.faceid[i][j]][0] != i:
                Faces.cellid[Cells.faceid[i][j]] = (Faces.cellid[Cells.faceid[i][j]][0], i)

    Cells.cellfid = [[] for i in range(len(Cells.nodeid))]
    #Création des 3 triangles voisins par face
    for i in range(len(Cells.faceid)):
        for j in range(3):
            f = Cells.faceid[i][j]
            if Faces.cellid[f][1] != -1:
                if i == Faces.cellid[f][0]:
                    Cells.cellfid[i].append(Faces.cellid[f][1])
                else:
                    Cells.cellfid[i].append(Faces.cellid[f][0])
            else:
                Cells.cellfid[i].append(-1)
    
#    for i in range(len(faces)):
#        if Faces.cellid[i][1] != -1:
#            Cells.cellfid[Faces.cellid[i][1]].append(Faces.cellid[i][0])
#            Cells.cellfid[Faces.cellid[i][0]].append(Faces.cellid[i][1])
#
#    for i in range(len(Cells.cellfid)):
#        if len(Cells.cellfid[i]) == 2 :
#            Cells.cellfid[i].append(-1)

    #Faces aux bords (1,2,3,4), Faces à l'interieur 0    A VOIR !!!!!
    for i in range(len(faces)):
        Faces.name.append(0)
        if (Faces.cellid[i][1] == -1 and Faces.cellid[i][1] != -10):
            if Nodes.name[faces[i][0]] == Nodes.name[faces[i][1]]:
                Faces.name[i] = Nodes.name[faces[i][0]]
            if ((Nodes.name[faces[i][0]] == 3 and Nodes.name[faces[i][1]] != 0) or
                    (Nodes.name[faces[i][0]] != 0 and Nodes.name[faces[i][1]] == 3)):
                Faces.name[i] = 3
            if ((Nodes.name[faces[i][0]] == 4 and Nodes.name[faces[i][1]] != 0) or
                    (Nodes.name[faces[i][0]] != 0 and Nodes.name[faces[i][1]] == 4)):
                Faces.name[i] = 4
        normal, mesure = normalvector(np.asarray(Nodes.vertex[faces[i][0]]),
                                         np.asarray(Nodes.vertex[faces[i][1]]),
                                         Cells.center[Faces.cellid[i][0]])
        Faces.normal.append(normal)
        Faces.mesure.append(mesure)
        Faces.center.append([0.5 * (Nodes.vertex[faces[i][0]][0] + Nodes.vertex[faces[i][1]][0]),
                             0.5 * (Nodes.vertex[faces[i][0]][1] + Nodes.vertex[faces[i][1]][1])])
    for i in range(len(faces)):
        if Faces.name[i] != 0:
            Faces.bound += 1
        
    #compute the outgoing normal faces for each cell
    for i in range(len(Cells.nodeid)):
        ss = np.zeros((3, 2))

        G = np.asarray(Cells.center[i])

        for j in range(3):
            f = Cells.faceid[i][j]
            c = Faces.center[f]

            if np.dot(G-c, Faces.normal[f]) < 0.0:
                ss[j] = Faces.normal[f]
            else:
                ss[j] = -1.0*Faces.normal[f]
                
        Cells.nf.append(ss)

    Faces.ghostcenter = [[] for i in range(len(Faces.name))]
    Nodes.ghostcenter = [[] for i in range(len(Nodes.name))]
        
    #compute the ghostcenter for each face and each node
    for i in range(len(Faces.name)):
        nod1 = Faces.nodeid[i][1]
        nod2 = Faces.nodeid[i][0]
        if Faces.name[i] != 0 and Faces.name[i] != 10:
            
            x_1 = Nodes.vertex[nod1]
            x_2 = Nodes.vertex[nod2]
            
            c_left = Faces.cellid[i][0]
            v_1 = Cells.center[c_left]
            gamma = ((v_1[0] - x_2[0])*(x_1[0]-x_2[0]) + (v_1[1]-x_2[1])*(x_1[1]-x_2[1]))/((x_1[0]-x_2[0])**2 + (x_1[1]-x_2[1])**2)
           
            kk = np.array([gamma * x_1[0] + (1 - gamma) * x_2[0], gamma * x_1[1] + (1 - gamma) * x_2[1]])
            
            v_2 = np.array([2 * kk[0] + ( -1 * v_1[0]), 2 * kk[1] + ( -1 * v_1[1])])
            
            Faces.ghostcenter[i] = v_2
            Nodes.ghostcenter[nod1].append([v_2[0], v_2[1], i])
            Nodes.ghostcenter[nod2].append([v_2[0], v_2[1], i])
            
        else:
            Faces.ghostcenter[i] = [0,0]
            #Nodes.ghostcenter[nod1].append([0,0])
            #Nodes.ghostcenter[nod2].append([0,0])
            
    
    for i in range(len(Nodes.name)):
        if len(Nodes.ghostcenter[i]) == 0 :
            Nodes.ghostcenter[i].append([0,0,0])
            Nodes.ghostcenter[i].append([0,0,0])
        elif len(Nodes.ghostcenter[i]) == 1 :
            Nodes.ghostcenter[i].append([0,0,0])


def create_halo_structure(file):

    #txt_file = open(filename)
    for line in file:
        if "endhalosint" in line:
            break
        Halo.halosint.append(int(line))#.append(int(line))#for x in line.split()])

    for line in file:
        # process the line
        if "halosext" in line:
            continue
        if "centvol" in line:
            break
        Halo.halosext.append([int(x) for x in line.split()])

    for line in file:
        if "centvol" in line:
            continue
        if "globalcelltolocal" in line:
            break
        Halo.centvol.append([float(x) for x in line.split()])

    #Cells.globalindex = [-1]*len(Cells.nodeid)
    cmpt = 0
    for line in file:
    #read Global cell to local
        if line == "globalcelltomocal\n":
            continue
        if line == "endglobalcelltolocal\n":
            break
        Cells.globalindex[int(line)] = cmpt
        cmpt += 1

    cmpt = 0
    for line in file:
        #read Local Node To Global
        if line == "localnodetoglobal\n":
            continue
        if line == "endlocalnodetoglobal\n":
            break
        Nodes.globalindex[cmpt] = int(line)
        cmpt += 1
    for line in file:
        #read LocalToGlobal
        if line == "neigh\n":
            continue
        if line == "endneigh\n":
            break
        Halo.neigh.append([int(x) for x in line.split()])

    #halofaces = []
    Faces.halofid = np.zeros(len(Faces.nodeid), dtype=int)
    if SIZE > 1:
        k = 1
        for i in range(len(Halo.halosext)):
            Halo.faces[tuple([Halo.halosext[i][0], Halo.halosext[i][1]])] = k
            Halo.faces[tuple([Halo.halosext[i][1], Halo.halosext[i][2]])] = k + 1
            Halo.faces[tuple([Halo.halosext[i][0], Halo.halosext[i][2]])] = k + 2
            k = k+3

        for i in range(len(Faces.nodeid)):
            n1 = Nodes.globalindex[Faces.nodeid[i][0]]
            n2 = Nodes.globalindex[Faces.nodeid[i][1]]
            if  Halo.faces.get(tuple([n1, n2])):

                Faces.cellid[i] = (Faces.cellid[i][0], -10)
                Faces.name[i] = 10
                Nodes.name[Faces.nodeid[i][0]] = 10
                Nodes.name[Faces.nodeid[i][1]] = 10
                Faces.halofid[i] = int((-1+Halo.faces.get(tuple([n1, n2])))/3)

            if Halo.faces.get(tuple([n2, n1])):

                Faces.cellid[i] = (Faces.cellid[i][0], -10)
                Faces.name[i] = 10
                Nodes.name[Faces.nodeid[i][0]] = 10
                Nodes.name[Faces.nodeid[i][1]] = 10

                Faces.halofid[i] = int((-1+Halo.faces.get(tuple([n2, n1])))/3)

        longueur = 0
        tmp = [[] for i in range(len(Nodes.name))]
        for i in range(len(Nodes.name)):
            if Nodes.name[i] == 10:
                Halo.nodes[i] = Nodes.globalindex[i]
                arg = np.where(np.asarray(Halo.halosext) == Nodes.globalindex[i])
                tmp[i].append(arg[0])
                longueur = max(longueur, len(arg[0]))
            else:
                tmp[i].append(np.array([-1]))

        Nodes.halonid = [[-1]*longueur for i in range(len(Nodes.name))]

        for i in range(len(tmp)):
            for j in range(len(tmp[i][0])):
                Nodes.halonid[i][j] = tmp[i][0][j]

    cells = Cells(np.asarray(Cells.nodeid), np.asarray(Cells.faceid), np.asarray(Cells.center),
                  np.asarray(Cells.volume), Cells.father, Cells.son, Cells.iadiv, Cells.globalindex, 
                  np.asarray(Cells.cellfid), np.asarray(Cells.cellnid), np.asarray(Cells.nf))
    
    nodes = Nodes(np.asarray(Nodes.vertex), np.asarray(Nodes.name), np.asarray(Nodes.cellid),
                  Nodes.globalindex, np.asarray(Nodes.halonid), np.asarray(Nodes.ghostcenter))

    faces = Faces(np.asarray(Faces.nodeid), np.asarray(Faces.cellid), np.asarray(Faces.name),
                  np.asarray(Faces.normal), np.asarray(Faces.mesure), np.asarray(Faces.center), 
                  Faces.bound, np.asarray(Faces.halofid), np.asarray(Faces.ghostcenter))

    halos = Halo(np.asarray(Halo.halosint), np.asarray(Halo.halosext), np.asarray(Halo.centvol),
                 np.asarray(Halo.neigh), Halo.faces, Halo.nodes)

    return cells, nodes, faces, halos

@njit
def define_halosend(w_c, w_halosend, indsend):
    if SIZE > 1:
        for i in range(len(w_halosend)):
            w_halosend[i] = w_c[indsend[i]]

    return w_halosend

def all_to_all(w_halosend, taille, mystruct, variables, scount, sdepl, rcount, rdepl):

    w_halorecv = np.zeros(taille, dtype=mystruct)

    for var in literal_unroll(variables):
        s_msg = r_msg = 0
        s_msg = [np.array(w_halosend[var]), (scount, sdepl), MPI.DOUBLE_PRECISION]
        r_msg = [np.array(w_halorecv[var]), (rcount, rdepl), MPI.DOUBLE_PRECISION]

        COMM.Alltoallv(s_msg, r_msg)

        w_halorecv[var] = r_msg[0]

    return w_halorecv

@njit
def halo_value(w_halo, w_halorecv):

    for i in range(len(w_halo)):
        w_halo[i] = w_halorecv[i]

    return w_halo

def prepare_comm(cells, halos):

    scount = np.zeros(SIZE, dtype=int)
    sdepl = np.zeros(SIZE, dtype=int)
    rcount = np.zeros(SIZE, dtype=int)
    rdepl = np.zeros(SIZE, dtype=int)
    taille = 0
    indsend = 0

    if SIZE > 1:
        for i in range(len(halos.neigh[0])):
            scount[halos.neigh[0][i]] = halos.neigh[1][i]

        for i in range(SIZE):
            if i > 0:
                sdepl[i] = sdepl[i-1] + scount[i-1]

        rcount = COMM.alltoall(scount)

        for i in range(SIZE):
            if i > 0:
                rdepl[i] = rdepl[i-1] + rcount[i-1]

        for i in range(SIZE):
            taille += rcount[i]

        taille = int(taille)

        indsend = np.zeros(0, dtype=int)
        for i in range(len(halos.halosint)):
            indsend = np.append(indsend, cells.globalindex[Halo.halosint[i]])

#    indRecv = np.zeros(len(faces.cellid), dtype = int)
#    for i in range(len(faces.cellid)):
#        if faces.cellid[i][1] == -10:
#            indRecv[i] = int((-1+halos.faces.get(
#                    tuple([nodes.globalindex[Faces.nodeid[i][0]],
#                           nodes.globalindex[Faces.nodeid[i][1]]])))/3)

    return scount, sdepl, rcount, rdepl, taille, indsend

def generate_mesh():

    filename = 'mesh'+str(RANK)+'.txt'
    txt_file = open(filename)

    create_local_mesh(txt_file)

    cells, nodes, faces, halos = create_halo_structure(txt_file)

    grid = {}

    grid["cells"] = cells
    grid["nodes"] = nodes
    grid["faces"] = faces
    grid["halos"] = halos

    txt_file.close()

    return grid

def clear_class(cells, nodes, faces):

    cells.nodeid = []
    cells.volume = []
    cells.center = []
    cells.faceid = []
    cells.father = Dict()
    cells.son = Dict()
    cells.iadiv = Dict()
    cells.globalindex = OrderedDict()
    cells.cellfid = []
    cells.cellnid = []
    cells.nf = []

    faces.nodeid = []
    faces.name = []
    faces.cellid = []
    faces.normal = []
    faces.bound = 0
    faces.center = []
    faces.ghostcenter = []

    nodes.cellid = []
    nodes.name = []
    nodes.vertex = []
    nodes.globalindex = OrderedDict()
    nodes.halonid = []
    nodes.ghostcenter = []

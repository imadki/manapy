#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:41:28 2020

@author: kissami
"""
import numpy as np
import os
from manapy import ddm
from copy import deepcopy
from collections import OrderedDict 
from numba.typed import  Dict
from numba import njit


def refine_mesh(cells, nodes, faces):
    
    NODE = []
    ELE  = []
 
    
    NODE = deepcopy(nodes.vertex)
    ELE  = cells.nodeid
    nbnode = len(nodes.vertex)
    nbelement = len(cells.nodeid)
    nbedge = len(faces.nodeid)
    newnbnode = nbnode
    newnbelement = nbelement
    ladiv = Dict()#np.zeros(nbelement);
    tocut = []#OrderedDict()
    kcut = 0
    ELENEW = deepcopy(ELE)
    ELEOLD = deepcopy(ELE)
    
 
    marker = np.zeros(nbedge, dtype=int);
    
    for i,j in cells.Iadiv.items():
        for l in range(3):
            e = cells.faceid[i][l]
            netv1 = faces.nodeid[e][0];
            netv2 = faces.nodeid[e][1];
            if(marker[e]==0):
                marker[e]=newnbnode;
                xnew = 0.5*(NODE[netv1][0] + NODE[netv2][0]);
                ynew = 0.5*(NODE[netv1][1] + NODE[netv2][1]);
                
                NODE.resize(newnbnode+1, 4, refcheck=False)
                NODE[newnbnode] = [xnew, ynew, 0., faces.name[e]]
                newnbnode=newnbnode+1 

            if (faces.name[e] == 0):
                if(faces.cellid[e][0] == i):
                    L = faces.cellid[e][1]
                else:
                    L = faces.cellid[e][0]

                if(not(cells.Iadiv.get(L))):
                    ladiv[L] = -1
                    tocut.append(L)#[kcut] = L
                    kcut = kcut + 1                
        for nloc in range(3):
            ni = ELE[i][nloc]
            if(ni != netv1 and ni != netv2):
                nfe=ni;
                break
        

        ELENEW[i] = [nfe, marker[cells.faceid[i][0]], marker[cells.faceid[i][1]]]
        ELENEW.resize(len(ELENEW)+3, 3, refcheck=False)
       
        
        newnele1=newnbelement;
        ELENEW[newnele1] = [marker[cells.faceid[i][0]], marker[cells.faceid[i][1]], 
               marker[cells.faceid[i][2]]]
    
       # cells.father.rezise(len(cells.father)+1, refcheck=False)
        cells.father[newnele1] = i

        newnele2=newnele1+1;
        ELENEW[newnele2] = [netv1, marker[cells.faceid[i][0]], 
               marker[cells.faceid[i][2]]]
        
        newnele3=newnele2+1;
        ELENEW[newnele3] = [netv2, marker[cells.faceid[i][2]], 
               marker[cells.faceid[i][1]]]
        
        newnbelement = newnele3+1;
        
        cells.son[i] = np.array([i, newnele1, newnele2, newnele3])

        
    nbtr=len(tocut) ;
    nele=len(ELE);
    
    if (nbtr ==1):
        nbtr = 0
    
    itr=0;
    while(itr<nbtr):
        index=1;
        t=tocut[itr]
        while(index==1):
            if(ladiv[t]<0):
                #print(t, [t])
                nv=ELE[t][0]; ve1=ELE[t][1]; ve2=ELE[t][2];
                
                for ie in range(3):
                    e=cells.faceid[t][ie];
                    noded1=faces.nodeid[e][0]; noded2 = faces.nodeid[e][1]
                    if(nv != noded1 and nv != noded2):
                        base=e;
                        break

                if(marker[base]>0):
                    index=0;
                else:
                    vm=len(NODE);   marker[base]=vm;
                    xvm = 0.5*(NODE[ve1][0] + NODE[ve2][0]);
                    yvm = 0.5*(NODE[ve1][1] + NODE[ve2][1]);
                    
                    
                    name = nodes.name[ve1]*nodes.name[ve2]
                    NODE.resize(len(NODE)+1, 4, refcheck=False)
                    NODE[vm] = [xvm, yvm, 0., name]
                    
                    itright=faces.cellid[base][0];   itleft=faces.cellid[base][1];
                    if(itright*itleft>=0):
                        if(t==itright):
                            tneig=itleft;
                        else:
                            tneig=itright;
                        ladiv[tneig]=-1;   nbtr=nbtr+1;  tocut.append(tneig)#[nbtr-1]=tneig;
                    else:
                        index=0;         
            else:
                index=0;
        itr=itr+1;
    ELE   = ELENEW;
    

    for t in range(nele):
         if(not(cells.Iadiv.get(t))):
             cells.son[t] = np.array([t])
        
    for t,m in ladiv.items():
       
        
        if(ladiv[t] < 0 and not(cells.Iadiv.get(t)) ):
            n1=ELEOLD[t][0]; n2=ELEOLD[t][1];    n3=ELEOLD[t][2]; 
            for ie in range(3):
                e=cells.faceid[t][ie];
                noded1=faces.nodeid[e][0]; noded2 = faces.nodeid[e][1]
                if(n1 != noded1 and n1 != noded2):
                    base=e;
                    break

            if(marker[base]>0):
     
                nt1=len(ELE);
                ELE[t] = [n1, n2, marker[base]]
                ELE.resize(len(ELE)+1, 3, refcheck=False)
                ELE[nt1] = [n1, marker[base], n3]
                cells.son[t] = np.concatenate((cells.son[t],[nt1]), axis=0)
    
                for ie in range(3):
                    e=cells.faceid[t][ie];
                    noded1=faces.nodeid[e][0]; noded2 = faces.nodeid[e][1]
                    if(n2 != noded1 and n2 != noded2):
                        base2=e;
                        break

                
                
                if(marker[base2] > 0):
                    nt2=len(ELE);
                    ELE.resize(len(ELE)+1, 3, refcheck=False)
                    ELE[nt1] = [n1, marker[base], marker[base2]]
                    ELE[nt2] = [marker[base2], marker[base], n3]
                    #cells.son[t] = cells.son[t]+[nt2]
                    cells.son[t] = np.concatenate((cells.son[t],[nt2]), axis=0)

                for ie in range(3):
                    e=cells.faceid[t][ie];
                    noded1=faces.nodeid[e][0]; noded2 = faces.nodeid[e][1]
                    if(n3 != noded1 and n3 != noded2):
                        base3=e;
                        break
 
                if(marker[base3] > 0):
                    nt3=len(ELE);
                    ELE.resize(len(ELE)+1, 3, refcheck=False)
                    ELE[t] = [n1, marker[base3], marker[base]]
                    ELE[nt3] = [marker[base3], n2, marker[base]]
                    #cells.son[t] = cells.son[t]+[nt3]
                    cells.son[t] = np.concatenate((cells.son[t],[nt3]), axis=0)

    for i in range(len(ELE)):
        ELE[i].sort()
    

    def create_cell_nodeid(elements):
        cell_nodeid = []
        cell_nodeid = elements
        return cell_nodeid
    
    def create_nodes(points):
        nodes = []
        nodes = points
        return nodes
   
    
    #coordinates x, y of each node
    nodes = create_nodes(NODE)
    #nodes of each cell
    cell_nodeid = create_cell_nodeid(ELE)
    
    
    if os.path.exists("mesh"+str(0)+".txt"):
        os.remove("mesh"+str(0)+".txt")
        #else:
        #    print("Can not delete the file as it doesn't exists")
    
    with open("mesh"+str(0)+".txt", "a") as text_file:
        text_file.write("Elements\n")
        np.savetxt(text_file, cell_nodeid, fmt='%u')
        text_file.write("EndElements\n")
    
    with open("mesh"+str(0)+".txt", "a") as text_file:
        text_file.write("Nodes\n")
        np.savetxt(text_file, nodes)
        text_file.write("EndNodes\n")


    
    father = cells.father
    son = cells.son
    
    return father, son
    

# 

@njit
def interpolateCoarse2Fine(F2S, nbelements, u):
        
    uinter = np.zeros(nbelements)

    for i in range(len(u)):
        for j in range(len(F2S[i])):
            uinter[F2S[i][j]] = u[i]
    return uinter
@njit
def interpolateFine2Coarse(F2S,volume, u):
    uinter =  np.zeros(len(F2S))
    somme  =  np.zeros(len(F2S))
    
    for i in range(len(F2S)):
        for j in range(len(F2S[i])):
            uinter[i] += volume[F2S[i][j]] * u[F2S[i][j]]
            somme[i]  += volume[F2S[i][j]]
       
        uinter[i] /= somme[i]
    
    return uinter

def compute_criterion(w, cells, faces):
    
    grad = compute_gradient(w, cells.nodeid, faces.cellid, faces.name, faces.normal, cells.volume)
    maxu = max(grad)
    
    tol = 0.0025*maxu
    criterion = Dict()#[0]*len(cells.nodeid)
    
    for i in range(len(cells.nodeid)):
        if (grad[i] > tol):
            criterion[i] = 1
    
    return criterion

def multilevelrefine(w, g_ref, g):

   
    w_coarse = interpolateFine2Coarse(g["cells"].son, g_ref["cells"].volume, w["h"])
    g["cells"].Iadiv = compute_criterion(w_coarse, g["cells"], g["faces"])
    father, son = refine_mesh(g["cells"], g["nodes"], g["faces"])
    g = ddm.generate_mesh()
    
  
    winterp1 = interpolateCoarse2Fine(son, len(g["cells"].nodeid), w_coarse)
    g["cells"].Iadiv = compute_criterion(winterp1, g["cells"], g["faces"])
    father1, son1 = refine_mesh(g["cells"], g["nodes"], g["faces"])
    g = ddm.generate_mesh()
    

    

    F2S = [[] for i in range(len(son))]
    for i, j in son.items():
        for l in range(len(j)):   
            for k in range(len(son1[j[l]])):
                F2S[i].append(son1[j[l]][k])

    for i in range(len(F2S)):
        g["cells"].son[i] = np.array(F2S[i])
        
        

    return g


    
def set_old_information(g_old, g, g_coarse, w_old, variables, mystruct):
   
    nbelements = len(g["cells"].nodeid)
    w = np.recarray(nbelements, dtype = mystruct )

    v1 = {}
    v1 = deepcopy(w)
    
    cell = g_old["cells"]
    volume = np.asarray(cell.volume)
    for i,j in g_old["cells"].son.items():
        for var in variables:
            v1[var][i] = sum(w_old[var][j[0:len(j)]] * volume[j[0:len(j)]])
            v1[var][i] /= g_coarse["cells"].volume[i]
    

    
    for i in range(len(g["cells"].son)):
        if (len(g_old["cells"].son[i]) == len(g["cells"].son[i])):
            if (len(w_old) > len(w)):
                for j in range(len(g["cells"].son[i])): 
                    for var in variables:
                        w[var][g["cells"].son[i][j]] = w_old[var][g_old["cells"].son[i][j]]
            else:
                for j in range(len(g_old["cells"].son[i])):
                    for var in variables:
                        w[var][g["cells"].son[i][j]] = w_old[var][g_old["cells"].son[i][j]]                  
        else:
            for j in range(len(g["cells"].son[i])):
                for var in variables:
                    w[var][g["cells"].son[i][j]] = v1[var][i]
                
    return w


@njit
def compute_gradient(w, nodeid, cellid, name, normal, volume):
   
    h_n = np.zeros((len(w), 2))
    grad = np.zeros(len(w))
    
    for i in range(len(name)):
        fleft = cellid[i][0]
        fright = cellid[i][1]
        
       
        if name[i] == 0:
            h = 0.5 * (w[fleft] + w[fright])
            h_n[fleft] +=  h * normal[i]
            h_n[fright] -= h * normal[i]
        else:
            h_n[fleft] += w[fleft] * normal[i]
    
    for i in range(len(h_n)):
        h_n[i] /= volume[i]
       # print(h_n[i][0], h_n[i][1])
        grad[i] = np.sqrt(pow(h_n[i][0],2) + pow(h_n[i][1],2))
#          
    return grad
    
#
    
    
    
    
    
    
    

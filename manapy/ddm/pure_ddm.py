#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:07:32 2023

@author: kissami
"""

import os
from collections import OrderedDict
from numpy import (zeros, int32, arange)

def read_mesh_file(size, rank, mesh_dir, precision):
    import h5py
    
    neigh = []; nparts = []; neigh = []
    Cellglobtoloc = OrderedDict()
    
    filename = os.path.join(mesh_dir, 'mesh'+str(rank)+'.hdf5')
    
    if size == 1:
        with h5py.File(filename, "r") as f:
            nodeid = f["elements"][:]
            vertex = f["nodes"][:]
            #tc to gather solution of linear system
            tc = arange(len(nodeid), dtype=int32)
            Cellloctoglob = arange(len(nodeid), dtype=int32)
            Cellglobtoloc = arange(len(nodeid), dtype=int32)
            Nodeloctoglob = arange(len(vertex), dtype=int32)
            halosext      = zeros((2,2), dtype=int32)
            halosint      = zeros((2,2), dtype=int32)
            centvol  = zeros((2,2), precision); 
            
        f.close()
    else:
        with h5py.File(filename, "r") as f:
            nodeid = f["elements"][:]
            vertex = f["nodes"][:]
            neigh = []
            tc = 0.
            if rank == 0:
                tc = f["GtoL"][:]
            centvol = f["centvol"][:].astype(precision)
            Nodeloctoglob = f["localnodetoglobal"][:]
            Cellloctoglob = f["localcelltoglobal"][:]
            neigh.append(list(f["neigh1"][:]))
            neigh.append(list(f["neigh2"][:]))
            halosint  = f["halosint"][:]
            halosext = f["halosext"][:]
            npart = f["nparts"][:]
            
        f.close()
        
        nparts = OrderedDict()
        for i in range(len(npart)):
            nparts[npart[i][-1]] = npart[i][:npart[i][-2]]

        cmpt = 0
        Cellglobtoloc = OrderedDict() 
        for i in Cellloctoglob:
            Cellglobtoloc[i] = cmpt  
            cmpt +=1

    if size > 1:
        Globnodetoloc = {}
        for i in range(len(Nodeloctoglob)):
            Globnodetoloc[Nodeloctoglob[i]] = i
            
        for i in range(len(nodeid)):
            if nodeid[i][-1] == 3:
                nodeid[i][0] = Globnodetoloc[nodeid[i][0]]
                nodeid[i][1] = Globnodetoloc[nodeid[i][1]]
                nodeid[i][2] = Globnodetoloc[nodeid[i][2]]
                
            elif nodeid[i][-1] == 4:
                nodeid[i][0] = Globnodetoloc[nodeid[i][0]]
                nodeid[i][1] = Globnodetoloc[nodeid[i][1]]
                nodeid[i][2] = Globnodetoloc[nodeid[i][2]]
                nodeid[i][3] = Globnodetoloc[nodeid[i][3]]
                
            elif nodeid[i][-1] == 5:
                nodeid[i][0] = Globnodetoloc[nodeid[i][0]]
                nodeid[i][1] = Globnodetoloc[nodeid[i][1]]
                nodeid[i][2] = Globnodetoloc[nodeid[i][2]]
                nodeid[i][3] = Globnodetoloc[nodeid[i][3]]
                nodeid[i][4] = Globnodetoloc[nodeid[i][4]]
            
            elif nodeid[i][-1] == 8:
                nodeid[i][0] = Globnodetoloc[nodeid[i][0]]
                nodeid[i][1] = Globnodetoloc[nodeid[i][1]]
                nodeid[i][2] = Globnodetoloc[nodeid[i][2]]
                nodeid[i][3] = Globnodetoloc[nodeid[i][3]]
                nodeid[i][4] = Globnodetoloc[nodeid[i][4]]
                nodeid[i][5] = Globnodetoloc[nodeid[i][5]]
                nodeid[i][6] = Globnodetoloc[nodeid[i][6]]
                nodeid[i][7] = Globnodetoloc[nodeid[i][7]]
                
                
  
              

    return tc, nodeid, vertex, halosint, halosext, centvol, Cellglobtoloc, Cellloctoglob, Nodeloctoglob, neigh, nparts

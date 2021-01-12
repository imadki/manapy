#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:51:37 2020

@author: kissami
"""
import timeit
import os
import numpy as np
from mpi4py import MPI
from manapy import ddm
from numba import njit


@njit(fastmath=True)
def update(w_c, wnew, dtime, rez, vol):
    nbelement = len(w_c)

    for i in range(nbelement):
        wnew.h[i] = w_c.h[i]  + dtime  * rez["h"][i]/vol[i] 

    return wnew

@njit(fastmath=True)
def explicitscheme_convective(w_c, w_ghost, w_halo, cellidf, normal, halofid,
                              name, mystruct):

    rezidus = np.zeros(len(w_c), dtype=mystruct)
    w_l = np.zeros(1, dtype=mystruct)[0]
    w_r = np.zeros(1, dtype=mystruct)[0]
    nbface = len(cellidf)

    flx = np.zeros(1, dtype=mystruct)[0]


    for i in range(nbface):
        
        w_l = w_c[cellidf[i][0]]
        norm = normal[i]

        if name[i] == 0:
            w_r = w_c[cellidf[i][1]]
            
            flx = rusanov_scheme(flx, w_l, w_r, norm)
            rezidus[cellidf[i][0]] = ddm.minus(rezidus[cellidf[i][0]], flx)
            rezidus[cellidf[i][1]] = ddm.add(rezidus[cellidf[i][1]], flx)

        elif name[i] == 10:
            w_r = w_halo[halofid[i]]

            flx = rusanov_scheme(flx, w_l, w_r, norm)
            rezidus[cellidf[i][0]] = ddm.minus(rezidus[cellidf[i][0]], flx)

        else:
            w_r = w_ghost[i]
            flx = rusanov_scheme(flx, w_l, w_r, norm)
            rezidus[cellidf[i][0]] = ddm.minus(rezidus[cellidf[i][0]], flx)
        
    return rezidus

@njit(fastmath=True)
def upwind_scheme(flux, wleft, wright, normal):

    sol = 0
    vel = np.zeros(2)

    vel[0] = 0.5*(wleft.hu + wright.hu)
    vel[1] = 0.5*(wleft.hv + wright.hv)
    
    sign = np.dot(vel, normal)

    if sign >= 0:
        sol = wleft.h
    else:
        sol = wright.h

    flux.h = sign * sol
 
    return flux


@njit(fastmath=True)
def rusanov_scheme(flux, wleft, wright, normal):

    q_l = wleft.hu * normal[0] + wleft.hv * normal[1]
    q_r = wright.hu * normal[0] + wright.hv * normal[1]

    f_l = np.fabs(q_l)
    f_r = np.fabs(q_r)

    if f_l > f_r:
        s_lr = f_l
    else:
        s_lr = f_r

    flux.h = 0.5 * (q_l + q_r) - 0.5 * s_lr * (wright.h - wleft.h)
   

    return flux

@njit(fastmath=True)
def initialisation(w_c, center):

    nbelements = len(center)
    sigma = 0.25
    x_1 = -1
    y_1 = 0
    
    for i in range(nbelements):
        xcent = center[i][0]
        ycent = center[i][1]
        
        w_c.h[i] = np.exp(-1* (pow(xcent - x_1, 2) + pow(ycent -y_1, 2))/ pow(sigma, 2))
        w_c.hu[i] = 1.
        w_c.hv[i] = 0.
        w_c.Z[i] = 0.
            
    return w_c
            
@njit(fastmath=True)
def time_step(w_c, cfl, normal, mesure, volume, faceid):
    nbelement =  len(faceid)
    dt_c = 1e6

    for i in range(nbelement):
        lam = 0
        for j in range(3):
            norm = normal[faceid[i][j]]
            mesn = mesure[faceid[i][j]]
            u_n = np.fabs(w_c.hu[i]*norm[0] + w_c.hv[i]*norm[1])
            lam_convect = u_n/mesn
            lam += lam_convect * mesn

        if lam != 0:                 
            dt_c = min(dt_c, cfl * volume[i]/lam)

    dtime = np.asarray(dt_c)#np.min(dt_c))

    return dtime

def ghost_value(w_c, w_ghost, cellid, name, normal, mesure, time):

    nbface = len(cellid)
    
    for i in range(nbface):
        w_ghost[i] = w_c[cellid[i][0]]
    
    return w_ghost


def test_tep():
    # ... get the mesh directory
    try:
        MESH_DIR = os.environ['MESH_DIR']
    
    except:
        BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        BASE_DIR = os.path.join(BASE_DIR, '..', '..')
        MESH_DIR = os.path.join(BASE_DIR, 'mesh')
    
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    
    #def test_tep():
    
    if RANK == 0:
        #reading gmsh file and partitioning into size subdomains
        filename = os.path.join(MESH_DIR, "carre.msh")
        ddm.meshpart(SIZE, filename)
        #removing existing vtk files
        mypath = "results"
        if not os.path.exists(mypath):
            os.mkdir(mypath)
        for root, dirs, files in os.walk(mypath):
            for file in files:
                os.remove(os.path.join(root, file))
    COMM.Barrier()
    
    start = timeit.default_timer()
    
    #generating local grid for each subdomain
    grid = {}
    grid = ddm.generate_mesh()
    
    faces = grid["faces"]
    cells = grid["cells"]
    halos = grid["halos"]
    nodes = grid["nodes"]
    
    nbelements = len(cells.center)
    nbfaces = len(faces.name)
    
    variables = tuple(['h', 'hu', 'hv', 'hc', 'Z'])
    mystruct = np.dtype([('h', np.float64),
                         ('hu', np.float64),
                         ('hv', np.float64),
                         ('hc', np.float64),
                         ('Z', np.float64),])
    
    w_c = np.recarray(nbelements, dtype=mystruct)
    rez = np.recarray(nbelements, dtype=mystruct)
    w_ghost = np.recarray(nbfaces, dtype=mystruct)
    uexact = np.zeros(nbelements)
    
    #compute the arrays needed for the mpi communication
    scount, sdepl, rcount, rdepl, taille, indsend = ddm.prepare_comm(cells, halos)
    
    w_halosend = np.zeros(len(halos.halosint), dtype=mystruct)
    
    w_n = w_c
    ###Initialisation
    w_c = initialisation(w_c, cells.center)
    
    if RANK == 0: print("Start Computation")
    cfl = 0.5
    time = 0
    tfinal = 2
    
    ####calculation of the time step
    d_t = time_step(w_c, cfl, faces.normal, faces.mesure, cells.volume, cells.faceid)
    dt_i = np.zeros(1)
    COMM.Allreduce(d_t, dt_i, MPI.MIN)
    d_t = np.float64(dt_i)
    #saving 25 vtk file
    tot = int(tfinal/d_t/50) + 1
    miter = 0
    niter = 0
    
    #loop over time
    while time < tfinal:
    
        time = time + d_t
    
        #update the ghost values for the boundary conditions
        w_ghost = ghost_value(w_c, w_ghost, faces.cellid, faces.name, faces.normal, faces.mesure,
                                  time)
        
       #update the halo values
        w_halosend = ddm.define_halosend(w_c, w_halosend, indsend)
        w_halo = ddm.all_to_all(w_halosend, taille, mystruct, variables,
                                scount, sdepl, rcount, rdepl)
    
    
        #update the rezidus using explicit scheme
        rez = explicitscheme_convective(w_c, w_ghost, w_halo,
                                        faces.cellid, faces.normal, faces.halofid, 
                                        faces.name, mystruct)
    
        #update the new solution
        w_n = update(w_c, w_n, d_t, rez, cells.volume)
        w_c = w_n
    
        #save vtk files for the solution
        if niter%tot == 0:
            ddm.save_paraview_results(w_c, uexact, niter, miter, time, d_t, RANK, SIZE,
                                      cells.nodeid, nodes.vertex)
    
            miter += 1
    
        niter += 1
    
    stop = timeit.default_timer()
    
    if RANK == 0: print(stop - start)

test_tep()

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
def update_friction(w_tild, w_n, cor, grav, d_t, nbelements):
    vent = [0, 0]
    eta = 0.03
    A1 = np.zeros(nbelements)
    A2 = np.zeros(nbelements)
    
    B = np.zeros(nbelements)
    
    for i in range(nbelements):
        
        ufric = w_tild[i].hu/w_tild[i].h
        vfric = w_tild[i].hv/w_tild[i].h
        hfric = w_tild[i].h
        
        A1[i] = w_tild[i].hu + d_t * (vent[0] + cor["hu"][i])
        A2[i] = w_tild[i].hv + d_t * (vent[1] + cor["hv"][i])
        
        B[i] = 1 + d_t * eta**2 * grav * np.sqrt(ufric**2 + vfric**2)/(pow(hfric, 4/3))

        
        w_n[i].h = w_tild[i].h
        w_n[i].hu = A1[i]/B[i]
        w_n[i].hv = A2[i]/B[i]
        w_n[i].hc = w_tild[i].hc
        w_n[i].Z = w_tild[i].Z
        
    return w_n

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

def test_swep():

    if RANK == 0:
        #reading gmsh file and partitioning into size subdomains
        filename = os.path.join(MESH_DIR, "Mesh200.msh")
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
    nbnodes = len(nodes.vertex)

    variables = tuple(['h', 'hu', 'hv', 'hc', 'Z'])
    mystruct = np.dtype([('h', np.float64),
                         ('hu', np.float64),
                         ('hv', np.float64),
                         ('hc', np.float64),
                         ('Z', np.float64),])

    w_c = np.recarray(nbelements, dtype=mystruct)
    w_x = np.recarray(nbelements, dtype=mystruct)
    w_y = np.recarray(nbelements, dtype=mystruct)
    src = np.recarray(nbelements, dtype=mystruct)
    rez = np.recarray(nbelements, dtype=mystruct)
    cor = np.recarray(nbelements, dtype=mystruct)
    dissip = np.recarray(nbelements, dtype=mystruct)
    w_ghost = np.recarray(nbfaces, dtype=mystruct)
    wx_face = np.zeros(nbfaces)
    wy_face = np.zeros(nbfaces)
    hexact = np.zeros(nbelements)
    uexact = np.zeros(nbelements)
    
    #compute the arrays needed for the mpi communication
    scount, sdepl, rcount, rdepl, taille, indsend = ddm.prepare_comm(cells, halos)
    #compute the interpolation variables
    R_x, R_y, lambda_x,lambda_y, number =  ddm.variables(cells.center, nodes.cellid, nodes.halonid,
                                                     nodes.vertex, nodes.name, nodes.ghostcenter,
                                                     halos.centvol)

    w_halosend = np.zeros(len(halos.halosint), dtype=mystruct)
    wx_halosend = np.zeros(len(halos.halosint), dtype=mystruct)
    wy_halosend = np.zeros(len(halos.halosint), dtype=mystruct)

    w_n = w_c
    ###Initialisation
    w_c = ddm.initialisation(w_c, cells.center)
    
    if RANK == 0: print("Start Computation")
    cfl = 0.5
    time = 0
    tfinal = 8
    order = 1#(1:first order, 2:barth jeperson 3:van albada 4:minmod)
    term_convec = "on"
    term_source = "on"
    term_coriolis = "off"
    term_dissipative = "off"
    grav = 9.81
    f_c = 0.
    D_xx = 0.1
    D_yy = 0.1

    ####calculation of the time step
    d_t = ddm.time_step(w_c, cfl, faces.normal, faces.mesure, cells.volume, cells.faceid, grav, 
                        term_dissipative, term_convec, D_xx, D_yy)
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
        w_ghost = ddm.ghost_value(w_c, w_ghost, faces.cellid, faces.name, faces.normal, faces.mesure,
                                  time)
        
       #update the halo values
        w_halosend = ddm.define_halosend(w_c, w_halosend, indsend)
        w_halo = ddm.all_to_all(w_halosend, taille, mystruct, variables,
                                scount, sdepl, rcount, rdepl)
        if order != 1:
            #compute derivative
            w_x, w_y = ddm.derivxy(w_c, w_ghost, w_halo, cells.center, cells.nodeid, cells.cellnid, 
                                   nodes.cellid, nodes.halonid, nodes.name, nodes.ghostcenter, 
                                   halos.centvol, w_x, w_y)
        
        #update the halo  derivatives values
        wx_halosend = ddm.define_halosend(w_x, wx_halosend, indsend)
        wy_halosend = ddm.define_halosend(w_y, wy_halosend, indsend)

        wx_halo = ddm.all_to_all(wx_halosend, taille, mystruct, variables,
                                 scount, sdepl, rcount, rdepl)
        wy_halo = ddm.all_to_all(wy_halosend, taille, mystruct, variables,
                                 scount, sdepl, rcount, rdepl)
        
        w_node = np.zeros(nbnodes, dtype=mystruct)
        unode_exact = np.zeros(nbnodes)

        if (term_convec == "on"):
            #update the rezidus using explicit scheme
            rez = ddm.explicitscheme_convective(w_c, w_x, w_y, w_ghost, w_halo, wx_halo, wy_halo,
                                               faces.cellid, cells.faceid, cells.nodeid, cells.center, 
                                               cells.cellfid, halos.centvol, faces.mesure,
                                               faces.center, faces.normal, faces.halofid, faces.name, 
                                               faces.ghostcenter, nodes.cellid, mystruct, order, grav)
        if (term_source == "on"):
            #update the source using explicit scheme
            src = ddm.term_source_srnh(w_c, w_ghost, w_halo, w_x, w_y, wx_halo, wy_halo, cells.nodeid, 
                                      cells.faceid, cells.cellfid, cells.center, cells.volume, cells.nf,
                                      faces.cellid, faces.nodeid, faces.normal, faces.center, faces.name, 
                                      faces.ghostcenter, nodes.vertex, faces.halofid, halos.centvol,
                                      mystruct, order, grav, src)
#            src = ddm.term_source_vasquez(w_c, w_ghost, w_halo, cells.nodeid, cells.faceid, cells.cellfid,
#                                          cells.center, cells.volume, faces.cellid, faces.nodeid,
#                                          faces.normal, faces.mesure, faces.name, faces.halofid, 
#                                          faces.center, nodes.vertex, mystruct, grav)
#
        if (term_dissipative == "on"):
            w_node, unode_exact = ddm.centertovertex(w_c, w_ghost, w_halo, cells.center, halos.centvol, 
                                                     nodes.cellid, nodes.halonid, nodes.vertex, nodes.name,
                                                     nodes.ghostcenter, w_node, hexact, unode_exact,
                                                     R_x, R_y, lambda_x,lambda_y, number)
            #update the dissipation of hc
            dissip = ddm.explicitscheme_dissipative(w_c, w_ghost, w_halo, w_node, wx_face, wy_face, 
                                                         cells.faceid, cells.center, faces.cellid, 
                                                         faces.nodeid, faces.center, faces.normal,
                                                         faces.name, faces.ghostcenter, faces.halofid, 
                                                         nodes.vertex, halos.centvol, mystruct, order,
                                                         grav, D_xx, D_yy) 
        

        #update the new solution
        w_tild = ddm.update(w_c, w_n, d_t, rez, src, cor, dissip, cells.volume)
        
        if (term_coriolis == "on"):
            #update the coriolis forces
            cor = ddm.term_coriolis(w_tild, f_c, mystruct)
            
        #update the friction
        w_n = update_friction(w_tild, w_n, cor, grav, d_t, nbelements)
        
        w_c = w_n
        
       
        
        #save vtk files for the solution
        if niter%tot == 0:
            saving_at_node = 1
            #uexact, hexact = ddm.exact_solution(uexact, hexact, zexact, cells.center, time, grav)
           
            if saving_at_node:
                w_node, unode_exact = ddm.centertovertex(w_c, w_ghost, w_halo, cells.center, halos.centvol, 
                                                     nodes.cellid, nodes.halonid, nodes.vertex, nodes.name,
                                                     nodes.ghostcenter, w_node, uexact, unode_exact,
                                                     R_x, R_y, lambda_x,lambda_y, number)
                
                ddm.save_paraview_results(w_node, unode_exact, niter, miter, time, d_t, RANK, SIZE,
                                          cells.nodeid, nodes.vertex)
           
            else:
                ddm.save_paraview_results(w_c, uexact, niter, miter, time, d_t, RANK, SIZE,
                                      cells.nodeid, nodes.vertex)


            miter += 1

        niter += 1
        ####calculation of the time step
        d_t = ddm.time_step(w_c, cfl, faces.normal, faces.mesure, cells.volume, cells.faceid, grav, 
                            term_dissipative, term_convec, D_xx, D_yy)
        dt_i = np.zeros(1)
        COMM.Allreduce(d_t, dt_i, MPI.MIN)
        d_t = np.float64(dt_i)
        #saving 25 vtk file
        tot = int(tfinal/d_t/50)


    stop = timeit.default_timer()

    if RANK == 0: print(stop - start)

test_swep()

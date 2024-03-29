from mpi4py import MPI
from numba import njit
import numpy as np

import timeit

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from manapy.partitions import MeshPartition
from manapy.ddm import Domain

from manapy.ast.functions2d import (get_rhs_glob_2d_with_contrib, get_triplet_2d_with_contrib,
                                    )
from manapy.solvers.advecdiff import explicitscheme_convective_2d, update_new_value, time_step
#from manapy.solvers.shallowater.tools_utils import initialisation_SW

from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.solvers.ls import PETScKrylovSolver, MUMPSSolver, ScipySolver

import os

start = timeit.default_timer()

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..','..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 

def face_gradient_info_2d(cellidf:'int32[:,:]', nodeidf:'int32[:,:]', centergf:'float[:,:]', namef:'int32[:]', normalf:'float[:,:]', 
                          centerc:'float[:,:]',  centerh:'float[:,:]', halofid:'int32[:]', vertexn:'float[:,:]', 
                          airDiamond:'float[:]', param1:'float[:]', param2:'float[:]', param3:'float[:]', param4:'float[:]', 
                          f_1:'float[:,:]', f_2:'float[:,:]', f_3:'float[:,:]', f_4:'float[:,:]', shift:'float[:,:]', 
                          dim:'int32', Kx:'float[:]', Ky:'float[:]'):
    
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
        
        n1 = normalf[i][0]*Kx[i]
        n2 = normalf[i][1]*Ky[i]
        
        airDiamond[i] = 0.5 *((xy_2[0] - xy_1[0]) * (v_2[1]-v_1[1]) + (v_1[0]-v_2[0]) * (xy_2[1] - xy_1[1]))
        
        param1[i] = 1./(2.*airDiamond[i]) * ((f_1[i][1]+f_2[i][1])*n1 - (f_1[i][0]+f_2[i][0])*n2)
        param2[i] = 1./(2.*airDiamond[i]) * ((f_2[i][1]+f_3[i][1])*n1 - (f_2[i][0]+f_3[i][0])*n2)
        param3[i] = 1./(2.*airDiamond[i]) * ((f_3[i][1]+f_4[i][1])*n1 - (f_3[i][0]+f_4[i][0])*n2)
        param4[i] = 1./(2.*airDiamond[i]) * ((f_4[i][1]+f_1[i][1])*n1 - (f_4[i][0]+f_1[i][0])*n2)

@njit(cache=True)
def compute_Pexact_Uimp(Pexact, Iexact, perm, visc, fi, Uin, t, x0, x):
    xf = x0 + Uin*t

    for i in range(len(x)):
        if x[i] > xf :
            Pexact[i] = 0.
            Iexact[i] = 0.
        else:
            Pexact[i] = (visc[i]*fi/perm[i]) * Uin * (xf - x[i])
            Iexact[i] = 1.

@njit
def compute_Pexact_Pimp(Pexact, Iexact,  perm0, visc0, fi, Pin, t, x0, x):
    xff = np.zeros(len(x))
    xf = np.zeros(len(x))

    xff =  2* (perm0/fi*visc0) * Pin*t + x0**2
    xf  = np.sqrt(xff)

    for i in range(len(x)):
        if x[i] > xf :
            Pexact[i] = 0.
            Iexact[i] = 0.
        else:
            Pexact[i] = Pin*(1. - x[i]/xf[i])
            Iexact[i] = 1.


@njit(cache=True)
def update_ghost_values_U(ughost , vghost, U_n, normal_faces, mesure, neumannNHfaces):

    for i in neumannNHfaces:
        normal = -normal_faces[i][0:2]/mesure[i]
        U_n_ = U_n*normal
        x = np.array((1., 0))
        y = np.array((0, 1.))
        #norm = np.sqrt()
        #teta = math.acos(np.dot(x, U_n*normal)/U_n)

        ughost[i] = np.dot(U_n_, x)
        vghost[i] = np.dot(U_n_, y)

@njit(cache=True)
def tau_remplissage(I):

    Tau_remp = sum(I > 0.9) / len(I)
    return Tau_remp
# Simulation parametres
##############################################################################
##############################################################################
test_para = 'test_2'

if test_para == "test_1" :
    fi    = 0.81
    U_n  = 3.e-3
    Pin = 1e5
    perm0 = 6.83e-9
    mu0 = 0.3
    tfinal = 122

    test = "pression"
    filename = "Geom_exp3.msh"


if test_para == "test_2" :
    C0 = 0.
    alpha0 = 1.
    sigma_u = 0.85
    a = 1.
    A = 0.68
    Pin = 3e5
    mu0  = 0.109
    fi0 =  0.45
    U_n  = 1e-1
    tfinal = 10

    test = "pression"
    filename = "TMesh.msh"


#File name
filename = os.path.join(MESH_DIR, filename)
dim = np.int32(2)

###Config###
#backend numba or python
#signature: add types to functions (make them faster) but compilation take time
#cache: avoid recompilation in the next run
running_conf = Struct(backend="numba", signature=True, float_precision="double", multithreading="single", cache=True)
mesh = MeshPartition(filename, dim=dim, conf=running_conf, periodic=[0,0,0])

#Create the informations about cells, faces and nodes
domain = Domain(dim=dim, conf=running_conf)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells
backend = domain.backend
signature = domain.signature

#TODO choix du test et du schéma

scheme = "Diamond"
#scheme = "FV4"
##################################################################################
##################################################################################

boundariesI = {"in" : "dirichlet",
                "out" : "neumann",
                "upper":"neumann",
                "bottom":"neumann"
            }
valuesI = {"in" : 1}

I = Variable(domain=domain, BC=boundariesI, values=valuesI,  terms = ["Flux"])
I.update_ghost_value()

# Concentration
boundariesC = {"in" : "dirichlet",
                "out" : "neumann",
                "upper":"neumann",
                "bottom":"neumann"
            }
valuesC = {"in" : C0}
C = Variable(domain=domain, BC=boundariesC, values=valuesC,  terms = ["Flux"])
C.update_ghost_value()

fiC = Variable(domain=domain, terms = ["Flux"])

perm_x_0  = Variable(domain=domain)
perm_y_0  = Variable(domain=domain)

perm_x  = Variable(domain=domain)
perm_y  = Variable(domain=domain)

visc  = Variable(domain=domain)
fi  = Variable(domain=domain)
sigma = Variable(domain=domain)
alpha = Variable(domain=domain)

## initialization of viscosity, porosity and filtration coieficient
visc.cell[:]  = mu0
fi.cell[:]  = fi0
alpha.cell[:] = alpha0
visc.update_ghost_value()
fi.update_ghost_value()
alpha.update_ghost_value()

## initialization of the permeability
for i in range(nbcells):
    if cells.center[i][1] >= 0.0014:
        perm_x_0.cell[i] =  2.e-11
        perm_y_0.cell[i] =  2.e-11

    else:
        perm_x_0.cell[i] =  2.e-11
        perm_y_0.cell[i] =  2.e-11

# Perm changes over time
perm_x.cell[:] = perm_x_0.cell[:]
perm_y.cell[:] = perm_y_0.cell[:]

perm_x.update_ghost_value()
perm_y.update_ghost_value()

## injection mode
if test == "pression":
    boundariesP = {"in" : "dirichlet",
                  "out" : "dirichlet",
                  "upper":"neumann",
                  "bottom":"neumann"
                }
    valuesP = {"in" : Pin, "out": 0. }
    boundariesU = {"in" : "neumann",
                  "out" : "neumann",
                  "upper":"noslip",
                  "bottom":"noslip"}
    
    u  = Variable(domain=domain, BC=boundariesU)
    v  = Variable(domain=domain, BC=boundariesU)


elif test == "debit":
    boundariesP = {"in" : "neumannNH",
                  "out" : "dirichlet",
                  "upper":"neumann",
                  "bottom":"neumann"
                 }
    cst = np.float64((mu0/perm0)*U_n)
    valuesP = {"in" : cst, "out": 0. }

    boundariesU = {"in" : "dirichlet",
                  "out" : "neumann",
                  "upper":"neumann",
                  "bottom":"neumann"
                }
    valuesU = {"in": U_n}
    u  = Variable(domain=domain)#, BC=boundariesU, values=valuesU)

v  = Variable(domain=domain)
w  = Variable(domain=domain)

P = Variable(domain=domain, BC=boundariesP, values=valuesP)
Pexact = Variable(domain=domain)
Iexact = Variable(domain=domain)

#TODO tfinal
if RANK == 0: print("Start Computation")

for i in domain.infaces:
    K = faces.cellid[i][0]
    I.cell[K] = 1.

x0    = 0.
cst = np.float64(0.)
time = 0
miter = 0
niter = 1
saving_at_node = 1
order = np.int32(2)
cfl = np.float64(0.8)
alpha_para = np.float64(2e-6)

dissip_I = np.zeros(nbcells)
src_C = np.zeros(nbcells)
src_I = np.zeros(nbcells)
div = np.zeros(nbcells)

conf = Struct(reuse_mtx=False, with_mtx=True, scheme='diamond', verbose=False, 
              precond='gamg', sub_precond="amg",
              eps_a=1e-10, eps_r=1e-10, method="gmres")
L = MUMPSSolver(domain=domain, var=P, conf=conf)


Errors = []
Times = []
x_front = []
c = 1
d_t = np.float64(1e-4)

#
get_triplet_2d_with_contrib  = backend.compile(get_triplet_2d_with_contrib, signature=signature)
get_rhs_glob_2d_with_contrib = backend.compile(get_rhs_glob_2d_with_contrib, signature=signature)
#
explicitscheme_convective_2d  = backend.compile(explicitscheme_convective_2d, signature=signature)
update_new_value = backend.compile(update_new_value, signature=signature)
time_step = backend.compile(time_step, signature=signature)
face_gradient_info_2d = backend.compile(face_gradient_info_2d, signature=signature)

start = MPI.Wtime()

#loop over time
while time < tfinal:

    C.update_halo_value()
    C.update_ghost_value()
    C.interpolate_celltonode()

    I.update_halo_value()
    I.update_ghost_value()
    I.interpolate_celltonode()

    visc.interpolate_celltoface()
    fi.interpolate_celltoface()

    I.compute_cell_gradient()
    
    explicitscheme_convective_2d(I.Flux, I.cell, I.ghost, I.halo, u.face[:]/fi0, v.face/fi0, w.face,
                                 I.gradcellx, I.gradcelly, I.gradcellz, I.gradhalocellx,
                                 I.gradhalocelly, I.gradhalocellz, I.psi, I.psihalo,
                                 cells.center, faces.center, halos.centvol, faces.ghostcenter,
                                 faces.cellid, faces.normal, faces.halofid, faces.name,
                                 domain.innerfaces, domain.halofaces, domain.boundaryfaces,
                                 domain.periodicboundaryfaces, cells.shift, order=order)

    update_new_value(I.cell, I.Flux,  dissip_I, src_I, d_t, cells.volume)

    
    constant = 1
    L.update_ghost_values()
    perm_x.interpolate_celltoface()
    perm_y.interpolate_celltoface()
    face_gradient_info_2d(faces._cellid, faces._nodeid, faces._ghostcenter, faces._name, faces._normal,
                              cells._center, halos._centvol, faces._halofid, nodes._vertex, faces._airDiamond,
                              faces._param1, faces._param2, faces._param3, faces._param4, faces._f_1,
                              faces._f_2, faces._f_3, faces._f_4, cells._shift, dim, perm_x.face, perm_y.face)
    
    get_triplet_2d_with_contrib(domain.faces.cellid, domain.faces.nodeid, domain.cells.faceid, domain.nodes.vertex,
                                domain.faces.halofid, domain.halos.halosext, domain.nodes.oldname,
                                domain.cells.volume, domain.nodes.cellid,
                                domain.cells.center, domain.halos.centvol, domain.nodes.halonid, domain.nodes.periodicid,
                                domain.nodes.ghostcenter, domain.nodes.haloghostcenter, domain.faces.airDiamond,
                                domain.nodes.lambda_x, domain.nodes.lambda_y, domain.nodes.number, domain.nodes.R_x, domain.nodes.R_y,
                                faces.param1, faces.param2, faces.param3, faces.param4, domain.cells.shift,
                                L.localsize, domain.cells.loctoglob, P.BCdirichlet, L._data, L._row, L._col,
                                L.matrixinnerfaces, domain.halofaces, P.dirichletfaces,
                                I.cell, alpha_para, np.ones(nbcells), visc.cell, P.BCneumannNH, faces.dist_ortho)
    
    L.rhs0_glob = np.zeros(L.globalsize)
    get_rhs_glob_2d_with_contrib(domain.faces.cellid, domain.faces.nodeid, domain.nodes.oldname,
                                 domain.cells.volume, domain.nodes.ghostcenter, domain.cells.loctoglob,
                                 domain.faces.param1, domain.faces.param2, domain.faces.param3, domain.faces.param4,
                                 domain.Pbordnode, domain.Pbordface,
                                 L.rhs0, P.BCdirichlet, domain.faces.ghostcenter,
                                 L.matrixinnerfaces, domain.halofaces, P.dirichletfaces, P.neumannNHfaces,
                                 I.cell, I.node, np.ones(nbcells), visc.cell, cst, faces.mesure, faces.normal, faces.dist_ortho)
#
    L()
    P.update_halo_value()
    P.update_ghost_value()
    P.interpolate_celltonode()
    L.compute_Sol_gradient()

    visc.update_ghost_value()
    visc.interpolate_celltoface()

    fi.update_ghost_value()
    fi.interpolate_celltoface()

    #update_variables(alpha.cell, alpha0, np.ones(nbcells), perm0, visc.cell, mu0, A, a, sigma.cell, C.cell, u.cell, v.cell, sigma_u, d_t, fi.cell, fi0 )

    alpha.update_ghost_value()
    sigma.update_ghost_value()
    fi.update_ghost_value()

    #src_C[:] =  - np.sqrt(u.cell[:]**2+v.cell[:]**2) * C.cell[:] * alpha.cell[:] * (1 - sigma.cell[:]/sigma_u)

    u.face[:] = constant * (perm_x.face[:]/visc.face[:]) * P.gradfacex[:]
    v.face[:] = constant * (perm_y.face[:]/visc.face[:]) * P.gradfacey[:]
#
#
#    if test == "debit":
#        update_ghost_values_U(u.ghost , v.ghost, U_n, faces.normal, faces.mesure, P.neumannNHfaces)
#
    u.interpolate_facetocell()
    v.interpolate_facetocell()
#
    ######calculation of the time step
    dt_c = time_step(u.cell[:]/fi.cell[:], v.cell[:]/fi.cell[:], np.zeros(nbcells), cfl, faces.normal, faces.mesure, 
                     cells.volume, cells.faceid,
                     dim, Dxx=np.float64(0.), Dyy=np.float64(0.), Dzz=np.float64(0.))

    d_t = COMM.allreduce(dt_c, MPI.MIN)
    tot = int(tfinal/d_t/1000)+1

    time = time + d_t

    # C.compute_cell_gradient()
    # explicitscheme_convective_2d(C.Flux, C.cell, C.ghost, C.halo, u.face, v.face, w.face,
    #                              C.gradcellx, C.gradcelly, C.gradcellz, C.gradhalocellx,
    #                              C.gradhalocelly, C.gradhalocellz, C.psi, C.psihalo,
    #                              cells.center, faces.center, halos.centvol, faces.ghostcenter,
    #                              faces.cellid, faces.mesure, faces.normal, faces.halofid, faces.name,
    #                              domain.innerfaces, domain.halofaces, domain.boundaryfaces,
    #                              domain.periodicboundaryfaces, cells.shift, order=2)

    # update_new_value(fiC.cell, u.cell, v.cell, P.cell, C.Flux,  dissip_I, src_C, d_t, cells.volume)

    # fiC.update_ghost_value()

    # C.cell[:] = fiC.cell[:]/fi.cell[:]

    ## Parameters (K, mu) updates
    perm_x.cell[:] = perm_x_0.cell[:] * ( (fi.cell[:]/fi0) * ((1-fi.cell[:])/(1-fi0))**(-2) )
    perm_y.cell[:] = perm_y_0.cell[:] * ( (fi.cell[:]/fi0) * ((1-fi.cell[:])/(1-fi0))**(-2) )
    visc.cell[:] = mu0 * (1 - C.cell[:]/A)**(-2)

    Tau = tau_remplissage(I.cell[:])

    if  niter == 1 or (int(time)%1 == 0 and int(time+d_t) != int(time)) :

        if saving_at_node:

            P.update_halo_value()
            P.update_ghost_value()
            P.interpolate_celltonode()

            u.interpolate_celltonode()
            v.interpolate_celltonode()

            I.update_halo_value()
            I.update_ghost_value()
            I.interpolate_celltonode()

            C.update_ghost_value()
            C.interpolate_celltonode()

            sigma.update_ghost_value()
            sigma.interpolate_celltonode()

            visc.interpolate_celltonode()

            alpha.update_ghost_value()
            alpha.interpolate_celltonode()

            perm_x.update_ghost_value()
            perm_x.interpolate_celltonode()

            perm_y.update_ghost_value()
            perm_y.interpolate_celltonode()

            domain.save_on_node_multi(d_t, time, niter, miter, variables=["I", "Iexact", "u","v", "w", "alpha", r"$C * \phi + \sigma$", "C", "sigma", "viscosity", 
                                                                          "porosity", "Perm_x", "Perm_y", "Pression_with_contr"],
                                      values=[I.node, Iexact.node, u.node,v.node,w.node, alpha.node, C.node * fi.node + sigma.node, 
                                              C.node, sigma.node, visc.node, fi.node, perm_x.node, perm_y.node, P.node])

        else:

            if test == "pression":
                1
                compute_Pexact_Pimp(Pexact.cell, Iexact.cell, np.zeros(nbcells), visc.cell, fi, Pin, time, x0, cells.center[:,0])
            elif test == "debit":
                1
                compute_Pexact_Uimp(Pexact.cell, Iexact.cell, np.zeros(nbcells), visc.cell, fi, U_n, time, x0, cells.center[:,0])

            # Errors.append(I.norml2(Iexact.cell, 2))
            # Times.append(time)


            domain.save_on_cell_multi(d_t, time, niter, miter, variables=["I", "Iexact", "u","v", "w", "P", "Pexact"],
                                      values=[I.cell, Iexact.cell, u.cell, v.cell, w.cell, P.cell, Pexact.cell])

        miter += 1
    # if Tau == 1:
    #     break
    niter += 1

#if RANK == 0:
#    L.destroy()

stop = MPI.Wtime()

cputime = COMM.reduce(stop - start, op=MPI.MAX, root=0)
if RANK == 0:

    #print("norm l2", norme_L2(Pexact.cell, P.cell, cells.volume))
    print( "cpu time:", cputime)

    #print(Errors, Times )
    print(x_front, Times )
#os.system("mv results results_"+(str(cfl)+"_"+str(nbcells)))

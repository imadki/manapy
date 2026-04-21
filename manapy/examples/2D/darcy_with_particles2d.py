from mpi4py import MPI
import timeit
from manapy.domain import Domain, Partitioning
from manapy.helpers import get_mesh
import manapy.solvers.advecdiff.fvm_utils_compute as advecdiff_compute
import manapy.solvers.ls.ls_compute as ls_compute
from manapy.solvers.ls import MUMPSSolver, PETScKrylovSolver
from manapy.core.Variable import Variable
import numpy as np
from manapy.backends.compile_fun import compile

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


def face_gradient_info_2d(face_cellid: 'int[:,:]', faces: 'int[:,:]', face_ghostid: 'int[:]', ghost_info_flt: 'float[:, :]', face_name: 'int[:]',
                          face_normal: 'float[:,:]',
                          cell_center: 'float[:,:]', halo_centvol: 'float[:,:]', face_halofid: 'int[:]', nodes: 'float[:,:]',
                          face_airDiamond: 'float[:]', face_param1: 'float[:]', face_param2: 'float[:]', face_param3: 'float[:]',
                          face_param4: 'float[:]',
                          face_f1: 'float[:,:]', face_f2: 'float[:,:]', face_f3: 'float[:,:]', face_f4: 'float[:,:]',
                          cell_shift: 'float[:,:]',
                          dim: 'int', Kx: 'float[:]', Ky: 'float[:]'):
  nbface = len(face_cellid)

  xy_1 = np.zeros(dim)
  xy_2 = np.zeros(dim)
  v_1 = np.zeros(dim)
  v_2 = np.zeros(dim)

  for i in range(nbface):

    c_left = face_cellid[i][0]
    c_right = face_cellid[i][1]

    i_1 = faces[i][0]
    i_2 = faces[i][1]

    xy_1[:] = nodes[i_1][0:dim]
    xy_2[:] = nodes[i_2][0:dim]

    v_1[:] = cell_center[c_left][0:dim]

    if face_name[i] == 0:
      v_2[:] = cell_center[c_right][0:dim]
    elif face_name[i] == 11 or face_name[i] == 22:
      v_2[0] = cell_center[c_right][0] + cell_shift[c_right][0]
      v_2[1] = cell_center[c_right][1]
    elif face_name[i] == 33 or face_name[i] == 44:
      v_2[0] = cell_center[c_right][0]
      v_2[1] = cell_center[c_right][1] + cell_shift[c_right][1]
    elif face_name[i] == 10:
      v_2[:] = halo_centvol[face_halofid[i]][0:dim]
    else:
      ghost_id = face_ghostid[i]
      v_2[:] = ghost_info_flt[ghost_id][0:dim]

    face_f1[i][:] = v_1[:] - xy_1[:]
    face_f2[i][:] = xy_2[:] - v_1[:]
    face_f3[i][:] = v_2[:] - xy_2[:]
    face_f4[i][:] = xy_1[:] - v_2[:]

    n1 = face_normal[i][0] * Kx[i]
    n2 = face_normal[i][1] * Ky[i]

    face_airDiamond[i] = 0.5 * ((xy_2[0] - xy_1[0]) * (v_2[1] - v_1[1]) + (v_1[0] - v_2[0]) * (xy_2[1] - xy_1[1]))

    face_param1[i] = 1. / (2. * face_airDiamond[i]) * ((face_f1[i][1] + face_f2[i][1]) * n1 - (face_f1[i][0] + face_f2[i][0]) * n2)
    face_param2[i] = 1. / (2. * face_airDiamond[i]) * ((face_f2[i][1] + face_f3[i][1]) * n1 - (face_f2[i][0] + face_f3[i][0]) * n2)
    face_param3[i] = 1. / (2. * face_airDiamond[i]) * ((face_f3[i][1] + face_f4[i][1]) * n1 - (face_f3[i][0] + face_f4[i][0]) * n2)
    face_param4[i] = 1. / (2. * face_airDiamond[i]) * ((face_f4[i][1] + face_f1[i][1]) * n1 - (face_f4[i][0] + face_f1[i][0]) * n2)


def compute_Pexact_Uimp(Pexact, Iexact, perm, visc, fi, Uin, t, x0, x):
  xf = x0 + Uin * t

  for i in range(len(x)):
    if x[i] > xf:
      Pexact[i] = 0.
      Iexact[i] = 0.
    else:
      Pexact[i] = (visc[i] * fi / perm[i]) * Uin * (xf - x[i])
      Iexact[i] = 1.


def compute_Pexact_Pimp(Pexact, Iexact, perm0, visc0, fi, Pin, t, x0, x):
  xff = np.zeros(len(x))
  xf = np.zeros(len(x))

  xff = 2 * (perm0 / fi * visc0) * Pin * t + x0 ** 2
  xf = np.sqrt(xff)

  for i in range(len(x)):
    if x[i] > xf:
      Pexact[i] = 0.
      Iexact[i] = 0.
    else:
      Pexact[i] = Pin * (1. - x[i] / xf[i])
      Iexact[i] = 1.


def update_ghost_values_U(ughost, vghost, U_n, normal_faces, mesure, neumannNHfaces):
  for i in neumannNHfaces:
    normal = -normal_faces[i][0:2] / mesure[i]
    U_n_ = U_n * normal
    x = np.array((1., 0))
    y = np.array((0, 1.))
    # norm = np.sqrt()
    # teta = math.acos(np.dot(x, U_n*normal)/U_n)

    ughost[i] = np.dot(U_n_, x)
    vghost[i] = np.dot(U_n_, y)


def tau_remplissage(I):
  Tau_remp = sum(I > 0.9) / len(I)
  return Tau_remp


get_triplet_2d_with_contrib = ls_compute.get_triplet_2d_with_contrib
get_rhs_glob_2d_with_contrib = ls_compute.get_rhs_glob_2d_with_contrib
#
explicitscheme_convective_2d = advecdiff_compute.explicitscheme_convective_2d
update_new_value = advecdiff_compute.update_new_value
time_step = advecdiff_compute.time_step
face_gradient_info_2d = compile(face_gradient_info_2d)

# Simulation parametres
##############################################################################
##############################################################################
test_para = 'test_1'

C0 = 0.
alpha0 = 1.
sigma_u = 0.85
a = 1.
A = 0.68
Pin = 3e5
mu0 = 0.109
fi0 = 0.45
U_n = 1e-1
tfinal = 10
fi = 0.81
perm0 = 6.83e-9
test = "pression"
filename = "big/carre.msh"

if test_para == "test_2":
  C0 = 0.
  alpha0 = 1.
  sigma_u = 0.85
  a = 1.
  A = 0.68
  Pin = 3e5
  mu0 = 0.109
  fi0 = 0.45
  U_n = 1e-1
  tfinal = 10

  test = "pression"
  filename = "big/carre.msh"
  # filename = "TMesh.msh"

else: # Default test
  fi = 0.81
  U_n = 3.e-3
  Pin = 1e5
  perm0 = 6.83e-9
  mu0 = 0.3
  tfinal = 122

  test = "pression"
  # filename = "Geom_exp3.msh"
  filename = "big/carre.msh"





start = timeit.default_timer()

dim, mesh_path, mesh_name = get_mesh(filename)
domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells


scheme = "Diamond"
# scheme = "FV4"
##################################################################################
##################################################################################

boundariesI = {"in": "dirichlet",
               "out": "neumann",
               "upper": "neumann",
               "bottom": "neumann"
               }
valuesI = {"in": 1}

I = Variable(domain=domain, BC=boundariesI, values_dict=valuesI)
I.add_term("Flux")
I.update_ghost_value()

# Concentration
boundariesC = {"in": "dirichlet",
               "out": "neumann",
               "upper": "neumann",
               "bottom": "neumann"
               }
valuesC = {"in": C0}
C = Variable(domain=domain, BC=boundariesC, values_dict=valuesC)
C.add_term("Flux")
C.update_ghost_value()

fiC = Variable(domain=domain)
fiC.add_term("Flux")

perm_x_0 = Variable(domain=domain)
perm_y_0 = Variable(domain=domain)

perm_x = Variable(domain=domain)
perm_y = Variable(domain=domain)

visc = Variable(domain=domain)
fi = Variable(domain=domain)
sigma = Variable(domain=domain)
alpha = Variable(domain=domain)

## initialization of viscosity, porosity and filtration coieficient
visc.cell[:] = mu0
fi.cell[:] = fi0
alpha.cell[:] = alpha0
visc.update_ghost_value()
fi.update_ghost_value()
alpha.update_ghost_value()

## initialization of the permeability
for i in range(nbcells):
  if cells.center[i][1] >= 0.0014:
    perm_x_0.cell[i] = 2.e-11
    perm_y_0.cell[i] = 2.e-11

  else:
    perm_x_0.cell[i] = 2.e-11
    perm_y_0.cell[i] = 2.e-11

# Perm changes over time
perm_x.cell[:] = perm_x_0.cell[:]
perm_y.cell[:] = perm_y_0.cell[:]

perm_x.update_ghost_value()
perm_y.update_ghost_value()

## injection mode
if test == "pression":
  boundariesP = {"in": "dirichlet",
                 "out": "dirichlet",
                 "upper": "neumann",
                 "bottom": "neumann"
                 }
  valuesP = {"in": Pin, "out": 0.}
  boundariesU = {"in": "neumann",
                 "out": "neumann",
                 "upper": "nonslip",
                 "bottom": "nonslip"}

  u = Variable(domain=domain, BC=boundariesU)
  v = Variable(domain=domain, BC=boundariesU)


elif test == "debit":
  boundariesP = {"in": "neumannNH",
                 "out": "dirichlet",
                 "upper": "neumann",
                 "bottom": "neumann"
                 }
  cst = np.float64((mu0 / perm0) * U_n)
  valuesP = {"in": cst, "out": 0.}

  boundariesU = {"in": "dirichlet",
                 "out": "neumann",
                 "upper": "neumann",
                 "bottom": "neumann"
                 }
  valuesU = {"in": U_n}
  u = Variable(domain=domain)  # , BC=boundariesU, values=valuesU)

v = Variable(domain=domain)
w = Variable(domain=domain)

P = Variable(domain=domain, BC=boundariesP, values_dict=valuesP)
Pexact = Variable(domain=domain)
Iexact = Variable(domain=domain)

# TODO tfinal
if RANK == 0: print("Start Computation")

for i in domain.infaces:
  K = faces.cellid[i][0]
  I.cell[K] = 1.

x0 = 0.
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

###Linear sys confi###
# If you want the default options please do conf = Struct()
# reuse_mtx: matrix does not change during the while loop
# scheme: diamond (fv4 not tested!!!)
# verbose: printing the mumps/petsc output
# L = MUMPSSolver(domain=domain, var=P, reuse_mtx=False, scheme='diamond')

L = PETScKrylovSolver(domain=domain, var=P, reuse_mtx=False, scheme='diamond',
              precond='gamg', sub_precond="amg",  # with_mtx=False,
              eps_a=1e-10, eps_r=1e-10, method="gmres")


Errors = []
Times = []
x_front = []
c = 1
d_t = np.float64(1e-4)



start = MPI.Wtime()

# loop over time
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

  explicitscheme_convective_2d(I.Flux, I.cell, I.ghost, I.halo, u.face[:] / fi0, v.face / fi0, w.face,
                               I.gradcellx, I.gradcelly, I.gradcellz, I.gradhalocellx,
                               I.gradhalocelly, I.gradhalocellz, I.psi, I.psihalo,
                               cells.center, faces.center, halos.centvol,
                               faces.cellid, faces.normal, faces.halofid, faces.name,
                               domain.innerfaces, domain.halofaces, domain.boundaryfaces,
                               domain.periodicboundaryfaces, cells.shift, order, domain.faces.ghost_id)

  update_new_value(I.cell, I.Flux, dissip_I, src_I, d_t, cells.volume)

  constant = 1
  L.update_ghost_values()
  perm_x.interpolate_celltoface()
  perm_y.interpolate_celltoface()

  face_gradient_info_2d(faces.cellid, faces.nodeid, faces.ghost_id, domain.ghost.info_flt, faces.name, faces.normal,
                        cells.center, halos.centvol, faces.halofid, nodes.vertex, faces.airDiamond,
                        faces.param1, faces.param2, faces.param3, faces.param4, faces.f_1,
                        faces.f_2, faces.f_3, faces.f_4, cells.shift, dim, perm_x.face, perm_y.face)

  get_triplet_2d_with_contrib(domain.faces.cellid, domain.faces.nodeid, domain.cells.faceid, domain.nodes.vertex,
                              domain.halos.halosext, domain.nodes.oldname,
                              domain.cells.volume, domain.nodes.cellid,
                              domain.cells.center, domain.halos.centvol, domain.nodes.halonid, domain.ghost.info_flt, domain.ghost.info_int,
                              domain.ghost.ext_info_flt, domain.ghost.ext_info_int, domain.nodes.ghostid, domain.nodes.haloghostid,
                              domain.nodes.lambda_x, domain.nodes.lambda_y, domain.nodes.number, domain.nodes.R_x,
                              domain.nodes.R_y,
                              faces.param1, faces.param2, faces.param3, faces.param4, domain.cells.loctoglob, P.BCdirichlet, L._data, L._row, L._col,
                              L.matrixinnerfaces, P.dirichletfaces,
                              I.cell, alpha_para, np.ones(nbcells), visc.cell)

  L.rhs0_glob = np.zeros(L.globalsize)

  get_rhs_glob_2d_with_contrib(domain.faces.cellid, domain.faces.nodeid, domain.nodes.oldname,
                               domain.cells.volume, domain.nodes.ghostid, domain.cells.loctoglob,
                               domain.faces.param1, domain.faces.param2, domain.faces.param3, domain.faces.param4,
                               domain.Pbordnode, domain.Pbordface,
                               L.rhs0, P.BCdirichlet,
                               L.matrixinnerfaces, domain.halofaces, P.dirichletfaces, P.neumannNHfaces,
                               I.cell, np.ones(nbcells), visc.cell, cst, faces.normal)
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

  # update_variables(alpha.cell, alpha0, np.ones(nbcells), perm0, visc.cell, mu0, A, a, sigma.cell, C.cell, u.cell, v.cell, sigma_u, d_t, fi.cell, fi0 )

  alpha.update_ghost_value()
  sigma.update_ghost_value()
  fi.update_ghost_value()

  # src_C[:] =  - np.sqrt(u.cell[:]**2+v.cell[:]**2) * C.cell[:] * alpha.cell[:] * (1 - sigma.cell[:]/sigma_u)

  u.face[:] = constant * (perm_x.face[:] / visc.face[:]) * P.gradfacex[:]
  v.face[:] = constant * (perm_y.face[:] / visc.face[:]) * P.gradfacey[:]
  #
  #
  #    if test == "debit":
  #        update_ghost_values_U(u.ghost , v.ghost, U_n, faces.normal, faces.mesure, P.neumannNHfaces)
  #
  u.interpolate_facetocell()
  v.interpolate_facetocell()
  #
  ######calculation of the time step
  dt_c = time_step(u.cell[:] / fi.cell[:], v.cell[:] / fi.cell[:], np.zeros(nbcells), cfl, faces.normal, faces.mesure,
                   cells.volume, cells.faceid,
                   dim, Dxx=np.float64(0.), Dyy=np.float64(0.), Dzz=np.float64(0.))

  d_t = COMM.allreduce(dt_c, MPI.MIN)
  tot = int(tfinal / d_t / 1000) + 1

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
  perm_x.cell[:] = perm_x_0.cell[:] * ((fi.cell[:] / fi0) * ((1 - fi.cell[:]) / (1 - fi0)) ** (-2))
  perm_y.cell[:] = perm_y_0.cell[:] * ((fi.cell[:] / fi0) * ((1 - fi.cell[:]) / (1 - fi0)) ** (-2))
  visc.cell[:] = mu0 * (1 - C.cell[:] / A) ** (-2)

  Tau = tau_remplissage(I.cell[:])

  if niter == 1 or (int(time) % 1 == 0 and int(time + d_t) != int(time)):

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

      domain.save_on_node_multi(d_t, time, niter, miter,
                                variables=["I", "Iexact", "u", "v", "w", "alpha", r"$C * \phi + \sigma$", "C", "sigma",
                                           "viscosity",
                                           "porosity", "Perm_x", "Perm_y", "Pression_with_contr"],
                                values=[I.node, Iexact.node, u.node, v.node, w.node, alpha.node,
                                        C.node * fi.node + sigma.node,
                                        C.node, sigma.node, visc.node, fi.node, perm_x.node, perm_y.node, P.node])

    else:

      if test == "pression":
        compute_Pexact_Pimp(Pexact.cell, Iexact.cell, np.zeros(nbcells), visc.cell, fi, Pin, time, x0,
                            cells.center[:, 0])
      elif test == "debit":
        compute_Pexact_Uimp(Pexact.cell, Iexact.cell, np.zeros(nbcells), visc.cell, fi, U_n, time, x0,
                            cells.center[:, 0])

      # Errors.append(I.norml2(Iexact.cell, 2))
      # Times.append(time)

      domain.save_on_cell_multi(d_t, time, niter, miter, variables=["I", "Iexact", "u", "v", "w", "P", "Pexact"],
                                values=[I.cell, Iexact.cell, u.cell, v.cell, w.cell, P.cell, Pexact.cell])

    miter += 1
  # if Tau == 1:
  #     break
  niter += 1

# if RANK == 0:
#    L.destroy()

stop = MPI.Wtime()

cputime = COMM.reduce(stop - start, op=MPI.MAX, root=0)
if RANK == 0:
  # print("norm l2", norme_L2(Pexact.cell, P.cell, cells.volume))
  print("cpu time:", cputime)

  # print(Errors, Times )
  print(x_front, Times)
# os.system("mv results results_"+(str(cfl)+"_"+str(nbcells)))

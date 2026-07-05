from mpi4py import MPI
import os
import timeit
from manapy.domain import Domain, Partitioning
from manapy.solvers.advec.tools_utils_compute import initialisation_gaussian_2d
from manapy.solvers.advec.system import AdvectionSolver
from manapy.core.Variable import Variable
# from manapy.backends.gpu import GPUBackend


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
start = timeit.default_timer()


try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes', 'geo')

filename = "carre.msh"
dim = 2
mesh_path = os.path.join(MESH_DIR, filename)

# gpu = GPUBackend(float_precision="float64", int_precision="int32", cache=False)
# gpu.init_stream()
# gpu.set_config(free=True)

# backend=gpu

domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True, 
                              # backend=backend
                              )
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells

end = timeit.default_timer()

tt = COMM.reduce(end - start, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to create the domain", tt)

# TODO tfinal
if RANK == 0: print("Start Computation ...")
time = 0
tfinal = .25
miter = 0
niter = 1
Pinit = 2.
saving_at_node = 0

boundaries = {"in": "dirichlet",
              "out": "dirichlet",
              "upper": "neumann",
              "bottom": "neumann"
              }
values = {"in": Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain, limiter='vanalbada')
u = Variable(domain=domain)
v = Variable(domain=domain)
P = Variable(domain=domain)

# Call the transport solver
S = AdvectionSolver(ne, vel=(u, v), order=2, cfl=0.8)

####Initialisation
initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, Pinit)
f = lambda x, y, z: Pinit * (1. - x)
COMM.Barrier()

ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")


# loop over time
while time < tfinal:

  # # TODO -1
  u.face[:] = 2.
  v.face[:] = 0.

  # if backend == gpu:
  #       gpu.assign(u.face, 2.0)
  #       gpu.assign(v.face, 0.0)

  u.interpolate_facetocell()
  v.interpolate_facetocell()

  d_t = S.stepper()
  tot = int(tfinal / d_t / 50) + 1

  time = time + d_t

  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter % tot == 0:
    if saving_at_node:
      # save vtk files for the solution
      ne.update_halo_value()
      ne.update_ghost_value()
      ne.interpolate_celltonode()

      # save vtk files for the solution
      u.update_halo_value()
      u.update_ghost_value()
      u.interpolate_celltonode()

      # save vtk files for the solution
      v.update_halo_value()
      v.update_ghost_value()
      v.interpolate_celltonode()

      domain.save_on_node_multi(["ne", "u", "v", "P"],
                                [ne.node, u.node, v.node, P.node], d_t, time, niter, miter)
    else:
      domain.save_on_cell_multi(["ne", "u", "v", "P"], [ne.cell, u.cell, v.cell, P.cell], d_t, time, niter, miter)
    miter += 1

  niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)


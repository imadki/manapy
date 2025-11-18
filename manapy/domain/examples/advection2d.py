from mpi4py import MPI
import timeit
from manapy.domain import Domain
from manapy.tests.meshes import get_mesh
from manapy.solvers.advec.tools_utils import initialisation_gaussian_2d
from manapy.solvers.advec import AdvectionSolver
from manapy.ast import Variable
from manapy.base.base import Struct
import os

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
start = timeit.default_timer()


dim, mesh_path, mesh_name = get_mesh(3)
domain = Domain.create_domain(mesh_path, dim, recreate=True)
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
saving_at_node = 1

boundaries = {"in": "dirichlet",
              "out": "dirichlet",
              "upper": "neumann",
              "bottom": "neumann"
              }
values = {"in": Pinit,
          "out": 0.,
          }

ne = Variable(domain=domain)
u = Variable(domain=domain)
v = Variable(domain=domain)
P = Variable(domain=domain)

# Call the transport solver
conf = Struct(order=1, cfl=0.8)
S = AdvectionSolver(ne, vel=(u, v), conf=conf)

####Initialisation
initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, Pinit)
f = lambda x, y, z: Pinit * (1. - x)

ne.update_halo_value()
for i in range(nbfaces):
  if faces.name[i] == 10 and RANK == 7:
    print("=<", ne.halo[faces.halofid[i]], faces.halofid[i])




ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")

# loop over time
while time < tfinal:

  # TODO -1
  u.face[:] = 2.
  v.face[:] = 0.

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

      domain.save_on_node_multi(d_t, time, niter, miter, variables=["ne", "u", "v", "P"],
                                values=[ne.node, u.node, v.node, P.node], file_format="vtu")
    else:
      domain.save_on_cell_multi(d_t, time, niter, miter, variables=["ne", "u", "v", "P"],
                                values=[ne.cell, u.cell, v.cell, P.cell], file_format="vtu")
    miter += 1

  niter += 1

te = MPI.Wtime()

tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
if RANK == 0:
  print("Time to do calculation", tt)


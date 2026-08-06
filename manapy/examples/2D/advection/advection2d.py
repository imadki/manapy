from mpi4py import MPI
import os
import timeit
from manapy.domain import Domain, Partitioning
from manapy.solvers.advec.system import AdvectionSolver
from manapy.core.Variable import Variable
from manapy.backends.ManapyArray import Device, ManapyArray
from manapy.backends.config import ManapyConfig


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

filename = "uns_square.msh"
dim = 2
mesh_path = os.path.join(MESH_DIR, filename)

# The config decides the precision pair and the device, and so which compiled
# kernels every Variable / Boundary / solver below binds -- there is nothing
# else to switch: the same code runs on both. Overridable from the environment
# so a GPU run needs no edit:
#     MANAPY_DEVICE=cuda python3 advection2d.py
config = ManapyConfig(
  float_precision=os.environ.get("MANAPY_FLOAT", "float64"),
  int_precision=os.environ.get("MANAPY_INT", "int64"),
  device=os.environ.get("MANAPY_DEVICE", "cpu"),
)

domain = Domain.create_domain(mesh_path, dim, config,
                              partitioning_method=Partitioning.Par_Nodal,
                              recreate=True)
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

S = AdvectionSolver(ne, vel=(u, v), order=2, cfl=0.8)

S.compute.initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, Pinit)
COMM.Barrier()

# Write-only accessor for the device this run is on. `cpu_w` / `gpu_w` hand out
# the buffer with no transfer and mark the other side stale, which is what the
# prescribed velocity below wants: it overwrites every face, and the kernels
# that read it next sync it in themselves. Taking it from the config keeps the
# face field on the device the solver runs on -- hardcoding `cpu_w` would push
# a host->device copy of it into every iteration of a CUDA run.
face_w = ManapyArray.gpu_w if config.device == Device.CUDA else ManapyArray.cpu_w

ts = MPI.Wtime()

if RANK == 0: print("Start While loop ...")


while time < tfinal:

  face_w(u.face)[:] = 2.
  face_w(v.face)[:] = 0.

  u.interpolate_facetocell()
  v.interpolate_facetocell()

  d_t = S.stepper()
  tot = int(tfinal / d_t / 50) + 1

  time = time + d_t

  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter % tot == 0:
    if saving_at_node:
      ne.update_halo_value()
      ne.update_ghost_value()
      ne.interpolate_celltonode()

      u.update_halo_value()
      u.update_ghost_value()
      u.interpolate_celltonode()

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


"""
Advection 2D sur GPU.

Variante GPU de advection2d.py : meme maillage, meme solveur, memes kernels
sources (annotes en chaines) ; seul le backend change. Le portage est valide
numeriquement contre le CPU (ecart ~1e-16) dans les tests de developpement.

Mono-rang pour l'instant (le multi-rang necessite des halos GPU). Lancer :

    python3 advection2d_gpu.py

Variable d'environnement utile : MANAPY_GPU_VERBOSE=1 pour tracer la compilation
des kernels et le GPU selectionne.
"""
from mpi4py import MPI
import os
import numpy as np

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.advec.system import AdvectionSolver
from manapy.solvers.advec.tools_utils_compute import initialisation_gaussian_2d

from manapy.backends.gpu import GPUBackend

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  BASE_DIR = os.path.join(BASE_DIR, '..', '..', '..')
  MESH_DIR = os.path.join(BASE_DIR, 'meshes', 'geo')

filename = "carre.msh"
dim = 2
mesh_path = os.path.join(MESH_DIR, filename)

gpu = GPUBackend(cache=False)        # precision = types globaux (float64/int32)
gpu.init_stream()
gpu.set_config(free=True)            # grille dimensionnee sur la taille du probleme

domain = Domain.create_domain(mesh_path, dim, Partitioning.Par_Nodal, recreate=True,
                              backend=gpu)
cells = domain.cells

# --- variables + solveur (identiques au cas CPU) ---
ne = Variable(domain=domain)
u = Variable(domain=domain)
v = Variable(domain=domain)
P = Variable(domain=domain)
S = AdvectionSolver(ne, vel=(u, v), order=2, cfl=0.8)

# --- initialisation sur l'hote (avant bascule GPU) ---
Pinit = 2.0
initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, Pinit)

if RANK == 0:
  print("Start GPU computation ...")

time = 0.0
tfinal = 0.25
niter = 1
miter = 0
saving_at_node = 0
ts = MPI.Wtime()

while time < tfinal:
  gpu.assign(u.face, 2.0)
  gpu.assign(v.face, 0.0)

  u.interpolate_facetocell()
  v.interpolate_facetocell()

  d_t = S.stepper()
  tot = int(tfinal / d_t / 50) + 1
  time += d_t

  S.compute_fluxes()
  S.compute_new_val()

  if niter == 1 or niter % tot == 0:
    if saving_at_node:
      ne.update_halo_value(); ne.update_ghost_value(); ne.interpolate_celltonode()
      u.update_halo_value(); u.update_ghost_value(); u.interpolate_celltonode()
      v.update_halo_value(); v.update_ghost_value(); v.interpolate_celltonode()
      domain.save_on_node_multi(["ne", "u", "v", "P"], [ne.node, u.node, v.node, P.node], d_t, time, niter, miter)
    else:
      domain.save_on_cell_multi(["ne", "u", "v", "P"], [ne.cell, u.cell, v.cell, P.cell], d_t, time, niter, miter)
    miter += 1

  niter += 1

te = MPI.Wtime()
tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)

# rapatriement du resultat pour inspection / sauvegarde
ne_host = np.asarray(ne.cell.to_host())
if RANK == 0:
  print(f"iterations: {niter}")
  print(f"Time to do calculation (GPU): {tt}")
  print(f"ne min/max = {ne_host.min():.6f} / {ne_host.max():.6f}")

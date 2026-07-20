#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbulent SWMHD (k-epsilon RANS) — magnetized shear layer.

A horizontal shear layer u~(y)=U0 tanh((y-0.5)/delta) over a flat bed with a weak
magnetic field seeds turbulence: the mean shear feeds the kinetic production
P^u_k = nu_t[2u_x^2+2v_y^2+(u_y+v_x)^2], the turbulent viscosity nu_t=Cmu k_c^2/eps_c
grows near y=0.5 and diffuses the layer, while the k-epsilon transport carries and
dissipates the turbulent energy. Writes VTK for ParaView (velocity, field, k_c, k_m,
nu_t, mu_t).

Run:  python turbulent_shear_swmhd.py   (or  mpirun -n 4 python turbulent_shear_swmhd.py)
"""
import os
import numpy as np
from mpi4py import MPI
from manapy.api.meshgen import rectangle
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.swmhd.turbulence import TurbulentSWMHDSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# A structured QUAD (orthogonal) mesh: the turbulent stress/diffusion operators
# are then machine-exact. On non-orthogonal triangles the FV face gradient is only
# ~1st order (manapy's known non-orthogonality issue, fixable with an n_nonorth
# deferred correction as in the incompressible solver), so the diffusive terms lose
# accuracy there -- use quads / orthogonal meshes for quantitative turbulence.
BASE = os.path.dirname(os.path.realpath(__file__))
MESH = os.path.join(BASE, 'shear_quad.msh')
if RANK == 0:
  rectangle(bounds=((0, 1), (0, 1)), n=100, cell_type="quad",
            transfinite=True, recombine=True, filename=MESH)
COMM.Barrier()
domain = Domain.create_domain(MESH, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
c = np.asarray(cells.center); x = c[:, 0]; y = c[:, 1]

bc = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}
mkb = lambda: Variable(domain=domain, BC=bc)
h = Variable(domain=domain)
hu, hv, hB1, hB2 = mkb(), mkb(), mkb(), mkb()
PSI = Variable(domain=domain); Z = Variable(domain=domain)
kc, km, epsc, epsm = mkb(), mkb(), mkb(), mkb()

U0, delta = 0.5, 0.05
h.cell[:] = 1.0
hu.cell[:] = U0 * np.tanh((y - 0.5) / delta)
hv.cell[:] = 0.0
hB1.cell[:] = 0.1
hB2.cell[:] = 0.05
kc.cell[:] = 1e-3; km.cell[:] = 1e-4          # seed turbulent energies
epsc.cell[:] = 1e-3; epsm.cell[:] = 1e-4

S = TurbulentSWMHDSolver(h, (hu, hv), (hB1, hB2), kc, km, epsc, epsm,
                         PSI=PSI, Z=Z, nu=1e-3, mu=1e-3, cfl=0.5, grav=1.0, GLM=10)

tfinal = 0.3
nout = 30
time = 0.0
niter = 0
miter = 0

# derived output fields
u_o, v_o, kc_o, km_o, nut_o, mut_o = (Variable(domain=domain) for _ in range(6))


def save(dt):
  global miter
  hh = np.asarray(h.cell)
  u_o.cell[:] = np.asarray(hu.cell) / hh
  v_o.cell[:] = np.asarray(hv.cell) / hh
  kc_o.cell[:] = np.asarray(kc.cell) / hh
  km_o.cell[:] = np.asarray(km.cell) / hh
  nut_o.cell[:] = np.asarray(S.nu_t.cell)
  mut_o.cell[:] = np.asarray(S.mu_t.cell)
  names = ["h", "u", "v", "kc", "km", "nu_t", "mu_t"]
  fields = [h, u_o, v_o, kc_o, km_o, nut_o, mut_o]
  for f in fields:
    f.update_halo_value(); f.update_ghost_value(); f.interpolate_celltonode()
  domain.save_on_node_multi(names, [f.node for f in fields], dt, time, niter, miter)
  miter += 1


if RANK == 0:
  print("Turbulent SWMHD (k-epsilon) shear layer — starting")
save(0.0)
while time < tfinal:
  dt = S.step()
  time += dt
  niter += 1
  if niter % max(1, int(tfinal / dt / nout)) == 0:
    save(dt)
    if RANK == 0:
      kmax = np.asarray(kc.cell).max()
      print("  it=%d  t=%.4f  dt=%.2e  max k_c=%.3e  max nu_t=%.3e"
            % (niter, time, dt, kmax, np.asarray(S.nu_t.cell).max()))
save(dt)
if RANK == 0:
  print("done: %d steps, VTK in ./vtk_results/" % niter)

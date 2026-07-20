#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Shallow-Water MHD: magneto-Rossby wave on a beta-plane (init choix = 42).

Purpose
-------
Measure the dispersion relation omega(kx) of the (magneto-)Rossby wave. A single
Fourier mode (KX, KY) is seeded (choix=42) and its complex modal amplitude

    A(t) = < hv , exp(i (KX x + KY y)) >_volume   (a global volume-weighted sum)

is recorded every step. A(t) rotates as exp(-i omega t), so the companion script
`plot_dispersion_swmhd.py` extracts omega(KX) by FFT and overlays the analytic
relations. Using the complex mode (not a real probe) also recovers the SIGN of
omega -> the wave propagates WESTWARD (omega < 0 for KX > 0), the classic Rossby
signature.

Physics
-------
Background: uniform depth H0=1, uniform zonal (toroidal) field B0 x-hat, no mean
flow, beta-plane Coriolis  f = F0 + BETA*(y - Y0). The Coriolis term is the new
`f0/beta/y0` solver option; with B0=0 this is a pure Rossby wave, with B0>0 it is
a magneto-Rossby wave (the branch splits).

Notes
-----
* Needs PERIODIC boundaries for a clean single-mode measurement, so KX, KY must
  be integer multiples of 2*pi on a unit-square [0,1]^2 mesh. Point MESH at a
  periodic square (meshgen transfinite/recombine) for best results; with a
  non-periodic mesh the low-frequency Rossby peak is still visible but noisier.
* Forward-Euler in time (as the base solver): keep CFL modest and integrate a few
  wave periods. The Rossby wave is SLOW, so TFINAL must be large enough.

Run (serial recommended for the diagnostic):
    KX=6.2831853 KY=0 B0=0.0 python swmhd_rossby2d.py    # pure Rossby
    KX=6.2831853 KY=0 B0=0.1 python swmhd_rossby2d.py    # magneto-Rossby
or in parallel:
    mpirun -n 4 python swmhd_rossby2d.py
"""
from mpi4py import MPI
import numpy as np
import os
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.swmhd.system import ShallowWaterMHDSolver
from manapy.solvers.swmhd.tools_utils_compute import initialisation_SWMHD

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# ------------------------------------------------------------------ parameters
KX     = float(os.environ.get('KX', 2.0 * np.pi))   # perturbation wavenumber x
KY     = float(os.environ.get('KY', 0.0))           # perturbation wavenumber y
B0     = float(os.environ.get('B0', 0.1))           # background zonal field (V_A)
EPS    = float(os.environ.get('EPS', 1e-3))         # perturbation amplitude
F0     = float(os.environ.get('F0', 1.0))           # reference Coriolis
BETA   = float(os.environ.get('BETA', 10.0))        # Rossby beta
Y0     = float(os.environ.get('Y0', 0.5))           # reference latitude
TFINAL = float(os.environ.get('TFINAL', 60.0))
OUT    = os.environ.get('OUT', f'rossby_kx{KX:.4f}_B0{B0:.3f}.txt')

# -------------------------------------------------------------------- domain
try:
  MESH_DIR = os.environ['MESH_DIR']
except KeyError:
  BASE_DIR = os.path.dirname(os.path.realpath(__file__))
  MESH_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'meshes', 'geo')
filename = os.environ.get('MESH', 'uns_square.msh')   # ideally a PERIODIC unit square
mesh_path = os.path.join(MESH_DIR, filename)

domain = Domain.create_domain(mesh_path, 2, Partitioning.Par_Nodal, recreate=True)
cells = domain.cells
nb = domain.nbcells

# periodic on all four sides for a clean single-mode dispersion measurement.
# Override with BCTYPE=neumann for a non-periodic mesh / quick smoke test.
BCTYPE = os.environ.get('BCTYPE', 'periodic')
boundaries = {"in": BCTYPE, "out": BCTYPE, "upper": BCTYPE, "bottom": BCTYPE}

h   = Variable(domain=domain)
hu  = Variable(domain=domain, BC=boundaries)
hv  = Variable(domain=domain, BC=boundaries)
hB1 = Variable(domain=domain, BC=boundaries)
hB2 = Variable(domain=domain, BC=boundaries)
PSI = Variable(domain=domain)
Z   = Variable(domain=domain)

# choix=42: k1=KX, k2=KY, eps=EPS amplitude, tol=B0 background zonal field
initialisation_SWMHD(h.cell, hu.cell, hv.cell, hB1.cell, hB2.cell, PSI.cell, Z.cell,
                     cells.center, 42, KX, KY, EPS, B0)

# Geostrophically balanced QG-Rossby eigenmode: psi = EPS*cos(KX x + KY y),
#   u = -d psi/dy = EPS*KY*sin ,  v = d psi/dx = -EPS*KX*sin ,  h' = (F0/g) psi.
# Balancing h against the velocity kills the fast gravity/Poincare waves, so the
# modal amplitude oscillates cleanly at the slow Rossby frequency (choix=42 alone
# left h uniform -> excited gravity waves -> a smeared spectrum).
GRAV = 1.0
H0 = float(os.environ.get('H0', 10.0))   # mean depth; large H0 -> valid beta-plane
_xc = cells.center[:nb, 0]
_yc = cells.center[:nb, 1]
_ph = KX * _xc + KY * _yc
_u = EPS * KY * np.sin(_ph)
_v = -EPS * KX * np.sin(_ph)
_hp = H0 + (F0 / GRAV) * EPS * np.cos(_ph)
h.cell[:nb] = _hp
hu.cell[:nb] = _hp * _u
hv.cell[:nb] = _hp * _v
hB1.cell[:nb] = _hp * B0
hB2.cell[:nb] = 0.0

S = ShallowWaterMHDSolver(h=h, hvel=(hu, hv), hB=(hB1, hB2), PSI=PSI, Z=Z,
                          order=1, cfl=0.8, grav=1.0, GLM=10,
                          f0=F0, beta=BETA, y0=Y0)

# ------------------------------------- modal projection weights (owned cells)
xc = cells.center[:nb, 0]
yc = cells.center[:nb, 1]
phase = KX * xc + KY * yc
wcos = cells.volume[:nb] * np.cos(phase)
wsin = cells.volume[:nb] * np.sin(phase)


def modal_amplitude():
  ac = COMM.allreduce(float(np.dot(hv.cell[:nb], wcos)), MPI.SUM)
  as_ = COMM.allreduce(float(np.dot(hv.cell[:nb], wsin)), MPI.SUM)
  return ac, as_


# --------------------------------------------------------------- time loop
if RANK == 0:
  print(f"magneto-Rossby: KX={KX:.4f} KY={KY:.4f} B0={B0} F0={F0} BETA={BETA} "
        f"-> tfinal={TFINAL}")

rec = []
time = 0.0
niter = 0
d_t = 0.0
while time < TFINAL:
  d_t = S.stepper()
  time += d_t
  S.compute_fluxes()
  S.compute_new_val()
  ac, as_ = modal_amplitude()
  if RANK == 0:
    rec.append((time, ac, as_))
  niter += 1

if RANK == 0:
  np.savetxt(OUT, np.array(rec),
             header=(f're<hv,e^ikx>  im<hv,e^ikx>  columns: t re im  |  '
                     f'KX={KX} KY={KY} B0={B0} F0={F0} BETA={BETA} '
                     f'grav=1 H0=1'))
  print(f"wrote {OUT}  ({niter} steps, tfinal={time:.3f})")

# --------------------------------------------- final VTK snapshot (ParaView)
for var in (h, hu, hv, hB1, hB2):
  var.update_halo_value()
  var.update_ghost_value()
  var.interpolate_celltonode()
domain.save_on_node_multi(["h", "hu", "hv", "hB1", "hB2"],
                          [h.node, hu.node, hv.node, hB1.node, hB2.node],
                          d_t, time, niter, 0)

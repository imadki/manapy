#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of the k-epsilon TurbulentSWMHDSolver against an INDEPENDENT 1D
k-epsilon reference (code-to-code verification).

Temporal turbulent mixing layer u(y)=U0 tanh((y-0.5)/delta), flat bed, B=0 -> the
SWMHD solver reduces to the standard hydro k-epsilon, whose 1D self-similar mixing
layer is solved here by an independent finite-difference reference. The 2D manapy
run is uniform in x, so its y-profiles must match the 1D reference.

Outputs:
  - vtk_results/  : 2D fields (u, k_c, nu_t) for ParaView
  - mixing_validation.png : manapy y-profiles vs 1D reference (u, k, nu_t)

Run:  python validate_turbulent_mixing.py
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

# ---- model constants (shared by manapy solver and the 1D reference) ----
Cmu, Ce1, Ce2, sk, se = 0.09, 1.3, 1.8, 1.0, 1.3
nu = 2e-4
U0, delta = 0.5, 0.05
seed_k, seed_e = 1e-2, 1e-2
kf, ef = 1e-8, 1e-8
Tfinal = 0.25
Ny = 100


# ---------------- independent 1D k-epsilon reference (stateful stepper) ----------------
class Reference1D:
  """1D finite-difference k-epsilon mixing layer, advanced to a target time so it
     can be co-evolved with the 2D run and written into the VTK at every output."""
  def __init__(self):
    self.y = np.linspace(0.0, 1.0, Ny); self.dy = self.y[1] - self.y[0]
    self.u = U0 * np.tanh((self.y - 0.5) / delta)
    self.k = np.full(Ny, seed_k); self.e = np.full(Ny, seed_e)
    self.t = 0.0

  def _diff(self, w, coef):        # d/dy(coef dw/dy), Neumann ends
    cf = 0.5 * (coef[:-1] + coef[1:])
    flux = cf * (w[1:] - w[:-1]) / self.dy
    d = np.zeros(Ny)
    d[1:-1] = (flux[1:] - flux[:-1]) / self.dy
    d[0] = flux[0] / self.dy; d[-1] = -flux[-1] / self.dy
    return d

  def advance_to(self, t_target):
    while self.t < t_target - 1e-14:
      nut = Cmu * np.maximum(self.k, kf) ** 2 / np.maximum(self.e, ef)
      dt = min(0.25 * self.dy * self.dy / (nu + nut).max(), t_target - self.t)
      dudy = np.gradient(self.u, self.dy)
      P = nut * dudy * dudy
      self.u = self.u + dt * self._diff(self.u, nu + nut)
      self.k = self.k + dt * (self._diff(self.k, nu + nut / sk) + P - self.e)
      self.e = self.e + dt * (self._diff(self.e, nu + nut / se)
                              + (self.e / np.maximum(self.k, kf)) * (Ce1 * P - Ce2 * self.e))
      np.maximum(self.k, 0.0, out=self.k); np.maximum(self.e, 0.0, out=self.e)
      self.t += dt

  def fields(self):
    nut = Cmu * np.maximum(self.k, kf) ** 2 / np.maximum(self.e, ef)
    return self.u, self.k, nut


# ---------------- manapy 2D run (B=0) ----------------
BASE = os.path.dirname(os.path.realpath(__file__))
MESH = os.path.join(BASE, 'mixing_quad.msh')
if RANK == 0:
  rectangle(bounds=((0, 1), (0, 1)), n=Ny, cell_type="quad", transfinite=True, recombine=True, filename=MESH)
COMM.Barrier()
dom = Domain.create_domain(MESH, 2, Partitioning.Par_Nodal, recreate=True)
c = np.asarray(dom.cells.center); x = c[:, 0]; y = c[:, 1]
bc = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}
mkb = lambda: Variable(domain=dom, BC=bc)
h = Variable(domain=dom); hu, hv, hB1, hB2 = mkb(), mkb(), mkb(), mkb()
PSI = Variable(domain=dom); Z = Variable(domain=dom)
kc, km, epsc, epsm = mkb(), mkb(), mkb(), mkb()
h.cell[:] = 1.0
hu.cell[:] = U0 * np.tanh((y - 0.5) / delta)
hB1.cell[:] = 0.0; hB2.cell[:] = 0.0            # B=0 -> reduces to hydro k-epsilon
kc.cell[:] = seed_k; km.cell[:] = seed_k; epsc.cell[:] = seed_e; epsm.cell[:] = seed_e
S = TurbulentSWMHDSolver(h, (hu, hv), (hB1, hB2), kc, km, epsc, epsm,
                         PSI=PSI, Z=Z, nu=nu, mu=nu, cfl=0.5, grav=1.0, GLM=10,
                         Cmu=Cmu, Ce1=Ce1, Ce2=Ce2, sigma_k=sk, sigma_e=se,
                         k_floor=kf, eps_floor=ef)
# output Variables: manapy solution, 1D reference (broadcast on the mesh), and error
u_o, kc_o, nut_o = Variable(domain=dom), Variable(domain=dom), Variable(domain=dom)
u_r, kc_r, nut_r = Variable(domain=dom), Variable(domain=dom), Variable(domain=dom)
u_e, kc_e, nut_e = Variable(domain=dom), Variable(domain=dom), Variable(domain=dom)
ref = Reference1D()
VNAMES = ["u", "u_ref", "u_err", "kc", "kc_ref", "kc_err", "nu_t", "nu_t_ref", "nu_t_err"]
VVARS = [u_o, u_r, u_e, kc_o, kc_r, kc_e, nut_o, nut_r, nut_e]


def save_vtk(dt, tm, it, mit):
  hh = np.asarray(h.cell)
  um = np.asarray(hu.cell) / hh; km = np.asarray(kc.cell) / hh; nm = np.asarray(S.nu_t.cell)
  ref.advance_to(tm)                              # co-evolve the reference to this time
  ur, kr, nr = ref.fields()
  # broadcast the 1D reference (function of y) onto the 2D mesh cells
  urc = np.interp(y, ref.y, ur); krc = np.interp(y, ref.y, kr); nrc = np.interp(y, ref.y, nr)
  u_o.cell[:] = um;  u_r.cell[:] = urc;  u_e.cell[:] = np.abs(um - urc)
  kc_o.cell[:] = km; kc_r.cell[:] = krc; kc_e.cell[:] = np.abs(km - krc)
  nut_o.cell[:] = nm; nut_r.cell[:] = nrc; nut_e.cell[:] = np.abs(nm - nrc)
  for f in VVARS:
    f.update_halo_value(); f.update_ghost_value(); f.interpolate_celltonode()
  dom.save_on_node_multi(VNAMES, [f.node for f in VVARS], dt, tm, it, mit)


t = 0.0; it = 0; mit = 0
save_vtk(0.0, 0.0, 0, mit); mit += 1
while t < Tfinal:
  dt = S.step(); t += dt; it += 1
  if it % 40 == 0:
    save_vtk(dt, t, it, mit); mit += 1
save_vtk(dt, t, it, mit)

# ---------------- gather manapy y-profile (bin over x) ----------------
u2d = np.asarray(hu.cell) / np.asarray(h.cell)
k2d = np.asarray(kc.cell) / np.asarray(h.cell)
n2d = np.asarray(S.nu_t.cell)
edges = np.linspace(0, 1, Ny + 1)
su = np.zeros(Ny); skk = np.zeros(Ny); sn = np.zeros(Ny); cnt = np.zeros(Ny)
for i in range(len(y)):
  b = min(int(y[i] * Ny), Ny - 1)
  su[b] += u2d[i]; skk[b] += k2d[i]; sn[b] += n2d[i]; cnt[b] += 1
for a in (su, skk, sn, cnt):
  COMM.Allreduce(MPI.IN_PLACE, a, op=MPI.SUM)
m = cnt > 0
yc = (0.5 * (edges[:-1] + edges[1:]))[m]
u_m, k_m, nut_m = su[m] / cnt[m], skk[m] / cnt[m], sn[m] / cnt[m]

if RANK == 0:
  yr = ref.y; ur, kr, nutr = ref.fields()          # co-evolved to the final time
  uri = np.interp(yc, yr, ur); kri = np.interp(yc, yr, kr); nri = np.interp(yc, yr, nutr)
  eu = np.linalg.norm(u_m - uri) / np.linalg.norm(uri)
  ek = np.linalg.norm(k_m - kri) / np.linalg.norm(kri)
  en = np.linalg.norm(nut_m - nri) / np.linalg.norm(nri)
  print("Turbulent mixing layer: manapy 2D (B=0) vs independent 1D k-epsilon, T=%.2f" % Tfinal)
  print("  relative L2 error  u: %.2e   k: %.2e   nu_t: %.2e" % (eu, ek, en))

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(1, 3, figsize=(13, 4))
  for a, (mm, rr, lab) in zip(ax, [(u_m, ur, "u"), (k_m, kr, "k"), (nut_m, nutr, r"$\nu_t$")]):
    a.plot(yr, rr, 'k-', lw=2, label="1D reference")
    a.plot(yc, mm, 'r--o', ms=3, mfc='none', label="manapy 2D (B=0)")
    a.set_xlabel("y"); a.set_title(lab); a.legend(); a.grid(alpha=0.3)
  fig.suptitle("k-epsilon turbulent mixing layer: manapy vs 1D reference (T=%.2f)" % Tfinal)
  fig.tight_layout()
  out = os.path.join(BASE, "mixing_validation.png")
  fig.savefig(out, dpi=110)
  print("  plot: %s" % out)
  print("  VTK : %s/vtk_results/" % BASE)

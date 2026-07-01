#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compressible Euler solver (2D, cell-centered finite volume).

Conservative state: (rho, rho*u, rho*v, rho*E); P is carried as a derived
variable and recomputed from the conservative state after every update.

Two Godunov-type numerical fluxes are wired here (order 1):
  - "rusanov" (local Lax-Friedrichs)  -- robust default
  - "Roe"     (Roe approximate Riemann solver)

The characteristic finite-volume flux ("fvc") present in the kernel module
needs the upstream interpolation/predictor pipeline and is intentionally
not wired yet (see compute kernels: departure/predictor/*ValForInterp).

@author: kissami
"""
from mpi4py import MPI
import numpy as np

import manapy.solvers.euler.fvm_utils2d_compute as _fvm2d
import manapy.solvers.euler.fvm_utils3d_compute as _fvm3d
from manapy.core.Variable import Variable


class EulerSolver:
  # Numerical-flux schemes wired to the explicit kernels.
  SCHEMES = ("rusanov", "Roe")
  # Boundary treatments available as ghost-value kernels.
  BOUNDARIES = ("Neumann", "TubeSchok", "Gamm2D", "DoubleMach", "NonReflecting")

  def __init__(self,
               rho: Variable,
               P: Variable,
               rhou: Variable,
               rhov: Variable,
               rhoE: Variable,
               rhow: Variable = None,
               gamma: float = 1.4,
               cfl: float = 0.5,
               order: int = 1,
               scheme: str = "rusanov",
               bc: str = "Neumann",
               dt: float = 0.0,
               viscous: bool = False,
               mu: float = 0.0,
               Pr: float = 0.72,
               R: float = 287.0,
               viscosity_law: str = "constant",
               mu_ref: float = 1.716e-5,
               T_ref: float = 273.15,
               S_suth: float = 110.4,
               cfl_visc: float = 0.3,
               les: bool = False,
               sgs_model: str = "smagorinsky",
               Cs: float = 0.16,
               Cw: float = 0.5,
               Prt: float = 0.9,
               bc_vel: dict = None,
               bc_temp: dict = None,
               vel_values: dict = None,
               temp_values: dict = None,
               variable_gamma: bool = False,
               doubleflux: bool = False,
               entropy_fix: float = 0.0,
               rho_inf: float = 1.0,
               u_inf: float = 0.0,
               v_inf: float = 0.0,
               w_inf: float = 0.0,
               p_inf: float = 1.0):

    self.rho = rho
    self.P = P
    self.rhou = rhou
    self.rhov = rhov
    self.rhow = rhow
    self.rhoE = rhoE

    self.domain = self.rho.domain
    self.dim = self.rho.dim
    if self.dim not in (2, 3):
      raise NotImplementedError("EulerSolver supports 2D and 3D only")
    if self.dim == 3 and rhow is None:
      raise ValueError("3D EulerSolver requires the z-momentum variable rhow")
    self.comm = self.domain.halo_comm.graph_comm

    if self.domain.backend.name == "gpu":
      raise NotImplementedError("EulerSolver has no GPU backend yet (CPU/numba only)")

    # Order matters for the halo exchange loop only; every var carries a cell field.
    if self.dim == 2:
      self.vars = (rho, rhou, rhov, rhoE, P)
    else:
      self.vars = (rho, rhou, rhov, rhow, rhoE, P)

    self.gamma = float(gamma)
    self.cfl = float(cfl)
    self.order = int(order)
    self.dt = float(dt)

    if scheme not in EulerSolver.SCHEMES:
      raise ValueError(f"unknown scheme '{scheme}'; choose from {list(EulerSolver.SCHEMES)}")
    self.scheme = scheme

    # bc is either a single treatment applied to every boundary (str), or a
    # per-boundary map {boundary_name: type} for mixed BCs in one run.
    self._per_boundary = isinstance(bc, dict)
    if self._per_boundary:
      self.bc_map = {k: v.lower() for k, v in bc.items()}
      valid = {"neumann", "slipwall", "nonreflecting"}
      bad = set(self.bc_map.values()) - valid
      if bad:
        raise ValueError(f"unknown per-boundary bc type(s) {bad}; choose from {valid}")
      self.bc = "_perbnd"
    else:
      if bc not in EulerSolver.BOUNDARIES:
        raise ValueError(f"unknown bc '{bc}'; choose from {list(EulerSolver.BOUNDARIES)}")
      self.bc = bc

    if self.order == 2 and scheme != "rusanov":
      raise NotImplementedError("order 2 (MUSCL) is wired for the rusanov flux only")

    # Pick the dimension-specific kernel module and compile once on every rank.
    fvm = _fvm2d if self.dim == 2 else _fvm3d
    self._fvm = fvm
    fvm.setup(self.dim)

    nbcells = self.domain.nbcells
    nbfaces = self.domain.nbfaces
    dtype = np.asarray(self.rho.cell).dtype

    # Per-cell residual accumulators (re-zeroed inside the explicit kernel).
    self.rez_rho = np.zeros(nbcells, dtype=dtype)
    self.rez_rhou = np.zeros(nbcells, dtype=dtype)
    self.rez_rhov = np.zeros(nbcells, dtype=dtype)
    self.rez_rhow = np.zeros(nbcells, dtype=dtype)   # used in 3D only
    self.rez_rhoE = np.zeros(nbcells, dtype=dtype)
    # Per-cell stable time step scratch.
    self.dt_c = np.zeros(nbcells, dtype=dtype)
    # Ghost primitive velocities filled by the BC kernel (not stored on a Variable).
    self.ug = np.zeros(nbfaces, dtype=dtype)
    self.vg = np.zeros(nbfaces, dtype=dtype)
    self.wg = np.zeros(nbfaces, dtype=dtype)         # used in 3D only
    # Face name codes (0 inner, 1..N boundaries, 10 halo); kernels are typed uint32.
    self.face_name = np.asarray(self.domain.faces.name, dtype=np.uint32)

    d = self.dim
    self._time_step = getattr(fvm, f"time_step_euler_{d}d")
    if scheme == "rusanov":
      self._explicitscheme = getattr(fvm, f"explicitscheme_euler_{d}d_rusanov")
    else:
      self._explicitscheme = getattr(fvm, f"explicitscheme_euler_{d}d_Roe")
    # Harten entropy-fix coefficient threaded to the Roe scheme (0 = plain Roe),
    # in both 2D and 3D.
    self.entropy_fix = float(entropy_fix)
    if scheme == "Roe":
      self._scheme_tail = (self.entropy_fix,)
    else:
      self._scheme_tail = ()
    self._explicitscheme_o2 = getattr(fvm, f"explicitscheme_euler_{d}d_rusanov_o2", None)
    self._update = getattr(fvm, f"update_euler_{d}d_fvc")

    # ---- variable-gamma (multispecies) coupling -------------------------
    # When enabled, the ratio of specific heats is a per-cell field (built from
    # the composition); the Rusanov wave speed and the pressure update use it.
    self.variable_gamma = bool(variable_gamma)
    if self.variable_gamma:
      if scheme != "rusanov" or self.order != 1:
        raise NotImplementedError("variable_gamma is wired for rusanov order 1 (2D & 3D)")
      d = self.dim
      self._explicitscheme_vg = getattr(fvm, f"explicitscheme_euler_{d}d_rusanov_vg")
      self._update_vg = getattr(fvm, f"update_euler_{d}d_vg")
      self.gamma_cell = np.full(nbcells, self.gamma, dtype=dtype)
      self.gamma_ghost = np.full(nbfaces, self.gamma, dtype=dtype)
      self._gamma_var = Variable(domain=self.domain)   # to halo-exchange gamma
      self._gamma_var.cell[:] = self.gamma
      # double-flux (Abgrall-Billet): pressure-equilibrium-preserving at multi-gamma
      # contacts (needed for flames). Uses a per-cell frozen-gamma residual.
      self._doubleflux = bool(doubleflux)
      if self._doubleflux:
        self._doubleflux_residual = getattr(fvm, f"doubleflux_residual_euler_{d}d")
        self._update_df = getattr(fvm, f"update_euler_{d}d_df")
        self._cell_faceid = np.asarray(self.domain.cells.faceid, dtype=np.int32)
    else:
      self._doubleflux = False
    if self._per_boundary:
      self._setup_per_boundary()
    elif self.dim == 2:
      self._ghost_value = getattr(fvm, f"ghost_value_{bc}")
    else:
      gname = {"Neumann": "ghost_value_Neumann3D",
               "TubeSchok": "ghost_value_TubeSchok3D",
               "NonReflecting": "ghost_value_NonReflecting3D"}.get(bc, f"ghost_value_{bc}")
      self._ghost_value = getattr(fvm, gname)

    # Free-stream reference state for the characteristic far-field BC.
    self.rho_inf = float(rho_inf)
    self.u_inf = float(u_inf)
    self.v_inf = float(v_inf)
    self.w_inf = float(w_inf)
    self.p_inf = float(p_inf)
    # conservative variables that carry a reconstructed gradient at order 2 (2D MUSCL)
    if self.dim == 2:
      self._cons = (self.rho, self.rhou, self.rhov, self.rhoE)
    else:
      self._cons = (self.rho, self.rhou, self.rhov, self.rhow, self.rhoE)

    # ---- Viscous (Navier-Stokes) extension -------------------------------
    if les and not viscous:
      raise ValueError("les=True requires viscous=True (the SGS model augments the viscous stress)")
    self.viscous = bool(viscous)
    if self.viscous:
      if viscosity_law not in ("constant", "sutherland", "mixture"):
        raise ValueError("viscosity_law must be 'constant', 'sutherland' or 'mixture'")
      self.mu = float(mu)
      self.Pr = float(Pr)
      self.R = float(R)
      self.viscosity_law = viscosity_law
      # 0 constant, 1 Sutherland, 2 mixture (mu,kappa supplied per cell each step)
      self._law = {"constant": 0, "sutherland": 1, "mixture": 2}[viscosity_law]
      self.mu_ref = float(mu_ref)
      self.T_ref = float(T_ref)
      self.S_suth = float(S_suth)
      self.cfl_visc = float(cfl_visc)
      # cp from the (calorically perfect) ideal gas relation
      self.cp = self.gamma * self.R / (self.gamma - 1.0)

      # Primitive Variables used only to obtain diamond face gradients of the
      # velocity components and T. The caller-supplied BC dicts carry the physical
      # wall treatment (no-slip -> velocity dirichlet 0; isothermal/adiabatic ->
      # T dirichlet/neumann). In 3D the z-velocity w is added to the set.
      self._u = Variable(domain=self.domain, BC=bc_vel, values_dict=vel_values)
      self._v = Variable(domain=self.domain, BC=bc_vel, values_dict=vel_values)
      self._T = Variable(domain=self.domain, BC=bc_temp, values_dict=temp_values)
      if self.dim == 2:
        self._w = None
        self._prim = (self._u, self._v, self._T)
      else:
        self._w = Variable(domain=self.domain, BC=bc_vel, values_dict=vel_values)
        self._prim = (self._u, self._v, self._w, self._T)

      # Per-face transport properties; constant law is filled once here.
      self.mu_face = np.zeros(nbfaces, dtype=dtype)
      self.kappa_face = np.zeros(nbfaces, dtype=dtype)
      self._face_transport_props = getattr(fvm, f"face_transport_props_{d}d")
      self._explicitscheme_viscous = getattr(fvm, f"explicitscheme_euler_{d}d_viscous")
      self._viscous_time_step = getattr(fvm, f"viscous_time_step_{d}d")
      if self._law == 0:
        self.mu_face[:] = self.mu
        self.kappa_face[:] = self.mu * self.cp / self.Pr
      elif self._law == 2:
        # mixture transport: mu, kappa are supplied per cell each step via
        # set_transport(); they are face-averaged into the laminar base.
        self.mu_cell = np.full(nbcells, self.mu, dtype=dtype)
        self.kappa_cell = np.full(nbcells, self.mu * self.cp / self.Pr, dtype=dtype)
        self.mu_face_lam = np.zeros(nbfaces, dtype=dtype)
        self.kappa_face_lam = np.zeros(nbfaces, dtype=dtype)
        self._mu_var = Variable(domain=self.domain)
        self._kappa_var = Variable(domain=self.domain)

      # ---- LES / subgrid-scale eddy viscosity ----------------------------
      # mu_t (from the resolved strain rate) is added to the laminar mu on every
      # face each iteration; the turbulent conductivity uses Pr_t. Requires the
      # viscous path (it augments the same Newtonian stress).
      self.les = bool(les)
      if self.les:
        if sgs_model not in ("smagorinsky", "wale"):
          raise ValueError("sgs_model must be 'smagorinsky' or 'wale'")
        self.sgs_model = sgs_model
        self.Cs = float(Cs)
        self.Cw = float(Cw)
        self.Prt = float(Prt)
        # filter width per cell: delta = volume^(1/dim)
        self.delta = np.asarray(self.domain.cells.volume, dtype=dtype) ** (1.0 / self.dim)
        # eddy viscosity carried on a Variable to reuse the halo exchange
        self._mut = Variable(domain=self.domain)
        self._mu_sgs = getattr(fvm, f"mu_sgs_{sgs_model}_{d}d")
        self._add_sgs_face_props = getattr(fvm, f"add_sgs_face_props_{d}d")
      else:
        self.les = False
    else:
      self.les = False

  def stepper(self):
    """CFL-limited explicit time step, reduced to the global minimum."""
    if self.dim == 2:
      self._time_step(self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell,
                      self.cfl, self.domain.faces.normal, self.domain.faces.mesure,
                      self.domain.cells.volume, self.domain.cells.faceid,
                      self.gamma, self.dt_c)
    else:
      self._time_step(self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell,
                      self.rhow.cell, self.cfl, self.domain.faces.normal,
                      self.domain.faces.mesure, self.domain.cells.volume,
                      self.domain.cells.faceid, self.gamma, self.dt_c)
    d_t = self.dt_c.min()
    if self.viscous:
      # primitives are also reused by compute_viscous_fluxes() this iteration
      self._compute_primitives()
      if self._law == 1:
        self._refresh_transport_props()
      self._viscous_time_step(self.rho.cell, self.mu_face, self.kappa_face, self.cp,
                              self.cfl_visc, self.domain.faces.mesure,
                              self.domain.cells.volume, self.domain.cells.faceid, self.dt_c)
      d_t = min(d_t, self.dt_c.min())
    self.dt = self.comm.allreduce(d_t, op=MPI.MIN)
    return self.dt

  def update_halo_values(self):
    for var in self.vars:
      var.update_halo_value()

  def _setup_per_boundary(self):
    """Precompute, per BC type, the boundary face indices, owner cells and unit
    outward normals (2D & 3D). Several named boundaries can share a type."""
    dom = self.domain
    name = self.face_name
    fn = np.asarray(dom.faces.normal)
    mesure = np.asarray(dom.faces.mesure)
    cellid = np.asarray(dom.faces.cellid)
    groups = {}
    for bname, btype in self.bc_map.items():
      code = dom.BCs[bname][1]
      f = np.nonzero(name == code)[0]
      if f.size == 0:
        continue
      n = fn[f][:, :self.dim] / mesure[f][:, None]
      groups.setdefault(btype, []).append((f, cellid[f, 0], n))
    self._bnd = {t: (np.concatenate([g[0] for g in lst]),
                     np.concatenate([g[1] for g in lst]),
                     np.concatenate([g[2] for g in lst]))
                 for t, lst in groups.items()}

  def _apply_per_boundary_ghosts(self, t: float = 0.0):
    """Fill the ghost state on each boundary according to its per-boundary type:
    'neumann' (zero-gradient), 'slipwall' (reflect the normal velocity) or
    'nonreflecting' (characteristic far-field, Riemann invariants)."""
    if self.dim == 3:
      self._apply_per_boundary_ghosts_3d(t)
      return
    rho, P, rhou, rhov, rhoE = self.rho, self.P, self.rhou, self.rhov, self.rhoE
    g = self.gamma
    for btype, (f, il, n) in self._bnd.items():
      rl = rho.cell[il]; pl = P.cell[il]
      ul = rhou.cell[il] / rl; vl = rhov.cell[il] / rl
      nx = n[:, 0]; ny = n[:, 1]
      if btype == "neumann":
        rho.ghost[f] = rl; P.ghost[f] = pl
        rhou.ghost[f] = rhou.cell[il]; rhov.ghost[f] = rhov.cell[il]
        rhoE.ghost[f] = rhoE.cell[il]
        self.ug[f] = ul; self.vg[f] = vl
      elif btype == "slipwall":
        un = ul * nx + vl * ny
        ug = ul - 2.0 * un * nx; vg = vl - 2.0 * un * ny
        rho.ghost[f] = rl; P.ghost[f] = pl
        rhou.ghost[f] = rl * ug; rhov.ghost[f] = rl * vg
        rhoE.ghost[f] = pl / (g - 1.0) + 0.5 * rl * (ug * ug + vg * vg)
        self.ug[f] = ug; self.vg[f] = vg
      else:  # nonreflecting: characteristic far-field via Riemann invariants
        cl = np.sqrt(g * pl / rl)
        unl = ul * nx + vl * ny
        c_inf = np.sqrt(g * self.p_inf / self.rho_inf)
        un_inf = self.u_inf * nx + self.v_inf * ny
        Rp = unl + 2.0 * cl / (g - 1.0)
        Rm = un_inf - 2.0 * c_inf / (g - 1.0)
        un_b = 0.5 * (Rp + Rm)
        c_b = 0.25 * (g - 1.0) * (Rp - Rm)
        inflow = un_b <= 0.0
        s_b = np.where(inflow, self.p_inf / self.rho_inf ** g, pl / rl ** g)
        utx = np.where(inflow, self.u_inf - un_inf * nx, ul - unl * nx)
        uty = np.where(inflow, self.v_inf - un_inf * ny, vl - unl * ny)
        rho_b = (c_b * c_b / (g * s_b)) ** (1.0 / (g - 1.0))
        p_b = rho_b * c_b * c_b / g
        # supersonic overrides
        suproff = unl >= cl
        supinf = unl <= -cl
        u_b = un_b * nx + utx; v_b = un_b * ny + uty
        rho_b = np.where(suproff, rl, np.where(supinf, self.rho_inf, rho_b))
        p_b = np.where(suproff, pl, np.where(supinf, self.p_inf, p_b))
        u_b = np.where(suproff, ul, np.where(supinf, self.u_inf, u_b))
        v_b = np.where(suproff, vl, np.where(supinf, self.v_inf, v_b))
        rho.ghost[f] = rho_b; P.ghost[f] = p_b
        rhou.ghost[f] = rho_b * u_b; rhov.ghost[f] = rho_b * v_b
        rhoE.ghost[f] = p_b / (g - 1.0) + 0.5 * rho_b * (u_b * u_b + v_b * v_b)
        self.ug[f] = u_b; self.vg[f] = v_b

  def _apply_per_boundary_ghosts_3d(self, t: float = 0.0):
    """3D analogue of _apply_per_boundary_ghosts (adds the z-momentum / w-velocity)."""
    rho, P = self.rho, self.P
    rhou, rhov, rhow, rhoE = self.rhou, self.rhov, self.rhow, self.rhoE
    g = self.gamma
    for btype, (f, il, n) in self._bnd.items():
      rl = rho.cell[il]; pl = P.cell[il]
      ul = rhou.cell[il] / rl; vl = rhov.cell[il] / rl; wl = rhow.cell[il] / rl
      nx = n[:, 0]; ny = n[:, 1]; nz = n[:, 2]
      if btype == "neumann":
        rho.ghost[f] = rl; P.ghost[f] = pl
        rhou.ghost[f] = rhou.cell[il]; rhov.ghost[f] = rhov.cell[il]; rhow.ghost[f] = rhow.cell[il]
        rhoE.ghost[f] = rhoE.cell[il]
        self.ug[f] = ul; self.vg[f] = vl; self.wg[f] = wl
      elif btype == "slipwall":
        un = ul * nx + vl * ny + wl * nz
        ug = ul - 2.0 * un * nx; vg = vl - 2.0 * un * ny; wg = wl - 2.0 * un * nz
        rho.ghost[f] = rl; P.ghost[f] = pl
        rhou.ghost[f] = rl * ug; rhov.ghost[f] = rl * vg; rhow.ghost[f] = rl * wg
        rhoE.ghost[f] = pl / (g - 1.0) + 0.5 * rl * (ug * ug + vg * vg + wg * wg)
        self.ug[f] = ug; self.vg[f] = vg; self.wg[f] = wg
      else:  # nonreflecting: characteristic far-field via Riemann invariants
        cl = np.sqrt(g * pl / rl)
        unl = ul * nx + vl * ny + wl * nz
        c_inf = np.sqrt(g * self.p_inf / self.rho_inf)
        un_inf = self.u_inf * nx + self.v_inf * ny + self.w_inf * nz
        Rp = unl + 2.0 * cl / (g - 1.0)
        Rm = un_inf - 2.0 * c_inf / (g - 1.0)
        un_b = 0.5 * (Rp + Rm)
        c_b = 0.25 * (g - 1.0) * (Rp - Rm)
        inflow = un_b <= 0.0
        s_b = np.where(inflow, self.p_inf / self.rho_inf ** g, pl / rl ** g)
        utx = np.where(inflow, self.u_inf - un_inf * nx, ul - unl * nx)
        uty = np.where(inflow, self.v_inf - un_inf * ny, vl - unl * ny)
        utz = np.where(inflow, self.w_inf - un_inf * nz, wl - unl * nz)
        rho_b = (c_b * c_b / (g * s_b)) ** (1.0 / (g - 1.0))
        p_b = rho_b * c_b * c_b / g
        suproff = unl >= cl
        supinf = unl <= -cl
        u_b = un_b * nx + utx; v_b = un_b * ny + uty; w_b = un_b * nz + utz
        rho_b = np.where(suproff, rl, np.where(supinf, self.rho_inf, rho_b))
        p_b = np.where(suproff, pl, np.where(supinf, self.p_inf, p_b))
        u_b = np.where(suproff, ul, np.where(supinf, self.u_inf, u_b))
        v_b = np.where(suproff, vl, np.where(supinf, self.v_inf, v_b))
        w_b = np.where(suproff, wl, np.where(supinf, self.w_inf, w_b))
        rho.ghost[f] = rho_b; P.ghost[f] = p_b
        rhou.ghost[f] = rho_b * u_b; rhov.ghost[f] = rho_b * v_b; rhow.ghost[f] = rho_b * w_b
        rhoE.ghost[f] = p_b / (g - 1.0) + 0.5 * rho_b * (u_b * u_b + v_b * v_b + w_b * w_b)
        self.ug[f] = u_b; self.vg[f] = v_b; self.wg[f] = w_b

  def _face_avg(self, cellarr, haloarr):
    """Average a per-cell array to faces (owner at boundaries, halo-aware)."""
    cid = self.domain.faces.cellid
    hid = self.domain.faces.halofid
    name = self.face_name
    il = cid[:, 0]
    out = cellarr[il].astype(np.asarray(cellarr).dtype, copy=True)
    inner = name == 0
    out[inner] = 0.5 * (cellarr[il[inner]] + cellarr[cid[inner, 1]])
    halo = name == 10
    out[halo] = 0.5 * (cellarr[il[halo]] + haloarr[hid[halo]])
    return out

  def set_transport(self, mu_cell, kappa_cell):
    """Supply per-cell dynamic viscosity and conductivity (mixture law). They are
    face-averaged into the laminar transport base used by the viscous flux."""
    self.mu_cell[:] = mu_cell
    self.kappa_cell[:] = kappa_cell
    self._mu_var.cell[:] = mu_cell; self._mu_var.update_halo_value()
    self._kappa_var.cell[:] = kappa_cell; self._kappa_var.update_halo_value()
    self.mu_face_lam[:] = self._face_avg(self.mu_cell, self._mu_var.halo)
    self.kappa_face_lam[:] = self._face_avg(self.kappa_cell, self._kappa_var.halo)
    self.mu_face[:] = self.mu_face_lam
    self.kappa_face[:] = self.kappa_face_lam

  def set_gamma(self, gamma_cell):
    """Set the per-cell ratio of specific heats (variable_gamma path) and refresh
    its ghost (zero-gradient) and halo values for the flux."""
    self.gamma_cell[:] = gamma_cell
    self.gamma_ghost[:] = self.gamma_cell[self.domain.faces.cellid[:, 0]]
    self._gamma_var.cell[:] = self.gamma_cell
    self._gamma_var.update_halo_value()
    self.gamma_halo = self._gamma_var.halo

  def update_ghost_values(self, t: float = 0.0):
    if self._per_boundary:
      self._apply_per_boundary_ghosts(t)
      return
    if self.bc == "NonReflecting":
      # characteristic far-field kernel: extra free-stream reference arguments
      if self.dim == 2:
        self._ghost_value(self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost,
                          self.ug, self.vg, self.rhoE.ghost,
                          self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,
                          self.domain.faces.cellid, self.face_name,
                          self.domain.faces.normal, self.domain.faces.mesure,
                          self.domain.faces.center, t,
                          self.gamma, self.rho_inf, self.u_inf, self.v_inf, self.p_inf)
      else:
        self._ghost_value(self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost,
                          self.rhow.ghost, self.ug, self.vg, self.wg, self.rhoE.ghost,
                          self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell,
                          self.rhow.cell, self.rhoE.cell,
                          self.domain.faces.cellid, self.face_name,
                          self.domain.faces.normal, self.domain.faces.mesure,
                          self.gamma, self.rho_inf, self.u_inf, self.v_inf, self.w_inf, self.p_inf)
      return
    if self.dim == 2:
      self._ghost_value(self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost,
                        self.ug, self.vg, self.rhoE.ghost,
                        self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,
                        self.domain.faces.cellid, self.face_name,
                        self.domain.faces.normal, self.domain.faces.mesure,
                        self.domain.faces.center, t)
    else:
      self._ghost_value(self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost,
                        self.rhow.ghost, self.ug, self.vg, self.wg, self.rhoE.ghost,
                        self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell,
                        self.rhow.cell, self.rhoE.cell,
                        self.domain.faces.cellid, self.face_name,
                        self.domain.faces.normal, self.domain.faces.mesure)

  def compute_fluxes(self, t: float = 0.0):
    # neighbour (halo) cell values, then physical-boundary ghost values
    self.update_halo_values()
    self.update_ghost_values(t)

    if self.order >= 2:
      self._explicit_o2()
    elif self.variable_gamma and self._doubleflux and self.dim == 2:
      self._doubleflux_residual(self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhoE,
                                self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell,
                                self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost,
                                self.rho.halo, self.P.halo, self.rhou.halo, self.rhov.halo,
                                self.gamma_cell, self._cell_faceid,
                                self.domain.faces.cellid, self.domain.faces.halofid,
                                self.domain.faces.normal, self.domain.faces.mesure, self.face_name)
    elif self.variable_gamma and self._doubleflux:
      self._doubleflux_residual(self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhow, self.rez_rhoE,
                                self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhow.cell,
                                self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost, self.rhow.ghost,
                                self.rho.halo, self.P.halo, self.rhou.halo, self.rhov.halo, self.rhow.halo,
                                self.gamma_cell, self._cell_faceid,
                                self.domain.faces.cellid, self.domain.faces.halofid,
                                self.domain.faces.normal, self.domain.faces.mesure, self.face_name)
    elif self.variable_gamma and self.dim == 2:
      self._explicitscheme_vg(self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhoE,
                              self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,
                              self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost, self.rhoE.ghost,
                              self.rho.halo, self.P.halo, self.rhou.halo, self.rhov.halo, self.rhoE.halo,
                              self.domain.faces.cellid, self.domain.faces.halofid,
                              self.domain.faces.normal, self.domain.faces.mesure,
                              self.face_name, self.gamma_cell, self.gamma_ghost, self.gamma_halo)
    elif self.variable_gamma:
      self._explicitscheme_vg(self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhow, self.rez_rhoE,
                              self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhow.cell, self.rhoE.cell,
                              self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost, self.rhow.ghost, self.rhoE.ghost,
                              self.rho.halo, self.P.halo, self.rhou.halo, self.rhov.halo, self.rhow.halo, self.rhoE.halo,
                              self.domain.faces.cellid, self.domain.faces.halofid,
                              self.domain.faces.normal, self.domain.faces.mesure,
                              self.face_name, self.gamma_cell, self.gamma_ghost, self.gamma_halo)
    elif self.dim == 2:
      self._explicitscheme(self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhoE,
                           self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,
                           self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost, self.rhoE.ghost,
                           self.rho.halo, self.P.halo, self.rhou.halo, self.rhov.halo, self.rhoE.halo,
                           self.domain.faces.cellid, self.domain.faces.halofid,
                           self.domain.faces.normal, self.domain.faces.mesure,
                           self.face_name, self.gamma, *self._scheme_tail)
    else:
      self._explicitscheme(self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhow, self.rez_rhoE,
                           self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhow.cell, self.rhoE.cell,
                           self.rho.ghost, self.P.ghost, self.rhou.ghost, self.rhov.ghost, self.rhow.ghost, self.rhoE.ghost,
                           self.rho.halo, self.P.halo, self.rhou.halo, self.rhov.halo, self.rhow.halo, self.rhoE.halo,
                           self.domain.faces.cellid, self.domain.faces.halofid,
                           self.domain.faces.normal, self.domain.faces.mesure,
                           self.face_name, self.gamma,
                           self.domain.faces.tangent, self.domain.faces.binormal,
                           *self._scheme_tail)

    # viscous (Navier-Stokes) contribution, added into the same rez_* accumulators
    if self.viscous:
      self.compute_viscous_fluxes()

  def _compute_primitives(self):
    """Fill the primitive Variables u, v, (w,) T from the conservative state."""
    rho = self.rho.cell
    self._u.cell[:] = self.rhou.cell / rho
    self._v.cell[:] = self.rhov.cell / rho
    if self.dim == 3:
      self._w.cell[:] = self.rhow.cell / rho
    self._T.cell[:] = self.P.cell / (rho * self.R)

  def _refresh_transport_props(self):
    """Recompute per-face mu, kappa (Sutherland only; constant is set once)."""
    self._T.update_halo_value()
    self._T.update_ghost_value()
    self._face_transport_props(self.mu_face, self.kappa_face,
                               self._T.cell, self._T.ghost, self._T.halo,
                               self.domain.faces.cellid, self.domain.faces.halofid,
                               self.face_name, self._law, self.mu,
                               self.mu_ref, self.T_ref, self.S_suth, self.cp, self.Pr)

  def _compute_sgs(self):
    """LES: add the SGS eddy viscosity (Smagorinsky/WALE) to the face props.

    Uses unlimited cell gradients of the resolved velocity. For a constant
    laminar law the face props are reset to the laminar base first (Sutherland
    has already been refreshed this iteration), then the turbulent part is added.
    """
    if self._law == 0:
      self.mu_face[:] = self.mu
      self.kappa_face[:] = self.mu * self.cp / self.Pr
    elif self._law == 2:
      self.mu_face[:] = self.mu_face_lam
      self.kappa_face[:] = self.kappa_face_lam

    # unlimited resolved-velocity cell gradients (halo gradients exchanged too)
    self._u.compute_cell_gradient()
    self._v.compute_cell_gradient()
    coef = self.Cs if self.sgs_model == "smagorinsky" else self.Cw
    if self.dim == 2:
      self._mu_sgs(self._mut.cell, self.rho.cell,
                   self._u.gradcellx, self._u.gradcelly,
                   self._v.gradcellx, self._v.gradcelly,
                   self.delta, coef)
    else:
      self._w.compute_cell_gradient()
      self._mu_sgs(self._mut.cell, self.rho.cell,
                   self._u.gradcellx, self._u.gradcelly, self._u.gradcellz,
                   self._v.gradcellx, self._v.gradcelly, self._v.gradcellz,
                   self._w.gradcellx, self._w.gradcelly, self._w.gradcellz,
                   self.delta, coef)
    # turbulent viscosity on the halo cells for the face average
    self._mut.update_halo_value()
    self._add_sgs_face_props(self.mu_face, self.kappa_face,
                             self._mut.cell, self._mut.halo,
                             self.domain.faces.cellid, self.domain.faces.halofid,
                             self.face_name, self.cp, self.Prt)

  def compute_viscous_fluxes(self):
    self._compute_primitives()
    # diamond face gradients of u, v, T (same pipeline as DiffusionSolver)
    for var in self._prim:
      var.update_halo_value()
      var.update_ghost_value()
      var.interpolate_celltonode()
      var.compute_face_gradient()
    if self._law == 1:
      self._face_transport_props(self.mu_face, self.kappa_face,
                                 self._T.cell, self._T.ghost, self._T.halo,
                                 self.domain.faces.cellid, self.domain.faces.halofid,
                                 self.face_name, self._law, self.mu,
                                 self.mu_ref, self.T_ref, self.S_suth, self.cp, self.Pr)
    if self.les:
      self._compute_sgs()
    if self.dim == 2:
      self._explicitscheme_viscous(
          self.rez_rhou, self.rez_rhov, self.rez_rhoE,
          self._u.gradfacex, self._u.gradfacey, self._v.gradfacex, self._v.gradfacey,
          self._T.gradfacex, self._T.gradfacey,
          self._u.cell, self._u.ghost, self._u.halo,
          self._v.cell, self._v.ghost, self._v.halo,
          self.mu_face, self.kappa_face,
          self.domain.faces.cellid, self.domain.faces.halofid,
          self.domain.faces.normal, self.face_name)
    else:
      self._explicitscheme_viscous(
          self.rez_rhou, self.rez_rhov, self.rez_rhow, self.rez_rhoE,
          self._u.gradfacex, self._u.gradfacey, self._u.gradfacez,
          self._v.gradfacex, self._v.gradfacey, self._v.gradfacez,
          self._w.gradfacex, self._w.gradfacey, self._w.gradfacez,
          self._T.gradfacex, self._T.gradfacey, self._T.gradfacez,
          self._u.cell, self._u.ghost, self._u.halo,
          self._v.cell, self._v.ghost, self._v.halo,
          self._w.cell, self._w.ghost, self._w.halo,
          self.mu_face, self.kappa_face,
          self.domain.faces.cellid, self.domain.faces.halofid,
          self.domain.faces.normal, self.face_name)

  def _explicit_o2(self):
    # limited cell gradients of each conservative variable (also exchanges the
    # halo gradients and limiter), then the MUSCL Rusanov scheme.
    for var in self._cons:
      var.compute_cell_gradient()

    if self.dim == 2:
      rho, rhou, rhov, rhoE = self._cons
      self._explicitscheme_o2(
          self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhoE,
          rho.cell, rhou.cell, rhov.cell, rhoE.cell,
          rho.ghost, rhou.ghost, rhov.ghost, rhoE.ghost,
          rho.halo, rhou.halo, rhov.halo, rhoE.halo,
          rho.gradcellx, rho.gradcelly, rhou.gradcellx, rhou.gradcelly,
          rhov.gradcellx, rhov.gradcelly, rhoE.gradcellx, rhoE.gradcelly,
          rho.gradhalocellx, rho.gradhalocelly, rhou.gradhalocellx, rhou.gradhalocelly,
          rhov.gradhalocellx, rhov.gradhalocelly, rhoE.gradhalocellx, rhoE.gradhalocelly,
          rho.psi, rhou.psi, rhov.psi, rhoE.psi,
          rho.psihalo, rhou.psihalo, rhov.psihalo, rhoE.psihalo,
          self.domain.faces.cellid, self.domain.faces.halofid,
          self.domain.faces.normal, self.domain.faces.mesure, self.face_name,
          self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol,
          self.gamma, self.order)
    else:
      rho, rhou, rhov, rhow, rhoE = self._cons
      self._explicitscheme_o2(
          self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhow, self.rez_rhoE,
          rho.cell, rhou.cell, rhov.cell, rhow.cell, rhoE.cell,
          rho.ghost, rhou.ghost, rhov.ghost, rhow.ghost, rhoE.ghost,
          rho.halo, rhou.halo, rhov.halo, rhow.halo, rhoE.halo,
          rho.gradcellx, rho.gradcelly, rho.gradcellz,
          rhou.gradcellx, rhou.gradcelly, rhou.gradcellz,
          rhov.gradcellx, rhov.gradcelly, rhov.gradcellz,
          rhow.gradcellx, rhow.gradcelly, rhow.gradcellz,
          rhoE.gradcellx, rhoE.gradcelly, rhoE.gradcellz,
          rho.gradhalocellx, rho.gradhalocelly, rho.gradhalocellz,
          rhou.gradhalocellx, rhou.gradhalocelly, rhou.gradhalocellz,
          rhov.gradhalocellx, rhov.gradhalocelly, rhov.gradhalocellz,
          rhow.gradhalocellx, rhow.gradhalocelly, rhow.gradhalocellz,
          rhoE.gradhalocellx, rhoE.gradhalocelly, rhoE.gradhalocellz,
          rho.psi, rhou.psi, rhov.psi, rhow.psi, rhoE.psi,
          rho.psihalo, rhou.psihalo, rhov.psihalo, rhow.psihalo, rhoE.psihalo,
          self.domain.faces.cellid, self.domain.faces.halofid,
          self.domain.faces.normal, self.domain.faces.mesure, self.face_name,
          self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol,
          self.gamma, self.order)

  def compute_new_val(self):
    if self.variable_gamma and self._doubleflux and self.dim == 2:
      self._update_df(self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,
                      self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhoE,
                      self.gamma_cell, self.dt, self.domain.cells.volume)
    elif self.variable_gamma and self._doubleflux:
      self._update_df(self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhow.cell, self.rhoE.cell,
                      self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhow, self.rez_rhoE,
                      self.gamma_cell, self.dt, self.domain.cells.volume)
    elif self.variable_gamma and self.dim == 2:
      self._update_vg(self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,
                      self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhoE,
                      self.gamma_cell, self.dt, self.domain.cells.volume)
    elif self.variable_gamma:
      self._update_vg(self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhow.cell, self.rhoE.cell,
                      self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhow, self.rez_rhoE,
                      self.gamma_cell, self.dt, self.domain.cells.volume)
    elif self.dim == 2:
      self._update(self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhoE.cell,
                   self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhoE,
                   self.gamma, self.dt, self.domain.cells.volume)
    else:
      self._update(self.rho.cell, self.P.cell, self.rhou.cell, self.rhov.cell, self.rhow.cell, self.rhoE.cell,
                   self.rez_rho, self.rez_rhou, self.rez_rhov, self.rez_rhow, self.rez_rhoE,
                   self.gamma, self.dt, self.domain.cells.volume)

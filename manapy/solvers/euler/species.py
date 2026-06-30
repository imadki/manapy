#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passive multispecies transport riding on the compressible Euler/NS solver.

Transports N species mass fractions Y_k (as partial densities q_k = rho*Y_k) with
a Rusanov convective flux consistent with the bulk density flux. This is the
precursor to reactive chemistry (Phase 5): the same q_k carry the chemical state,
later augmented with diffusion (Phase 6) and reaction source terms (Phase 5).

The bulk EulerSolver owns rho, velocity and pressure; SpeciesTransport reads them
and advances its own q_k. Mass fractions are recovered as Y_k = q_k / rho and, if
requested, renormalised so sum_k Y_k = 1.

Usage:
    sp = SpeciesTransport(solver, Y0=[Y_fuel, Y_oxidizer], names=["F","O"])
    ...
    while t < tend:
        dt = solver.stepper()
        solver.compute_fluxes(t); solver.compute_new_val()
        sp.advance(dt)            # advect species with the updated flow
"""
import numpy as np

from manapy.core.Variable import Variable
import manapy.solvers.euler.species_compute as _spec


class SpeciesTransport:

  def __init__(self, solver, Y0, names=None, renormalize=True):
    self.solver = solver
    self.domain = solver.domain
    self.dim = solver.dim
    self.gamma = solver.gamma
    self.comm = solver.comm
    self.renormalize = bool(renormalize)

    self.nspec = len(Y0)
    if self.nspec < 1:
      raise ValueError("need at least one species")
    self.names = names if names is not None else [f"Y{k}" for k in range(self.nspec)]

    _spec.setup(self.dim)
    self._scheme = (_spec.explicitscheme_species_2d if self.dim == 2
                    else _spec.explicitscheme_species_3d)
    self._update = _spec.update_species

    nbcells = self.domain.nbcells
    dtype = np.asarray(solver.rho.cell).dtype

    # partial densities q_k = rho * Y_k carried on Variables (halo exchange reuse)
    rho = solver.rho.cell
    self.q = []
    self.rez = []
    for k in range(self.nspec):
      qk = Variable(domain=self.domain)
      Yk = np.asarray(Y0[k], dtype=dtype)
      qk.cell[:] = rho * Yk          # broadcast handles scalar or per-cell Y0
      self.q.append(qk)
      self.rez.append(np.zeros(nbcells, dtype=dtype))
    self._Yvar = None                  # lazily created Variables for diffusion gradients

  def _refresh_flow_halos(self):
    self.solver.rho.update_halo_value()
    self.solver.P.update_halo_value()
    self.solver.rhou.update_halo_value()
    self.solver.rhov.update_halo_value()
    if self.dim == 3:
      self.solver.rhow.update_halo_value()

  def advance(self, dt):
    """One explicit step of species advection with the current flow field."""
    s = self.solver
    self._refresh_flow_halos()
    cellid = self.domain.faces.cellid
    halofid = self.domain.faces.halofid
    normal = self.domain.faces.normal
    mesure = self.domain.faces.mesure
    vol = self.domain.cells.volume
    name = s.face_name

    for k in range(self.nspec):
      qk = self.q[k]
      rez = self.rez[k]
      qk.update_halo_value()
      if self.dim == 2:
        self._scheme(rez, qk.cell, qk.halo,
                     s.rho.cell, s.P.cell, s.rhou.cell, s.rhov.cell,
                     s.rho.halo, s.P.halo, s.rhou.halo, s.rhov.halo,
                     cellid, halofid, normal, mesure, name, self.gamma)
      else:
        self._scheme(rez, qk.cell, qk.halo,
                     s.rho.cell, s.P.cell, s.rhou.cell, s.rhov.cell, s.rhow.cell,
                     s.rho.halo, s.P.halo, s.rhou.halo, s.rhov.halo, s.rhow.halo,
                     cellid, halofid, normal, mesure, name, self.gamma)
      self._update(qk.cell, rez, dt, vol)

    if self.renormalize:
      self._renormalize()

  def _to_face(self, cellarr):
    """Average a per-cell array to faces (owner at physical boundaries, halo-aware)."""
    cid = self.domain.faces.cellid
    hid = self.domain.faces.halofid
    name = self.solver.face_name
    il = cid[:, 0]
    out = cellarr[il].copy()
    inner = name == 0
    out[inner] = 0.5 * (cellarr[il[inner]] + cellarr[cid[inner, 1]])
    return out

  def diffuse(self, dt, D):
    """Fickian diffusion sub-step of all species over dt.

    D : per-species mass diffusion coefficient(s) [m^2/s], either a length-nspec
        sequence (constant per species) or an (nspec, ncells) array (e.g. the
        mixture-averaged D_k from CanteraChemistry.transport_array). With equal
        D_k the species-sum (rho) is preserved; unequal D_k do not conserve the
        sum exactly (a correction-velocity closure is a later refinement)."""
    if self._Yvar is None:
      self._Yvar = [Variable(domain=self.domain) for _ in range(self.nspec)]
    s = self.solver
    rho = s.rho.cell
    rho_f = self._to_face(rho)
    cellid = self.domain.faces.cellid
    normal = self.domain.faces.normal
    vol = self.domain.cells.volume
    name = s.face_name
    for k in range(self.nspec):
      Yk = self._Yvar[k]
      Yk.cell[:] = self.q[k].cell / rho
      Yk.update_halo_value(); Yk.update_ghost_value()
      Yk.interpolate_celltonode(); Yk.compute_face_gradient()
      Dk = D[k]
      Dk_face = self._to_face(np.asarray(Dk) * np.ones_like(rho)) if np.ndim(Dk) == 0 \
          else self._to_face(np.asarray(Dk))
      coef_f = rho_f * Dk_face
      gz = Yk.gradfacez if self.dim == 3 else Yk.gradfacey  # gz unused in 2D kernel branch
      _spec.explicitscheme_diffusion(self.rez[k], Yk.gradfacex, Yk.gradfacey, gz,
                                     coef_f, cellid, normal, name, self.dim)
      self._update(self.q[k].cell, self.rez[k], dt, vol)
    if self.renormalize:
      self._renormalize()

  def _renormalize(self):
    """Rescale partial densities so sum_k q_k = rho exactly (mass-fraction sum 1)."""
    rho = self.solver.rho.cell
    tot = np.zeros_like(rho)
    for qk in self.q:
      tot += qk.cell
    # avoid div-by-zero; where tot==0 leave as is
    scale = np.where(tot > 0.0, rho / tot, 1.0)
    for qk in self.q:
      qk.cell[:] *= scale

  def mass_fractions(self):
    rho = self.solver.rho.cell
    return [qk.cell / rho for qk in self.q]

  def total_mass(self, k):
    """Global integral of partial density q_k (species mass), for conservation checks."""
    from mpi4py import MPI
    vol = self.domain.cells.volume
    loc = float(np.sum(vol * self.q[k].cell))
    return self.comm.allreduce(loc, op=MPI.SUM)

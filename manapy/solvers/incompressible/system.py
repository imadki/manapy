#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incompressible Navier-Stokes by a face-flux-consistent projection (Chorin) method on
the unstructured collocated finite-volume grid -- the manapy analogue of OpenFOAM's
icoFoam (laminar, transient, single phase).

Per step (velocity u=(u,v), pressure P, kinematic viscosity nu, density rho):
  1. predictor  u* = u^n + dt (-conv + nu*diff)              (conv by the div-free flux)
  2. face flux  phi* = u*_face . S_f
  3. pressure   A P = -(rho/dt) sum_f phi*_f                  (two-point Laplacian)
  4. correct    phi = phi* - (dt/rho) a_f (P_N - P_P)         (divergence-free by design)
                u   = u*   - (dt/rho) grad(P)                 (cell reconstruction)

All three operators (divergence, pressure Laplacian, correction) share the same
two-point face coefficient a_f = area/dist, so the corrected face flux is exactly
divergence-free -- this is what makes the collocated method stable. Validated on the
lid-driven cavity vs Ghia et al. (1982). Serial (direct sparse factorisation of A);
an MPI/PETSc assembly of the same operator is the next step.
"""
import numpy as np
from mpi4py import MPI

from manapy.solvers.incompressible.fvm_utils_compute import get_kernels


class IncompressibleSolver:

  def __init__(self, u, v, P, nu=1e-2, rho=1.0, cfl=0.4, ncorr=2, u_bc=None, v_bc=None,
               poisson=None, implicit_momentum=False, mom_predictor=True, momentum=None,
               backend='mumps', alpha=None, rho1=None, rho2=None, mu1=None, mu2=None,
               cAlpha=1.0, gravity=(0.0, 0.0), sigma=0.0,
               n_outer=1, n_alpha_sub=1, ddt_corr=False, conv_order=1, n_nonorth=0):
    """
    u, v, P : cell velocity / pressure Variables. P must carry BCs (Neumann on the
              walls + one Dirichlet reference so the pure-Neumann system is regular),
              e.g. Variable(domain, BC={..., 'bottom':'dirichlet'}, values_dict={'bottom':0}).
    nu, rho : kinematic viscosity, density.
    u_bc, v_bc : {boundary_name: wall velocity component} (default 0 on every wall).
    backend : which manapy LS backend to build the default solvers with when `poisson`
              / `momentum` are not given -- 'mumps' (direct, COO-native assembly, the
              default) or 'petsc' (Krylov). Ignored for any solver passed explicitly.
    poisson : a manapy LinearSolver for the pressure Poisson -- the backend is the
              caller's *choice* (PETScKrylovSolver / MUMPSSolver / GinkgoDistributedSolver),
              built with scheme='fv' (the two-point cell Laplacian, consistent with the
              collocated correction and MPI-ready). If None, one is built from `backend`.
    implicit_momentum : if True, run the "true" PISO -- the momentum equation is
              assembled and solved implicitly (a_P/H split + Rhie-Chow face flux),
              so large `dt` stays stable (implicit convection+diffusion). If False
              (default) the simple explicit-momentum projection is used.
    mom_predictor : (implicit_momentum only) solve the momentum predictor with the
              old pressure gradient before the pressure correctors (icoFoam's
              momentumPredictor). Cheap to disable for pure pressure-driven steps.
    momentum : (implicit_momentum only) a manapy LinearSolver for the (non-symmetric)
              momentum system, built with scheme='fv', with_mtx=True, reuse_mtx=False.
              If None a PETSc BiCGStab is created.
    """
    self.u = u; self.v = v; self.P = P
    self.domain = dom = u.domain
    if u.dim != 2:
      raise NotImplementedError("IncompressibleSolver is wired for 2D")
    self.nu = float(nu); self.rho = float(rho); self.cfl = float(cfl)
    self.ncorr = int(ncorr)                            # PISO-style pressure correctors
    self.implicit_momentum = bool(implicit_momentum)
    self.mom_predictor = bool(mom_predictor)
    # interFoam-model parameters (all reduce to the plain single-corrector path at their
    # defaults). n_outer = PIMPLE outer correctors (re-solve alpha+momentum+pressure with
    # the latest flux); n_alpha_sub = alpha sub-cycles per step; ddt_corr = transient
    # Rhie-Chow face-flux correction (interFoam's fvc::ddtCorr).
    self.n_outer = int(n_outer)
    self.n_alpha_sub = int(n_alpha_sub)
    self.ddt_corr = bool(ddt_corr)
    # conv_order=2: deferred limited linear-upwind momentum convection (implicit part
    # stays first-order upwind; the bounded flux difference goes to the RHS). Zero at
    # rest -> the hydrostatic well-balance is untouched. Implicit-momentum path only.
    self.conv_order = int(conv_order)
    if self.conv_order not in (1, 2):
      raise ValueError("conv_order must be 1 or 2")
    # deferred non-orthogonal pressure correctors (single-phase). 0 = the current
    # orthogonal-only projection (exact on quads). >0 restores consistency on
    # non-orthogonal meshes (triangles) via the over-relaxed correction.
    self.n_nonorth = int(n_nonorth)
    self.backend = str(backend).lower()
    if self.backend not in ('mumps', 'petsc'):
      raise ValueError("backend must be 'mumps' or 'petsc'")
    # two-phase (VOF): rho(alpha)=alpha*rho1+(1-alpha)*rho2, mu likewise. alpha is
    # transported each step by the conservative flux. Requires implicit_momentum.
    self.two_phase = alpha is not None
    self.gravity = (float(gravity[0]), float(gravity[1]))
    if self.two_phase:
      if not self.implicit_momentum:
        raise ValueError("two-phase (alpha) requires implicit_momentum=True")
      self.alpha = alpha
      self.rho1 = float(rho1); self.rho2 = float(rho2)
      self.mu1 = float(mu1); self.mu2 = float(mu2)
      self.cAlpha = float(cAlpha); self.sigma = float(sigma)

    self.cellid = np.asarray(dom.faces.cellid, dtype=np.int64)
    self.halofid = np.asarray(dom.faces.halofid, dtype=np.int64)
    self.fname = np.asarray(dom.faces.name, dtype=np.int64)
    self.normal = np.ascontiguousarray(np.asarray(dom.faces.normal)[:, :2])
    self.vol = np.asarray(dom.cells.volume)
    self.nc = dom.nbcells; self.nf = len(self.cellid)
    self.nh = int(getattr(dom, "nbhalos", 0))
    self._uh = np.zeros(self.nh); self._vh = np.zeros(self.nh)
    codes = {k: dom.BCs[k][1] for k in dom.BCs}

    # reuse manapy's FV face coefficient (fv_coeff = |Sf|^2/|Sf.d|); it is exactly the
    # coefficient the scheme='fv' pressure Laplacian assembles, so the divergence,
    # the Poisson and the correction share one operator.
    self.af = np.asarray(dom.faces.fv_coeff)
    # non-orthogonal correction geometry: the over-relaxed decomposition gives
    #   Sf . grad(P)_f = af_geom (P_R - P_P) + fv_corr . interp(grad P)_f
    # fv_corr = Sf - af_geom * d  (precomputed per face); fv_wl = linear face weight.
    self.fv_corrx = np.asarray(dom.faces.fv_corrx)
    self.fv_corry = np.asarray(dom.faces.fv_corry)
    self.fv_wl = np.asarray(dom.faces._fv_weight_left)
    self._nof = np.zeros(self.nf)
    # cell-gradient buffers for the deferred momentum viscous non-ortho correction
    self._gnux = np.zeros(self.nc); self._gnuy = np.zeros(self.nc)
    self._gnvx = np.zeros(self.nc); self._gnvy = np.zeros(self.nc)
    # halo-exchanged gradients for the non-ortho correction on partition faces (MPI)
    self._gxh = np.zeros(self.nh); self._gyh = np.zeros(self.nh)
    self._gnuxh = np.zeros(self.nh); self._gnuyh = np.zeros(self.nh)
    self._gnvxh = np.zeros(self.nh); self._gnvyh = np.zeros(self.nh)
    self.uw = np.zeros(self.nf); self.vw = np.zeros(self.nf)
    for name, val in (u_bc or {}).items():
      self.uw[self.fname == codes[name]] = val
    for name, val in (v_bc or {}).items():
      self.vw[self.fname == codes[name]] = val
    self._is_int = self.fname == 0
    self._is_hal = self.fname == 10
    self._bnd = ~self._is_int
    self._pb = self._bnd & ~self._is_hal          # physical (non-partition) boundary faces

    # pressure Poisson: reuse manapy's two-point FV Laplacian (scheme='fv') through a
    # distributed linear solver -- backend is the caller's choice ('mumps' by default).
    if poisson is None:
      poisson = self._default_solver(P, with_mtx=False)
    self.L = poisson

    self._face_flux, self._mom_rhs, self._gg_grad = get_kernels()
    self._phi = np.zeros(self.nf)
    self._du = np.zeros(self.nc); self._dv = np.zeros(self.nc)
    self._gx = np.zeros(self.nc); self._gy = np.zeros(self.nc)
    self._psign = 1.0                                  # pressure-Poisson RHS sign
    self.dt = 0.0

    if self.implicit_momentum:
      self._setup_piso(dom, momentum)

  def _setup_piso(self, dom, momentum):
    from manapy.solvers.incompressible.fvm_utils_compute import get_piso_kernels
    from manapy.core.Variable import Variable
    (self._mom_assemble, self._hbya, self._dcoeff, self._corr_flux,
     self._face_avg, self._plap, self._ho_corr) = get_piso_kernels()

    # global 0-based indices + halo columns for the momentum matrix triplets
    self.loctoglob = np.asarray(dom.cells.loctoglob, dtype=np.int64)
    self.halosext = np.ascontiguousarray(np.asarray(dom.halos.halosext, dtype=np.int64))
    # geometric two-point coefficient (a snapshot: the pressure solve overwrites
    # dom.faces.fv_coeff with the variable D_f = a_f * interp(V/(rho a_P)) each step).
    self.af_geom = np.array(dom.faces.fv_coeff, dtype=np.float64)

    self.aP = np.zeros(self.nc); self.bsu = np.zeros(self.nc); self.bsv = np.zeros(self.nc)
    self.Hu = np.zeros(self.nc); self.Hv = np.zeros(self.nc); self.rAU = np.zeros(self.nc)
    self.Df = np.zeros(self.nf); self._phiH = np.zeros(self.nf); self._phinew = np.zeros(self.nf)
    self._phiHH = np.zeros(self.nf)                     # HbyA face flux without body force
    self._massflux = np.zeros(self.nf)                  # convecting mass flux (rhoPhi / rho*phi)
    self._alphaPhi = np.zeros(self.nf)                  # bounded alpha face flux (step-effective)
    self._phi_n = np.zeros(self.nf)                     # flux at time level n (ddtCorr)
    self._ddtc = np.zeros(self.nf)                      # transient Rhie-Chow face correction
    self._Huh = np.zeros(self.nh); self._Hvh = np.zeros(self.nh); self._rAUh = np.zeros(self.nh)
    self._un = np.zeros(self.nc); self._vn = np.zeros(self.nc)
    self._rhon = np.zeros(self.nc)                      # density at time level n
    self._cold_p = True                                 # hydrostatic p init pending

    if self.conv_order == 2:
      cc = np.asarray(dom.cells.center)
      fc = np.asarray(dom.faces.center)
      self._ccx = np.ascontiguousarray(cc[:, 0]); self._ccy = np.ascontiguousarray(cc[:, 1])
      self._fcx = np.ascontiguousarray(fc[:, 0]); self._fcy = np.ascontiguousarray(fc[:, 1])
      self._hcx = np.zeros(self.nh); self._hcy = np.zeros(self.nh)
      if self.nh:                                       # halo-cell centres (frozen geometry)
        dom.halo_comm.exchange(self._ccx.copy(), recv_buffer=self._hcx)
        dom.halo_comm.exchange(self._ccy.copy(), recv_buffer=self._hcy)
      self._gux = np.zeros(self.nc); self._guy = np.zeros(self.nc)
      self._gvx = np.zeros(self.nc); self._gvy = np.zeros(self.nc)
      self._guxh = np.zeros(self.nh); self._guyh = np.zeros(self.nh)
      self._gvxh = np.zeros(self.nh); self._gvyh = np.zeros(self.nh)
      self._sou = np.zeros(self.nc); self._sov = np.zeros(self.nc)
    mmax = self.nc + 2 * self.nf                        # diag block + <=2 off-diag/face
    self._mrow = np.zeros(mmax, dtype=np.int64)
    self._mcol = np.zeros(mmax, dtype=np.int64)
    self._mdata = np.zeros(mmax, dtype=np.float64)

    # per-cell/face material properties (density, dynamic viscosity) + body force.
    # Single-phase: constants (rho_c=rho, rhof=rho, muf=rho*nu) -> reduces exactly to
    # the scalar path. Two-phase: rebuilt from alpha each step.
    self.rho_c = np.full(self.nc, self.rho)
    self.mu_c = np.full(self.nc, self.rho * self.nu)
    self.rhof = np.full(self.nf, self.rho)
    self.muf = np.full(self.nf, self.rho * self.nu)
    self._rhoc_h = np.zeros(self.nh); self._muc_h = np.zeros(self.nh)
    self.gsu = np.zeros(self.nc); self.gsv = np.zeros(self.nc)
    gx, gy = self.gravity
    self.gsu[:] = self.vol * self.rho_c * gx; self.gsv[:] = self.vol * self.rho_c * gy

    # the pressure Laplacian coefficient changes every step -> no matrix reuse.
    self.L.reuse_mtx = False

    # momentum linear solver (non-symmetric: implicit convection). Same matrix for
    # u and v; solved into a scratch var then copied out.
    self._msol = Variable(domain=dom)
    if momentum is None:
      momentum = self._default_solver(self._msol, with_mtx=True)
    self.M = momentum

    if self.two_phase:
      from manapy.solvers.incompressible.vof import VOFAdvection
      from manapy.solvers.incompressible.vof_compute import get_vof_st_kernels
      # Two-phase pressure reference: PURE-NEUMANN Laplacian pinned at ONE global cell
      # (interFoam pRefCell), assembled by _plap_assemble_2d and fed through the
      # with_mtx path -- it REPLACES the BC-built operator (the P Variable's Dirichlet
      # wall, if any, becomes inactive). A whole Dirichlet wall over-determines the
      # wall rows -> O(1) div residual in the reference band -> the alpha transport is
      # unbounded there and the clip destroys phase volume (measured: bubble +85%).
      # Walls stay exactly closed, div(phi)=0 in every cell, hydrostatic rest exact.
      self._pin = 0                                     # global reference cell
      loc = np.where(self.loctoglob == self._pin)[0]
      self._pin_loc = int(loc[0]) if len(loc) else -1   # local index on the owning rank
      self._pdiag = np.zeros(self.nc)
      pmax = self.nc + 2 * self.nf
      self._prow = np.zeros(pmax, dtype=np.int64)
      self._pcol = np.zeros(pmax, dtype=np.int64)
      self._pdata = np.zeros(pmax, dtype=np.float64)
      if self.backend == 'mumps':
        from manapy.solvers.ls import MUMPSSolver
        self.L = MUMPSSolver(domain=dom, var=self.P, scheme='fv', with_mtx=True,
                             reuse_mtx=False, reuse_ij=True, memory_relaxation=200)
      else:
        from manapy.solvers.ls import PETScKrylovSolver
        self.L = PETScKrylovSolver(domain=dom, var=self.P, scheme='fv', with_mtx=True,
                                   reuse_mtx=False, method="cg", precond="gamg",
                                   eps_a=1e-12, eps_r=1e-10)
      self.vof = VOFAdvection(self.alpha, cAlpha=self.cAlpha)
      (self._gg_div, self._st_flux, self._smooth, self._buoy,
       self._reconstruct) = get_vof_st_kernels()
      self.nsmooth = 2                                  # curvature-normal smoothing passes
      # inverse metric (sum_f S_f S_f^T)^-1 per cell for fvc::reconstruct (owner gets all
      # faces, neighbour the interior ones -- matches the reconstruct accumulation).
      nx = self.normal[:, 0]; ny = self.normal[:, 1]; ci = self.cellid
      Mxx = np.zeros(self.nc); Mxy = np.zeros(self.nc); Myy = np.zeros(self.nc)
      np.add.at(Mxx, ci[:, 0], nx * nx); np.add.at(Mxy, ci[:, 0], nx * ny); np.add.at(Myy, ci[:, 0], ny * ny)
      ii = self._is_int
      np.add.at(Mxx, ci[ii, 1], nx[ii] * nx[ii]); np.add.at(Mxy, ci[ii, 1], nx[ii] * ny[ii]); np.add.at(Myy, ci[ii, 1], ny[ii] * ny[ii])
      det = Mxx * Myy - Mxy * Mxy + 1e-30
      self._ixx = Myy / det; self._ixy = -Mxy / det; self._iyy = Mxx / det
      self._psi = np.zeros(self.nf); self._rx = np.zeros(self.nc); self._ry = np.zeros(self.nc)
      self._agx = np.zeros(self.nc); self._agy = np.zeros(self.nc)
      self._nx = np.zeros(self.nc); self._ny = np.zeros(self.nc)
      self._sx = np.zeros(self.nc); self._sy = np.zeros(self.nc)
      self._nxh = np.zeros(self.nh); self._nyh = np.zeros(self.nh)
      self._kappa = np.zeros(self.nc); self._kappah = np.zeros(self.nh)
      self._phist = np.zeros(self.nf)
      self._alpha_n = np.zeros(self.nc)                 # alpha at time level n (PIMPLE re-solve)
      # p_rgh buoyancy: g.x at cell centres and face centres (frozen geometry)
      gx, gy = self.gravity
      cc = np.asarray(dom.cells.center)
      fc = np.asarray(dom.faces.center)
      self._ghc = gx * cc[:, 0] + gy * cc[:, 1]
      self._ghf = gx * fc[:, 0] + gy * fc[:, 1]
      self._grx = np.zeros(self.nc); self._gry = np.zeros(self.nc)
      self._has_gravity = (gx != 0.0 or gy != 0.0)
      self._has_body = self._has_gravity or self.sigma > 0.0
      self._update_properties()

  def _body_forces(self, dt):
    """Two-phase: prepare the interface curvature for the well-balanced surface-tension
    face flux. The body forces (buoyancy + surface tension) are added ONLY to the face
    flux phiHbyA (`_body_face_flux`) and reconstructed to the cell velocity -- there is
    NO cell body force in H (that would divide a large force by the tiny a_P of the
    light phase and blow up at high density ratio). So gsu/gsv stay zero."""
    self.gsu[:] = 0.0; self.gsv[:] = 0.0
    if self.sigma <= 0.0:
      return
    self.alpha.update_halo_value()
    self._gg_grad(self.alpha.cell, self.alpha.halo, self.normal, self.cellid,
                  self.halofid, self.fname, self.vol, self._agx, self._agy)
    gmag = np.sqrt(self._agx ** 2 + self._agy ** 2) + 1e-30
    self._nx[:] = self._agx / gmag; self._ny[:] = self._agy / gmag
    # smooth the normal field a few passes (much less noisy discrete curvature)
    for _ in range(self.nsmooth):
      if self.nh:
        self.domain.halo_comm.exchange(np.ascontiguousarray(self._nx), recv_buffer=self._nxh)
        self.domain.halo_comm.exchange(np.ascontiguousarray(self._ny), recv_buffer=self._nyh)
      self._smooth(self._nx, self._ny, self._nxh, self._nyh, self.cellid, self.halofid,
                   self.fname, self._sx, self._sy)
      self._nx, self._sx = self._sx, self._nx; self._ny, self._sy = self._sy, self._ny
    nmag = np.sqrt(self._nx ** 2 + self._ny ** 2) + 1e-30   # re-normalise the unit normal
    self._nx /= nmag; self._ny /= nmag
    if self.nh:
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._nx), recv_buffer=self._nxh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._ny), recv_buffer=self._nyh)
    self._gg_div(self._nx, self._ny, self._nxh, self._nyh, self.normal, self.cellid,
                 self.halofid, self.fname, self.vol, self._kappa)
    self._kappa *= -1.0                                # kappa = -div(n_hat)
    if self.nh:
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._kappa), recv_buffer=self._kappah)
    # surface tension enters ONLY the face flux (phist, see _body_face_flux) and is
    # reconstructed to the cell velocity -- no cell body force here (that would double
    # it and blow up the light phase at high density ratio).

  def _body_face_flux(self):
    """Well-balanced body-force face flux added to phiHbyA (needs D_f): the surface
    tension flux D_f sigma K_f snGrad(alpha) and the p_rgh buoyancy -D_f (g.x)_f
    snGrad(rho). Balanced by the p_rgh solve -> no spurious flux at equilibrium."""
    self._phist[:] = 0.0
    if self.sigma > 0.0:
      self._st_flux(self.alpha.cell, self.alpha.halo, self._kappa, self._kappah,
                    self.sigma, self.Df, self.cellid, self.halofid, self.fname, self._phist)
    if self._has_gravity:
      self._buoy(self.rho_c, self._rhoc_h, self._ghf, self.Df, self.cellid,
                 self.halofid, self.fname, self._phist)

  def _update_properties(self):
    """Two-phase: rebuild rho/mu (cell + face) from alpha. The body force (buoyancy +
    surface tension) is rebuilt separately by `_body_forces` at each step start."""
    a = self.alpha.cell
    self.rho_c[:] = self.rho2 + (self.rho1 - self.rho2) * a
    self.mu_c[:] = self.mu2 + (self.mu1 - self.mu2) * a
    if self.nh:
      self.domain.halo_comm.exchange(np.ascontiguousarray(self.rho_c), recv_buffer=self._rhoc_h)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self.mu_c), recv_buffer=self._muc_h)
    self._face_avg(self.rho_c, self._rhoc_h, self.cellid, self.halofid, self.fname, self.rhof)
    self._face_avg(self.mu_c, self._muc_h, self.cellid, self.halofid, self.fname, self.muf)

  def _default_solver(self, var, with_mtx):
    """Build a default pressure/momentum solver of the chosen `backend`. MUMPS is a
    direct COO-native solver (no per-entry assembly loop); PETSc is a Krylov method
    (CG+GAMG for the SPD pressure, BiCGStab+Jacobi for the non-symmetric momentum)."""
    dom = self.domain
    # momentum matrix (with_mtx) changes every step -> no reuse; the pressure matrix
    # is constant in the explicit path (reuse) and rebuilt each step in PISO
    # (_setup_piso flips self.L.reuse_mtx to False).
    reuse = not with_mtx
    if self.backend == 'mumps':
      from manapy.solvers.ls import MUMPSSolver
      # reuse_ij=True everywhere: the sparsity pattern is the mesh connectivity
      # (constant), so the MUMPS symbolic analysis is registered once and only the
      # numeric factorisation is redone when the values change. (The historical
      # with_mtx+reuse_ij failure was a dangling-pointer bug in the registration --
      # mumps4py keeps no python reference -- fixed in MUMPSSolver.presolve.)
      # Two-phase matrices (large density jump) need more MUMPS workspace (ICNTL(14)).
      return MUMPSSolver(domain=dom, var=var, scheme='fv', with_mtx=with_mtx,
                         reuse_mtx=reuse, reuse_ij=True,
                         memory_relaxation=200 if self.two_phase else 20)
    from manapy.solvers.ls import PETScKrylovSolver
    if with_mtx:                                        # momentum: non-symmetric
      return PETScKrylovSolver(domain=dom, var=var, scheme='fv', with_mtx=True,
                               reuse_mtx=False, method="bcgs", precond="jacobi",
                               eps_a=1e-14, eps_r=1e-12)
    return PETScKrylovSolver(domain=dom, var=var, scheme='fv', reuse_mtx=True,
                             method="cg", precond="gamg", eps_a=1e-12, eps_r=1e-10)

  def stepper(self):
    # dt must be identical on every rank (it scales the global V/dt matrix), so the
    # velocity max and the min cell size are reduced across ranks, not taken locally.
    umax = max(np.max(np.abs(self.u.cell)), np.max(np.abs(self.v.cell)), 1e-12)
    hmin2 = self.vol.min()
    if self.nh or self.domain.comm.Get_size() > 1:
      comm = self.domain.comm
      umax = comm.allreduce(float(umax), op=MPI.MAX)
      hmin2 = comm.allreduce(float(hmin2), op=MPI.MIN)
    h = np.sqrt(hmin2)
    dt_c = self.cfl * h / umax
    dt_d = self.cfl * h * h / (4.0 * self.nu) if self.nu > 0 else 1e30
    self.dt = min(dt_c, dt_d)
    return self.dt

  def _divergence(self, u, v):
    self._face_flux(u, v, self.uw, self.vw, self.normal, self.cellid, self.fname, self._phi)
    d = np.zeros(self.nc)
    np.add.at(d, self.cellid[self._is_int, 0], -self._phi[self._is_int])
    np.add.at(d, self.cellid[self._is_int, 1], self._phi[self._is_int])
    np.add.at(d, self.cellid[self._bnd, 0], -self._phi[self._bnd])
    return d

  def _cell_divergence(self, u, v):
    """Per-cell velocity divergence (1/vol) sum_f phi_f from the face fluxes. Exchanges
    the velocity to the halo so partition faces use the neighbour-rank value."""
    if self.nh:
      self.domain.halo_comm.exchange(np.ascontiguousarray(u), recv_buffer=self._uh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(v), recv_buffer=self._vh)
    self._face_flux(u, v, self._uh, self._vh, self.uw, self.vw, self.normal,
                    self.cellid, self.halofid, self.fname, self._phi)
    d = np.zeros(self.nc)
    np.add.at(d, self.cellid[self._is_int, 0], self._phi[self._is_int])
    np.add.at(d, self.cellid[self._is_int, 1], -self._phi[self._is_int])
    np.add.at(d, self.cellid[self._bnd, 0], self._phi[self._bnd])
    return d / self.vol

  def _psolve(self, rhs):
    """Pressure solve into P.cell. Two-phase (pinned pure-Neumann operator): the
    reference-cell row is the identity, so its rhs entry must be 0 (p_ref = 0)."""
    if self.two_phase and self._pin_loc >= 0:
      rhs[self._pin_loc] = 0.0
    self.L(rhs=rhs)

  def _mom_divergence(self, phi):
    """Cell divergence (1/vol) sum_f phi_f from a given face-flux array."""
    d = np.zeros(self.nc)
    np.add.at(d, self.cellid[self._is_int, 0], phi[self._is_int])
    np.add.at(d, self.cellid[self._is_int, 1], -phi[self._is_int])
    np.add.at(d, self.cellid[self._bnd, 0], phi[self._bnd])
    return d / self.vol

  def _nonortho_flux(self):
    """Deferred non-orthogonal pressure face flux:
        nof_f = interp(rAU)_f * ( fv_corr . interp(grad P)_f )
    i.e. the (Sf - af_geom*d).grad part of the face pressure gradient, scaled by the
    pressure coefficient. Added to the orthogonal D_f (P_R-P_P) it recovers the FULL
    face gradient Df/af_geom * Sf.grad(P)_f, consistent on non-orthogonal meshes.
    Interior + partition(halo) faces; physical Neumann boundaries carry no pressure flux
    (the Rhie-Chow flux keeps phiHbyA there). Caller must have refreshed P.halo."""
    self._gg_grad(self.P.cell, self.P.halo, self.normal, self.cellid, self.halofid,
                  self.fname, self.vol, self._gx, self._gy)
    nof = self._nof
    nof[:] = 0.0
    ii = self._is_int
    cL = self.cellid[ii, 0]; cR = self.cellid[ii, 1]
    wl = self.fv_wl[ii]; wr = 1.0 - wl
    gfx = wl * self._gx[cL] + wr * self._gx[cR]
    gfy = wl * self._gy[cL] + wr * self._gy[cR]
    corr = self.fv_corrx[ii] * gfx + self.fv_corry[ii] * gfy
    nof[ii] = (self.Df[ii] / self.af_geom[ii]) * corr
    if self.nh:
      # partition faces: interpolate with the neighbour-rank gradient (halo-exchanged)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._gx), recv_buffer=self._gxh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._gy), recv_buffer=self._gyh)
      hh = self._is_hal
      cLh = self.cellid[hh, 0]; nb = self.halofid[hh]
      wlh = self.fv_wl[hh]; wrh = 1.0 - wlh
      gfxh = wlh * self._gx[cLh] + wrh * self._gxh[nb]
      gfyh = wlh * self._gy[cLh] + wrh * self._gyh[nb]
      corrh = self.fv_corrx[hh] * gfxh + self.fv_corry[hh] * gfyh
      nof[hh] = (self.Df[hh] / self.af_geom[hh]) * corrh
    return nof

  def _mom_nonortho_source(self):
    """Deferred non-orthogonal VISCOUS correction for the momentum. The face viscous
    flux is muf*(af_geom (U_R-U_P) + fv_corr . interp(grad U)_f); the matrix carries the
    orthogonal af_geom part, so the fv_corr part is added (extensive, per cell) to the
    momentum source bsu/bsv. Uses the current (lagged) cell velocity. Interior + halo faces."""
    self.u.update_halo_value(); self.v.update_halo_value()
    self._gg_grad(self.u.cell, self.u.halo, self.normal, self.cellid, self.halofid,
                  self.fname, self.vol, self._gnux, self._gnuy)
    self._gg_grad(self.v.cell, self.v.halo, self.normal, self.cellid, self.halofid,
                  self.fname, self.vol, self._gnvx, self._gnvy)
    ii = self._is_int
    cL = self.cellid[ii, 0]; cR = self.cellid[ii, 1]
    wl = self.fv_wl[ii]; wr = 1.0 - wl
    cx = self.fv_corrx[ii]; cy = self.fv_corry[ii]; mf = self.muf[ii]
    gux = wl * self._gnux[cL] + wr * self._gnux[cR]
    guy = wl * self._gnuy[cL] + wr * self._gnuy[cR]
    gvx = wl * self._gnvx[cL] + wr * self._gnvx[cR]
    gvy = wl * self._gnvy[cL] + wr * self._gnvy[cR]
    fu = mf * (cx * gux + cy * guy)
    fv = mf * (cx * gvx + cy * gvy)
    np.add.at(self.bsu, cL, fu); np.add.at(self.bsu, cR, -fu)
    np.add.at(self.bsv, cL, fv); np.add.at(self.bsv, cR, -fv)
    if self.nh:
      # partition faces: interpolate with the neighbour-rank gradient; add to the owner
      # cell only (the neighbour rank adds its own equal-and-opposite contribution).
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._gnux), recv_buffer=self._gnuxh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._gnuy), recv_buffer=self._gnuyh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._gnvx), recv_buffer=self._gnvxh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._gnvy), recv_buffer=self._gnvyh)
      hh = self._is_hal
      cLh = self.cellid[hh, 0]; nb = self.halofid[hh]
      wlh = self.fv_wl[hh]; wrh = 1.0 - wlh
      cxh = self.fv_corrx[hh]; cyh = self.fv_corry[hh]; mfh = self.muf[hh]
      guxh = wlh * self._gnux[cLh] + wrh * self._gnuxh[nb]
      guyh = wlh * self._gnuy[cLh] + wrh * self._gnuyh[nb]
      gvxh = wlh * self._gnvx[cLh] + wrh * self._gnvxh[nb]
      gvyh = wlh * self._gnvy[cLh] + wrh * self._gnvyh[nb]
      np.add.at(self.bsu, cLh, mfh * (cxh * guxh + cyh * guyh))
      np.add.at(self.bsv, cLh, mfh * (cxh * gvxh + cyh * gvyh))
    # physical (Dirichlet-velocity) wall faces: the wall viscous flux nu*a_f*(uw-u_L)
    # is orthogonal too; add the non-ortho part with the OWNER-cell gradient (no
    # neighbour). Without this the boundary layer stays non-ortho-inconsistent -> the
    # ~1.5% edge offset seen on triangles (but not on orthogonal quads).
    pb = self._pb
    if pb.any():
      cLb = self.cellid[pb, 0]
      cxb = self.fv_corrx[pb]; cyb = self.fv_corry[pb]; mfb = self.muf[pb]
      np.add.at(self.bsu, cLb, mfb * (cxb * self._gnux[cLb] + cyb * self._gnuy[cLb]))
      np.add.at(self.bsv, cLb, mfb * (cxb * self._gnvx[cLb] + cyb * self._gnvy[cLb]))

  def _advance_alpha(self, dt):
    """Transport alpha (optionally sub-cycled n_alpha_sub times) by the current flux
    self._phi, accumulate the step-effective bounded alpha face flux self._alphaPhi, and
    refresh rho/mu. self._alphaPhi feeds the consistent mass flux rhoPhi."""
    ns = self.n_alpha_sub
    subdt = dt / ns
    self._alphaPhi[:] = 0.0
    for _ in range(ns):
      self.vof.step(self._phi, subdt)                  # mutates alpha, sets vof.alphaPhi
      self._alphaPhi += self.vof.alphaPhi
    if ns > 1:
      self._alphaPhi /= ns                             # time-averaged flux over the step
    self._update_properties()

  def _build_massflux(self):
    """Convecting mass flux for the momentum (Rudman consistency). Two-phase: rhoPhi =
    alphaPhi*(rho1-rho2) + phi*rho2 -- the SAME mass flux implied by the alpha transport.
    Single-phase: rho_f*phi (= rho*phi), so the momentum reduces EXACTLY to icoFoam."""
    if self.two_phase:
      self._massflux[:] = self._alphaPhi * (self.rho1 - self.rho2) + self._phi * self.rho2
    else:
      self._massflux[:] = self.rhof * self._phi

  def _ddt_corr(self, dt):
    """Transient Rhie-Chow face correction (interFoam fvc::ddtCorr): keeps the face flux
    consistent with the cell velocity across the ddt term -> suppresses pressure-velocity
    decoupling at high density ratio. ddtc_f = rAUf_f (phi^n_f - interp(u^n).S_f)/dt;
    zero at physical boundaries (flux prescribed there). Frozen across the correctors."""
    if self.nh:
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._un), recv_buffer=self._uh)
      self.domain.halo_comm.exchange(np.ascontiguousarray(self._vn), recv_buffer=self._vh)
    self._face_flux(self._un, self._vn, self._uh, self._vh, self.uw, self.vw,
                    self.normal, self.cellid, self.halofid, self.fname, self._ddtc)
    rAUf = self.Df / self.af_geom
    resid = self._phi_n - self._ddtc                   # phi^n - interp(u^n).Sf
    coeff = 1.0 - np.minimum(np.abs(resid) / (np.abs(self._phi_n) + 1e-30), 1.0)
    self._ddtc[:] = coeff * rAUf * resid / dt          # interFoam ddtCouplingCoeff
    self._ddtc[self._bnd] = 0.0

  def step_piso(self, dt=None):
    """interFoam-style two-phase PISO/PIMPLE step. Per step, after capturing the
    time-level-n {u, v, alpha, phi}, run the PIMPLE outer loop (n_outer):
      1. alpha-first: transport alpha by the latest flux -> bounded alphaPhi -> the
         CONSISTENT mass flux rhoPhi; refresh rho/mu; body forces (buoyancy + CSF);
      2. momentum matrix M (convected by rhoPhi) + optional predictor;
      3. PISO pressure correctors (ncorr): Rhie-Chow flux (+ body-force flux + ddtCorr),
         pressure solve, flux correction, cell-velocity reconstruct.
    Single-phase reduces EXACTLY to the validated icoFoam PISO (n_outer=1, no alpha)."""
    dt = self.stepper() if dt is None else dt
    u, v = self.u.cell, self.v.cell
    dom = self.domain
    self._un[:] = u; self._vn[:] = v                   # time-level-n velocity (ddt source)
    self._phi_n[:] = self._phi                         # time-level-n flux (ddtCorr)
    self._rhon[:] = self.rho_c                         # time-level-n density (ddt source)
    if self.two_phase:
      self._alpha_n[:] = self.alpha.cell               # time-level-n alpha (PIMPLE re-solve)

    for _outer in range(self.n_outer):
      # --- 1. alpha-first transport + consistent mass flux ---
      if self.two_phase:
        self.alpha.cell[:] = self._alpha_n             # re-solve alpha from level n each outer
        self._advance_alpha(dt)
        self._body_forces(dt)
      self._build_massflux()

      # --- 2. momentum matrix M (same for u,v), convected by the mass flux ---
      n = self._mom_assemble(self._massflux, self.af_geom, self.uw, self.vw, self.cellid,
                             self.halofid, self.fname, self.loctoglob, self.halosext,
                             self.vol, dt, self.rho_c, self.muf, self.aP,
                             self.bsu, self.bsv, self._mrow, self._mcol, self._mdata)
      self.M.set_matrix(self._mrow[:n].copy(), self._mcol[:n].copy(), self._mdata[:n].copy())
      self.M.reuse_mtx = False
      if self.conv_order == 2:
        # deferred limited linear-upwind convection: bounded explicit flux difference
        # from the current velocity, added to the source (flows into the predictor RHS
        # and into H). Frozen over the PISO correctors, refreshed each PIMPLE outer.
        self.u.update_halo_value(); self.v.update_halo_value()
        self._gg_grad(u, self.u.halo, self.normal, self.cellid, self.halofid,
                      self.fname, self.vol, self._gux, self._guy)
        self._gg_grad(v, self.v.halo, self.normal, self.cellid, self.halofid,
                      self.fname, self.vol, self._gvx, self._gvy)
        if self.nh:
          dom.halo_comm.exchange(np.ascontiguousarray(self._gux), recv_buffer=self._guxh)
          dom.halo_comm.exchange(np.ascontiguousarray(self._guy), recv_buffer=self._guyh)
          dom.halo_comm.exchange(np.ascontiguousarray(self._gvx), recv_buffer=self._gvxh)
          dom.halo_comm.exchange(np.ascontiguousarray(self._gvy), recv_buffer=self._gvyh)
        self._ho_corr(u, v, self.u.halo, self.v.halo, self._gux, self._guy,
                      self._gvx, self._gvy, self._guxh, self._guyh, self._gvxh,
                      self._gvyh, self._massflux, self._ccx, self._ccy, self._fcx,
                      self._fcy, self._hcx, self._hcy, self.cellid, self.halofid,
                      self.fname, self._sou, self._sov)
        self.bsu += self._sou; self.bsv += self._sov
      if self.n_nonorth > 0 and not self.two_phase:
        self._mom_nonortho_source()          # deferred non-ortho viscous source (bsu/bsv)
      # rAU = V/a_P and its halo; face pressure coeff D_f = a_f interp(rAU); body/ddt
      # fluxes. Computed BEFORE the predictor: the two-phase predictor source needs the
      # body-force face flux (it only depends on aP, final after the assembly).
      self.rAU[:] = self.vol / self.aP
      if self.nh:
        dom.halo_comm.exchange(np.ascontiguousarray(self.rAU), recv_buffer=self._rAUh)
      self._dcoeff(self.rAU, self._rAUh, self.af_geom, self.cellid, self.halofid,
                   self.fname, self.Df)
      dom.faces.fv_coeff[:] = self.Df                  # variable-coeff pressure Laplacian
      if self.two_phase:
        # pinned pure-Neumann pressure operator from the fresh D_f (with_mtx path)
        n = self._plap(self.Df, self.cellid, self.halofid, self.fname, self.loctoglob,
                       self.halosext, self.vol, self._pin, self._pdiag,
                       self._prow, self._pcol, self._pdata)
        self.L.set_matrix(self._prow[:n].copy(), self._pcol[:n].copy(),
                          self._pdata[:n].copy())
      if self.two_phase and self._has_body:
        self._body_face_flux()
      if self.ddt_corr:
        self._ddt_corr(dt)
      self.L.reuse_mtx = False                         # a_P/D_f frozen over the correctors

      if self._cold_p and self.two_phase and self._has_body and self.mom_predictor:
        # cold-start hydrostatic p_rgh: the stored p (=0) does not balance the body-force
        # face flux yet, so the first predictor would get a spurious O(g h) kick that then
        # only decays viscously. One Poisson solve against phist alone gives the balanced
        # p_rgh (exact at rest: phiHbyA = 0 there).
        self._psolve(self._psign * self._mom_divergence(self._phist))
        self.L.reuse_mtx = True                        # same D_f matrix for the correctors
        self._cold_p = False

      if self.mom_predictor:                           # M u* = b0 - V grad(p^n) (+ body)
        self.P.update_halo_value()
        # conservative ddt: rho^n u^n V/dt source vs rho^{n+1} V/dt diagonal, so with
        # div(rhoPhi) = -(rho^{n+1}-rho^n)V/dt (alpha-transport mass consistency) a
        # UNIFORM velocity is an exact solution. Using rho^{n+1} on both sides puts a
        # spurious force -(rho^{n+1}-rho^n)V/dt u on every interface-crossed cell --
        # water-scale force on air inertia (measured: air |u| -> 100+ m/s, dam-break
        # front 2x too fast, at ANY dt).
        b0u = self._rhon * (self.vol / dt) * self._un + self.bsu + self.gsu
        b0v = self._rhon * (self.vol / dt) * self._vn + self.bsv + self.gsv
        if self.two_phase and self._has_body:
          # interFoam UEqn source: V reconstruct((phi_body/rAUf - snGrad(p_rgh)) |Sf|),
          # i.e. the RAW face imbalance a_f (phist_f/D_f - dp_f). At the discrete
          # hydrostatic balance phist_f = D_f dp_f on EVERY face, so this source
          # vanishes identically and the predictor preserves u = 0 exactly. A cell
          # Green-Gauss grad(p) does not: it reintroduces the collocated truncation
          # error the face balance just removed, ratio-amplified by rAU in the light
          # phase (measured: parasitic |u| grows 1e-13 -> 8e-3 in 40 steps with it).
          ii = self._is_int; Pc = self.P.cell; ci = self.cellid
          self._psi[:] = 0.0
          self._psi[ii] = self.af_geom[ii] * (self._phist[ii] / self.Df[ii]
                                              - (Pc[ci[ii, 1]] - Pc[ci[ii, 0]]))
          if self.nh:
            hh = self._is_hal
            self._psi[hh] = self.af_geom[hh] * (self._phist[hh] / self.Df[hh]
                                                - (self.P.halo[self.halofid[hh]] - Pc[ci[hh, 0]]))
          self._reconstruct(self._psi, self.normal, self.cellid, self.halofid, self.fname,
                            self._ixx, self._ixy, self._iyy, self._rx, self._ry)
          gpx = -self._rx; gpy = -self._ry
        else:
          self._gg_grad(self.P.cell, self.P.halo, self.normal, self.cellid, self.halofid,
                        self.fname, self.vol, self._gx, self._gy)
          gpx = self._gx; gpy = self._gy
        self.M(rhs=b0u - self.vol * gpx); u[:] = self._msol.cell
        self.M.reuse_mtx = True                        # same matrix for v -> reuse
        self.M(rhs=b0v - self.vol * gpy); v[:] = self._msol.cell

      # --- 3. PISO pressure correctors ---
      for _ in range(self.ncorr):
        self.u.update_halo_value(); self.v.update_halo_value()
        self._hbya(self._un, self._vn, u, v, self.u.halo, self.v.halo, self._massflux,
                   self.af_geom, self._rhon, self.muf, self.aP, self.bsu,
                   self.bsv, self.cellid, self.halofid, self.fname, self.vol, dt,
                   self.gsu, self.gsv, self.Hu, self.Hv)
        if self.nh:
          dom.halo_comm.exchange(np.ascontiguousarray(self.Hu), recv_buffer=self._Huh)
          dom.halo_comm.exchange(np.ascontiguousarray(self.Hv), recv_buffer=self._Hvh)
        # phiHbyA = interp(HbyA).S_f  (+ ddtCorr + body-force flux); walls prescribed
        self._face_flux(self.Hu, self.Hv, self._Huh, self._Hvh, self.uw, self.vw,
                        self.normal, self.cellid, self.halofid, self.fname, self._phiHH)
        self._phiH[:] = self._phiHH
        if self.ddt_corr:
          self._phiH += self._ddtc
        if self.two_phase and self._has_body:
          self._phiH += self._phist
        div_phiH = self._mom_divergence(self._phiH)
        self._psolve(self._psign * div_phiH)
        self.L.reuse_mtx = True
        self.P.update_halo_value(); self.P.update_ghost_value()
        # deferred non-orthogonal correctors: keep the compact orthogonal operator in
        # the matrix, add the explicit non-ortho flux divergence to the RHS and re-solve.
        # Restores consistency on non-orthogonal (triangle) meshes without breaking the
        # div-grad projection. Single-phase only for now.
        do_nonortho = self.n_nonorth > 0 and not self.two_phase
        for _no in range(self.n_nonorth if do_nonortho else 0):
          self._psolve(self._psign * (div_phiH - self._mom_divergence(self._nonortho_flux())))
          self.P.update_halo_value(); self.P.update_ghost_value()
        self._corr_flux(self._phiH, self.Df, self.P.cell, self.P.halo, self.cellid,
                        self.halofid, self.fname, self._phinew)
        if do_nonortho:
          self._phinew -= self._nonortho_flux()   # subtract the non-orthogonal pressure flux
        if self.two_phase:
          # cell velocity from the balanced flux: u = HbyA + rAU reconstruct((phig -
          # pEqn.flux())/rAUf); the ddtCorr is a flux-only term (excluded here).
          self._psi[:] = (self._phinew - self._phiHH - self._ddtc) * self.af_geom / self.Df
          self._reconstruct(self._psi, self.normal, self.cellid, self.halofid, self.fname,
                            self._ixx, self._ixy, self._iyy, self._rx, self._ry)
          u[:] = self.Hu + self.rAU * self._rx
          v[:] = self.Hv + self.rAU * self._ry
        else:
          self._gg_grad(self.P.cell, self.P.halo, self.normal, self.cellid, self.halofid,
                        self.fname, self.vol, self._gx, self._gy)
          u[:] = self.Hu - self.rAU * self._gx
          v[:] = self.Hv - self.rAU * self._gy
        self._phi, self._phinew = self._phinew, self._phi

    dom.faces.fv_coeff[:] = self.af_geom               # restore the geometric coeff
    return dt

  def step(self, dt=None):
    if self.implicit_momentum:
      return self.step_piso(dt)
    dt = self.stepper() if dt is None else dt
    u, v = self.u.cell, self.v.cell
    ff, mom, gg = self._face_flux, self._mom_rhs, self._gg_grad

    # 1-2. predictor (momentum convection by the div-free face flux + diffusion)
    self.u.update_halo_value(); self.v.update_halo_value()
    ff(u, v, self.u.halo, self.v.halo, self.uw, self.vw, self.normal,
       self.cellid, self.halofid, self.fname, self._phi)
    mom(u, v, self.u.halo, self.v.halo, self._phi, self.af, self.uw, self.vw,
        self.cellid, self.halofid, self.fname, self.vol, self.nu, self._du, self._dv)
    uc = u + dt * self._du; vc = v + dt * self._dv

    # 3-4. PISO-style correctors: each solves a pressure (correction) from the current
    #      velocity divergence and applies the gradient correction. Iterating drives
    #      the residual collocated cell divergence down; the pressures accumulate.
    Ptot = np.zeros(self.nc)
    for _ in range(self.ncorr):
      div = self._cell_divergence(uc, vc)
      self.L(rhs=self._psign * (self.rho / dt) * div)  # solves into P.cell
      self.P.update_halo_value(); self.P.update_ghost_value()
      Ptot += self.P.cell
      gg(self.P.cell, self.P.halo, self.normal, self.cellid, self.halofid, self.fname,
         self.vol, self._gx, self._gy)
      uc = uc - (dt / self.rho) * self._gx
      vc = vc - (dt / self.rho) * self._gy
    u[:] = uc; v[:] = vc; self.P.cell[:] = Ptot
    return dt

  def divergence_norm(self):
    """L2 norm of the discrete velocity divergence (from the face fluxes)."""
    d = self._cell_divergence(self.u.cell, self.v.cell)
    return float(np.sqrt(np.sum(d * d * self.vol)))

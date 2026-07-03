#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverse-osmosis membrane-channel solver (transient, with fouling).

This is a finite-volume solver for a cross-flow RO feed channel built on top of
manapy's unstructured-mesh machinery.  It reuses :class:`AdvectionDiffusionSolver`
to transport the dissolved-salt concentration in the channel and adds the two
pieces of physics that distinguish reverse osmosis from a passive scalar:

  1. a **permeable membrane wall** (one boundary patch, ``bottom`` by default)
     where water is sucked out at the local permeation velocity ``Jw`` and salt
     is mostly rejected.  Concentration polarisation (a salt build-up at the
     wall) emerges from the balance between this wall suction and back-diffusion
     -- it is *resolved* by the mesh, not prescribed by a Sherwood correlation;

  2. a **fouling layer** that grows on the membrane over time, adding hydraulic
     resistance ``R_f`` in series with the membrane and therefore making the
     permeate flux decline -- the transient effect we actually care about.

Geometry / boundary convention (see ``meshes/ro_channel.geo``)::

    upper (3): impermeable top wall / symmetry        y = H
       in (1) --> feed                concentrate --> out (2)    x: 0 -> L
    bottom (4): MEMBRANE wall                            y = 0

Coupling with the advection-diffusion kernel
--------------------------------------------
The convective kernel already removes ``Jw*area*c_cell`` through any out-flowing
boundary face.  A clean membrane only lets ``c_p`` through, so each step we add a
source term that re-injects the *rejected* salt ``(c_w - c_p)``; the net wall
removal is then exactly the physical salt flux ``Jw*c_p``.  Water removal is
carried by a divergence-free channel velocity field (cross-flow that decays
along ``x`` as water permeates, plus wall-ward suction), which concentrates the
salt near the membrane.

The membrane coupling (wall source term, fouling update) operates on host
arrays, so this solver targets the **CPU backend** (the manapy default); the
underlying transport is fully MPI-parallel.
"""
from mpi4py import MPI
import numpy as np

from manapy.core.Variable import Variable
from manapy.solvers.advecdiff.system import AdvectionDiffusionSolver
from manapy.solvers.ro import membrane as mb


class ReverseOsmosisSolver:
    """Transient cross-flow reverse-osmosis solver with membrane fouling.

    Parameters
    ----------
    conc : Variable
        Dissolved-salt concentration field [kg/m3].  Must be created with a
        boundary dict whose membrane patch (``bottom`` by default) is ``neumann``
        and whose inlet (``in``) is ``dirichlet`` set to ``feed_conc``.
    vel : tuple(Variable, Variable)
        The (u, v) velocity-component variables; they are *driven* by the solver
        (the channel flow field is rebuilt every step from the permeation rate).
    feed_conc : float
        Inlet salt concentration [kg/m3] (used as a reference scale and for the
        feed-side osmotic pressure).
    U0 : float
        Mean cross-flow velocity at the inlet [m/s].
    D : float
        Salt diffusivity [m2/s] (effective; includes unresolved mixing).
    A_w, B_s : float
        Membrane water permeability [m/(s.Pa)] and salt permeability [m/s].
    dP : float
        Trans-membrane pressure [Pa].
    osmotic_coeff : float
        van 't Hoff coefficient phi so that pi = phi * c [Pa per kg/m3].
    sigma : float
        Reflection coefficient (1.0 = perfectly rejecting).
    mu : float
        Permeate dynamic viscosity [Pa.s].
    fouling : bool
        Enable the transient fouling layer.
    fouling_coeff : float
        Fouling rate [1/s]: fraction of the clean membrane resistance added per
        second at nominal (feed) conditions.  NOTE this is an *accelerated*
        demonstration value -- real fouling acts over hours; here it is tuned so
        the flux decline is visible within a short simulation.
    Rf_max_factor : float
        Cap on R_f as a multiple of the clean resistance R_m.
    membrane, top, inlet, outlet : str
        Patch names (manapy locations) for the four channel sides.
    cfl, order, scheme : passed through to AdvectionDiffusionSolver.
    """

    _LOC2FACES = {"in": "infaces", "out": "outfaces",
                  "upper": "upperfaces", "bottom": "bottomfaces"}

    def __init__(self, conc, vel, *,
                 feed_conc,
                 U0=0.1,
                 D=5.0e-8,
                 A_w=2.0e-12,
                 B_s=5.0e-8,
                 dP=5.5e6,
                 osmotic_coeff=8.0e4,
                 sigma=1.0,
                 mu=1.0e-3,
                 fouling=True,
                 fouling_coeff=0.3,
                 Rf_max_factor=10.0,
                 flow_model="crossflow",
                 velocity_profile="plug",
                 membrane="bottom", top="upper", inlet="in", outlet="out",
                 cfl=0.4, order=1, scheme="upwind"):

        if flow_model not in ("crossflow", "uniform_suction"):
            raise ValueError("flow_model must be 'crossflow' or 'uniform_suction'")
        if velocity_profile not in ("plug", "parabolic"):
            raise ValueError("velocity_profile must be 'plug' or 'parabolic'")
        self.flow_model = flow_model
        # 'plug'      : axial velocity uniform across the channel (slip at the wall);
        #               crude, gives a Sh ~ x^-1/2 concentration boundary layer.
        # 'parabolic' : no-slip parabolic axial profile (u=0 at the membrane, symmetry
        #               at the top); gives the physical shear-driven Sh ~ x^-1/3
        #               (Leveque) layer that the WaterTAP Sherwood correlation encodes.
        #               Recommended for concentration-polarization fidelity.
        self.velocity_profile = velocity_profile
        self.c = conc
        self.u, self.v = vel[0], vel[1]
        self.domain = conc.domain
        self.comm = self.domain.halo_comm.graph_comm

        # --- membrane / physical parameters -------------------------------
        self.feed_conc = float(feed_conc)
        self.U0 = float(U0)
        self.D = float(D)
        self.A_w = float(A_w)
        self.B_s = float(B_s)
        self.dP = float(dP)
        self.osmotic_coeff = float(osmotic_coeff)
        self.sigma = float(sigma)
        self.mu = float(mu)
        self.R_m = mb.membrane_resistance(A_w, mu)

        self.fouling = bool(fouling)
        self.fouling_coeff = float(fouling_coeff)
        self.Rf_max = Rf_max_factor * self.R_m

        # --- geometry of the four channel patches -------------------------
        d = self.domain
        self.mface = getattr(d, self._LOC2FACES[membrane])   # membrane faces
        self.mcell = d.faces.cellid[self.mface, 0]           # adjacent cells
        self.marea = d.faces.mesure[self.mface]              # face length [m]
        self.mvol = d.cells.volume[self.mcell]               # cell volume [m2]

        # channel height H and a per-cell / per-face copy of the coordinates
        upper = getattr(d, self._LOC2FACES[top])
        self.H = float(d.faces.center[upper, 1].mean()) if len(upper) else \
            float(d.cells.center[:, 1].max())
        self.xc = d.cells.center[:, 0].copy()
        self.yc = d.cells.center[:, 1].copy()
        self.xf = d.faces.center[:, 0].copy()
        self.yf = d.faces.center[:, 1].copy()

        # --- fouling / coupling state -------------------------------------
        nmf = len(self.mface)
        self.R_f = np.zeros(nmf, dtype=float)     # fouling resistance per face
        self.cp = np.zeros(nmf, dtype=float)      # permeate conc (lagged)

        # nominal (clean, feed) permeation velocity -- reference scale
        pi_feed = mb.osmotic_pressure(self.feed_conc, self.osmotic_coeff)
        self.Jw_nom = max(self.A_w * (self.dP - self.sigma * pi_feed), 1e-12)

        # --- underlying advection-diffusion transport ---------------------
        self.transport = AdvectionDiffusionSolver(
            conc, vel=(self.u, self.v),
            Dxx=self.D, Dyy=self.D, Dzz=0.0,
            order=order, cfl=cfl, scheme=scheme)

        self.time = 0.0
        self.niter = 0

    # ----------------------------------------------------------------------
    def _membrane_state(self):
        """Wall concentration, permeation velocity and permeate conc per face."""
        c_w = np.maximum(self.c.cell[self.mcell], 0.0)
        pi_w = mb.osmotic_pressure(c_w, self.osmotic_coeff)
        pi_p = mb.osmotic_pressure(self.cp, self.osmotic_coeff)
        Jw = mb.water_flux(self.dP, pi_w, pi_p, self.mu, self.R_m, self.R_f,
                           self.sigma)
        cp = mb.permeate_conc(c_w, Jw, self.B_s)
        return c_w, Jw, cp

    def _set_velocity_field(self, Jw):
        """Rebuild the (divergence-free) channel velocity field.

        The axial velocity decays along the channel as water permeates and the
        wall-ward suction vanishes at the top wall; the membrane faces carry the
        *exact* per-face permeation velocity so the wall water balance is
        consistent.  ``velocity_profile`` selects a plug (slip) or a no-slip
        parabolic axial profile (see ``__init__``).
        """
        Jw_ref = float(Jw.mean()) if len(Jw) else 0.0

        if self.flow_model == "uniform_suction":
            # Dead-end / film-theory idealisation: uniform wall-ward velocity
            # (divergence-free), no cross-flow. Used to verify the resolved
            # concentration polarisation against the analytic film solution.
            self.u.cell[:] = 0.0
            self.v.cell[:] = -Jw_ref
            self.u.face[:] = 0.0
            self.v.face[:] = -Jw_ref
            self.v.face[self.mface] = -Jw
            return

        H = self.H
        if self.velocity_profile == "parabolic":
            # Divergence-free NO-SLIP field: parabolic axial profile g(eta)=2*eta-eta^2
            # (u=0 at the membrane eta=0, symmetry du/dy=0 at the top eta=1) scaled so
            # the cross-sectional mean is U0-(Jw/H)x, with the matching wall-normal
            # component v = Jw*(-1 + 1.5*eta^2 - 0.5*eta^3)  (=-Jw at the wall, 0 at the
            # top).  div(u,v)=0 by construction.  Reproduces the shear concentration
            # boundary layer (Sh ~ x^-1/3) of the Sherwood/Leveque regime.
            def _field(x, y):
                eta = y / H
                a = 1.5 * (self.U0 - (Jw_ref / H) * x)
                return a * (2.0 * eta - eta ** 2), \
                    Jw_ref * (-1.0 + 1.5 * eta ** 2 - 0.5 * eta ** 3)
            self.u.cell[:], self.v.cell[:] = _field(self.xc, self.yc)
            self.u.face[:], self.v.face[:] = _field(self.xf, self.yf)
            self.v.face[self.mface] = -Jw
            return

        slope = Jw_ref / H
        self.u.cell[:] = self.U0 - slope * self.xc
        self.v.cell[:] = -Jw_ref * (1.0 - self.yc / H)

        self.u.face[:] = self.U0 - slope * self.xf
        self.v.face[:] = -Jw_ref * (1.0 - self.yf / H)
        # exact suction on the membrane faces (outward normal points to -y there)
        self.v.face[self.mface] = -Jw

    def _apply_membrane_source(self, c_w, Jw, cp):
        """Re-inject the rejected salt so the net wall removal equals Jw*c_p."""
        self.c.source[:] = 0.0
        rejected = Jw * self.marea * (c_w - cp) / self.mvol
        np.add.at(self.c.source, self.mcell, rejected)

    def _update_fouling(self, dt, c_w, Jw):
        if not self.fouling:
            return
        # dR_f/dt = k * R_m * (Jw*c_w)/(Jw_nom*c_feed)  [1/s scaled]
        rate = self.fouling_coeff * self.R_m * (Jw * c_w) / \
            (self.Jw_nom * self.feed_conc)
        self.R_f = np.minimum(self.R_f + dt * rate, self.Rf_max)

    # ----------------------------------------------------------------------
    def step(self):
        """Advance the coupled system by one explicit time step. Returns dt."""
        c_w, Jw, cp = self._membrane_state()
        self.cp = cp

        self._set_velocity_field(Jw)
        self._apply_membrane_source(c_w, Jw, cp)

        dt = self.transport.stepper()
        self.transport.compute_fluxes()
        self.transport.compute_new_val()

        self._update_fouling(dt, c_w, Jw)

        self.time += dt
        self.niter += 1
        return dt

    def diagnostics(self):
        """Global, MPI-reduced performance indicators for the current state."""
        c_w, Jw, cp = self._membrane_state()
        # per-unit-depth permeate volumetric flow [m2/s] and feed flow
        Qp_local = float(np.sum(Jw * self.marea))
        area_local = float(np.sum(self.marea))
        cw_sum = float(np.sum(c_w * self.marea))
        cp_sum = float(np.sum(cp * self.marea))
        Rf_sum = float(np.sum(self.R_f * self.marea))

        Qp = self.comm.allreduce(Qp_local, op=MPI.SUM)
        area = self.comm.allreduce(area_local, op=MPI.SUM)
        cw = self.comm.allreduce(cw_sum, op=MPI.SUM)
        cp_m = self.comm.allreduce(cp_sum, op=MPI.SUM)
        Rf = self.comm.allreduce(Rf_sum, op=MPI.SUM)

        Qfeed = self.U0 * self.H
        area = area if area > 0 else 1.0
        return {
            "time": self.time,
            "Qp": Qp,                          # permeate flow  [m2/s]
            "flux_mean": Qp / area,            # mean permeation velocity [m/s]
            "flux_LMH": Qp / area * 1000.0 * 3600.0,   # [L/m2/h]
            "recovery": Qp / Qfeed if Qfeed > 0 else 0.0,
            "cw_mean": cw / area,              # mean wall concentration
            "cp_mean": cp_m / area,            # mean permeate concentration
            "Rf_mean": Rf / area,              # mean fouling resistance
            "Rf_over_Rm": (Rf / area) / self.R_m,
        }

    def run(self, *, nsteps=None, tfinal=None, history_every=1, verbose=False):
        """Time-march the solver.

        Provide either ``nsteps`` or ``tfinal``.  Returns a dict of 1-D arrays
        (one sample every ``history_every`` steps) tracking the key indicators.
        """
        if nsteps is None and tfinal is None:
            raise ValueError("give either nsteps or tfinal")

        keys = ("time", "flux_mean", "flux_LMH", "recovery",
                "cw_mean", "cp_mean", "Rf_mean", "Rf_over_Rm")
        hist = {k: [] for k in keys}
        rank0 = (MPI.COMM_WORLD.Get_rank() == 0)

        def record():
            d = self.diagnostics()
            for k in keys:
                hist[k].append(d[k])
            return d

        record()  # initial state
        i = 0
        while True:
            self.step()
            i += 1
            if (i % history_every) == 0:
                d = record()
                if verbose and rank0:
                    print(f"  it={i:5d}  t={d['time']:.4g}s  "
                          f"flux={d['flux_LMH']:6.2f} LMH  "
                          f"rec={d['recovery']*100:5.2f}%  "
                          f"cw={d['cw_mean']:6.2f}  cp={d['cp_mean']:.3f}  "
                          f"Rf/Rm={d['Rf_over_Rm']:.3f}")
            if nsteps is not None and i >= nsteps:
                break
            if tfinal is not None and self.time >= tfinal:
                break

        return {k: np.asarray(v) for k, v in hist.items()}

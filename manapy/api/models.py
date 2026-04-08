import os

import numpy as np
from mpi4py import MPI

from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.solvers.advec import AdvectionSolver
from manapy.solvers.diffusion import DiffusionSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_variable(val, mesh, name=""):
    """
    Accept a Variable, float, or callable → return a Variable.

      - Variable  → used as-is
      - float     → constant on cell and face arrays
      - callable  → f(x, y, z) evaluated on cell and face centres
    """
    if isinstance(val, Variable):
        val.interpolate_celltoface()
        return val

    domain = mesh.domain
    var = Variable(domain=domain, name=name)

    if callable(val):
        c = domain.cells.center
        var.cell[:] = val(c[:, 0], c[:, 1], c[:, 2])
        f = domain.faces.center
        var.face[:] = val(f[:, 0], f[:, 1], f[:, 2])
    else:
        var.cell[:] = float(val)
        var.face[:] = float(val)

    return var


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def _save(domain, variables, names, dt, time, niter, miter, mode, fmt):
    vals = []
    for var in variables:
        if mode == "node":
            var.update_halo_value()
            var.update_ghost_value()
            var.interpolate_celltonode()
            vals.append(var.node)
        else:
            vals.append(var.cell)

    if mode == "node":
        domain.save_on_node_multi(dt, time, niter, miter,
                                  variables=names, values=vals,
                                  file_format=fmt)
    else:
        domain.save_on_cell_multi(dt, time, niter, miter,
                                  variables=names, values=vals,
                                  file_format=fmt)


# ---------------------------------------------------------------------------
# AdvectionModel
# ---------------------------------------------------------------------------

class AdvectionModel:
    """
    Explicit advection solver.AdvectionModel

    Parameters
    ----------
    var      : Variable — transported scalar
    mesh     : Mesh     — the mesh (needed to resolve float/callable velocity)
    velocity : tuple    — (u, v) or (u, v, w)
               Each component: Variable, float, or callable f(x,y,z).
    cfl      : float    — CFL number (default 0.8)
    order    : int      — 1 (default) or 2
    output   : list of (Variable, name), optional
               Variables to save. Default: [(var, var._name or "phi")]

    Example
    -------
    ne  = Variable(domain=domain, name="ne")
    u   = Variable(domain=domain, name="u")
    v   = Variable(domain=domain, name="v")
    ...
    model = AdvectionModel(ne, mesh, velocity=(u, v), cfl=0.8)
    model.run(T=0.25, output_every=50)
    """

    def __init__(self, var, mesh, velocity, cfl=0.8, order=1, output=None):
        self._mesh = mesh
        self._var  = var

        vel_vars = tuple(
            _to_variable(v, mesh, name=f"vel{i}")
            for i, v in enumerate(velocity)
        )

        conf = Struct(order=order, cfl=cfl)
        self._solver = AdvectionSolver(var, vel=vel_vars, conf=conf)

        if output is None:
            self._out_vars  = [var]
            self._out_names = [var._name or "phi"]
        else:
            self._out_vars  = [v for v, _ in output]
            self._out_names = [n for _, n in output]

    def run(self, T, output_every=50, output_dir=".",
            output_mode="node", format="vtu"):
        """
        Run the time loop until t = T.

        Parameters
        ----------
        T            : float — final simulation time
        output_every : int   — save every N iterations (default 50)
        output_dir   : str   — directory where results/ will be created
        output_mode  : "cell" or "node"
        format       : "vtu" (default) or "vtk"
        """
        domain = self._mesh.domain
        solver = self._solver
        time, niter, miter = 0.0, 1, 0

        if RANK == 0:
            print(f"[AdvectionModel] T={T}  output_every={output_every}")

        original_cwd = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        try:
            while time < T:
                d_t   = solver.stepper()
                time += d_t
                solver.compute_fluxes()
                solver.compute_new_val()

                if niter == 1 or niter % output_every == 0:
                    _save(domain, self._out_vars, self._out_names,
                          d_t, time, niter, miter, output_mode, format)
                    miter += 1
                niter += 1
        finally:
            os.chdir(original_cwd)

        if RANK == 0:
            print(f"[AdvectionModel] Done — {niter-1} iters, t={time:.6f}")


# ---------------------------------------------------------------------------
# DiffusionModel
# ---------------------------------------------------------------------------

class DiffusionModel:
    """
    Explicit advection-diffusion solver.

    Parameters
    ----------
    var      : Variable
    mesh     : Mesh
    velocity : tuple — (u, v) or (u, v, w); Variable, float, or callable
    Dxx, Dyy, Dzz : float — diffusion coefficients
    cfl      : float
    order    : int (default 2)
    output   : list of (Variable, name), optional

    Example
    -------
    model = DiffusionModel(phi, mesh, velocity=(u, v), Dxx=0.1)
    model.run(T=0.25)
    """

    def __init__(self, var, mesh, velocity, Dxx=0.1, Dyy=0., Dzz=0.,
                 cfl=0.8, order=2, output=None):
        self._mesh = mesh
        self._var  = var

        vel_vars = tuple(
            _to_variable(v, mesh, name=f"vel{i}")
            for i, v in enumerate(velocity)
        )

        conf = Struct(Dxx=Dxx, Dyy=Dyy, Dzz=Dzz, order=order, cfl=cfl)
        self._solver = DiffusionSolver(var, vel=vel_vars, conf=conf)

        if output is None:
            self._out_vars  = [var]
            self._out_names = [var._name or "phi"]
        else:
            self._out_vars  = [v for v, _ in output]
            self._out_names = [n for _, n in output]

    def run(self, T, output_every=50, output_dir=".",
            output_mode="cell", format="vtu"):
        domain = self._mesh.domain
        solver = self._solver
        time, niter, miter = 0.0, 1, 0

        if RANK == 0:
            print(f"[DiffusionModel] T={T}  output_every={output_every}")

        original_cwd = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
        try:
            while time < T:
                d_t   = solver.stepper()
                time += d_t
                solver.compute_fluxes()
                solver.compute_new_val()

                if niter == 1 or niter % output_every == 0:
                    _save(domain, self._out_vars, self._out_names,
                          d_t, time, niter, miter, output_mode, format)
                    miter += 1
                niter += 1
        finally:
            os.chdir(original_cwd)

        if RANK == 0:
            print(f"[DiffusionModel] Done — {niter-1} iters, t={time:.6f}")

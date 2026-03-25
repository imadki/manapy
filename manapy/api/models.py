import os

import numpy as np
from mpi4py import MPI

from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.solvers.advec import AdvectionSolver
from manapy.solvers.diffusion import DiffusionSolver

from manapy.api.field import Field

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_velocity_var(val, mesh, name=""):
    """
    Convert a velocity component to a Variable.

    Accepts:
      - Field       → use its Variable directly
      - Variable    → use as-is
      - float       → constant field (face and cell both set)
      - callable    → f(x, y, z) evaluated on cell and face centres
    """
    domain = mesh.domain

    if isinstance(val, Field):
        return val.var

    if isinstance(val, Variable):
        return val

    # scalar or callable
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
# Shared output helper
# ---------------------------------------------------------------------------

def _save(domain, fields, dt, time, niter, miter, mode, fmt):
    names  = [f.name or f"field{i}" for i, f in enumerate(fields)]
    values = []

    for f in fields:
        if mode == "node":
            f.var.update_halo_value()
            f.var.update_ghost_value()
            f.var.interpolate_celltonode()
            values.append(f.var.node)
        else:
            values.append(f.var.cell)

    if mode == "node":
        domain.save_on_node_multi(dt, time, niter, miter,
                                  variables=names, values=values,
                                  file_format=fmt)
    else:
        domain.save_on_cell_multi(dt, time, niter, miter,
                                  variables=names, values=values,
                                  file_format=fmt)


# ---------------------------------------------------------------------------
# AdvectionModel
# ---------------------------------------------------------------------------

class AdvectionModel:
    """
    Explicit advection solver.

    Parameters
    ----------
    field    : Field   — transported scalar
    velocity : tuple   — (u, v) or (u, v, w)
               Each component can be a Field, Variable, float, or callable.
    cfl      : float   — CFL number (default 0.8)
    order    : int     — 1 (first order) or 2 (second order, default 1)
    output   : list[Field], optional
               Fields to save at each output step.
               Defaults to [field].

    Example
    -------
    model = AdvectionModel(phi, velocity=(2.0, 0.0), cfl=0.8)
    model.run(T=0.25, output_every=50)
    """

    def __init__(self, field, velocity, cfl=0.8, order=1, output=None):
        self._mesh   = field.mesh
        self._field  = field
        self._output = output if output is not None else [field]

        vel_vars = tuple(
            _make_velocity_var(v, self._mesh, name=f"vel{i}")
            for i, v in enumerate(velocity)
        )

        conf = Struct(order=order, cfl=cfl)
        self._solver = AdvectionSolver(field.var, vel=vel_vars, conf=conf)

    def run(self, T, output_every=50, output_dir=".", output_mode="cell", format="vtu"):
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
        domain  = self._mesh.domain
        solver  = self._solver

        time  = 0.0
        niter = 1
        miter = 0

        if RANK == 0:
            print(f"[AdvectionModel] T={T}  output_every={output_every}  dir={output_dir}")

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
                    _save(domain, self._output, d_t, time, niter, miter,
                          output_mode, format)
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
    field    : Field   — transported scalar
    velocity : tuple   — (u, v) or (u, v, w)
    Dxx, Dyy, Dzz : float — diffusion coefficients (default 0.1, 0., 0.)
    cfl      : float   — CFL number (default 0.8)
    order    : int     — 1 or 2 (default 2)
    output   : list[Field], optional

    Example
    -------
    model = DiffusionModel(phi, velocity=(1.0, 0.0), Dxx=0.1)
    model.run(T=0.25, output_every=50)
    """

    def __init__(self, field, velocity, Dxx=0.1, Dyy=0., Dzz=0.,
                 cfl=0.8, order=2, output=None):
        self._mesh   = field.mesh
        self._field  = field
        self._output = output if output is not None else [field]

        vel_vars = tuple(
            _make_velocity_var(v, self._mesh, name=f"vel{i}")
            for i, v in enumerate(velocity)
        )

        conf = Struct(Dxx=Dxx, Dyy=Dyy, Dzz=Dzz, order=order, cfl=cfl)
        self._solver = DiffusionSolver(field.var, vel=vel_vars, conf=conf)

    def run(self, T, output_every=50, output_dir=".", output_mode="cell", format="vtu"):
        """
        Run the time loop until t = T.

        Parameters
        ----------
        T, output_every, output_dir, output_mode, format : same as AdvectionModel.run
        """
        domain  = self._mesh.domain
        solver  = self._solver

        time  = 0.0
        niter = 1
        miter = 0

        if RANK == 0:
            print(f"[DiffusionModel] T={T}  output_every={output_every}  dir={output_dir}")

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
                    _save(domain, self._output, d_t, time, niter, miter,
                          output_mode, format)
                    miter += 1

                niter += 1
        finally:
            os.chdir(original_cwd)

        if RANK == 0:
            print(f"[DiffusionModel] Done — {niter-1} iters, t={time:.6f}")

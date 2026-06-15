import numpy as np
from mpi4py import MPI

from manapy.core.Variable import Variable
from manapy.solvers.advec.system import AdvectionSolver
from manapy.solvers.advecdiff.system import AdvectionDiffusionSolver
from manapy.solvers.ls import MUMPSSolver, PETScKrylovSolver

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_velocity(val, domain, name=""):
  """Return a Variable for a velocity component given a Variable, float or callable."""
  if isinstance(val, Variable):
    val.interpolate_celltoface()
    return val
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


def _resolve_output(output, default_var):
  if output is None:
    return [default_var], [default_var.name or "phi"]
  return [v for v, _ in output], [n for _, n in output]


def _save(domain, variables, names, dt, time, niter, miter, mode):
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
    domain.save_on_node_multi(names, vals, dt, time, niter, miter)
  else:
    domain.save_on_cell_multi(names, vals, dt, time, niter, miter)


# --------------------------------------------------------------------------- #
# explicit time-stepping models
# --------------------------------------------------------------------------- #
class _ExplicitModel:
  """Common explicit time loop. Subclasses set self._solver in __init__."""

  _label = "Model"

  def __init__(self, var, mesh, output=None):
    self._mesh = mesh
    self._var = var
    self._solver = None
    self._out_vars, self._out_names = _resolve_output(output, var)

  def run(self, T, output_every=50, output_mode="node"):
    domain = self._mesh.domain
    solver = self._solver
    time, niter, miter = 0.0, 1, 0

    if RANK == 0:
      print(f"[{self._label}] T={T}  output_every={output_every}")

    while time < T:
      d_t = solver.stepper()
      time += d_t
      solver.compute_fluxes()
      solver.compute_new_val()
      if niter == 1 or niter % output_every == 0:
        _save(domain, self._out_vars, self._out_names, d_t, time, niter, miter, output_mode)
        miter += 1
      niter += 1

    if RANK == 0:
      print(f"[{self._label}] done — {niter - 1} iters, t={time:.6f}")


class AdvectionModel(_ExplicitModel):
  """Explicit advection of `var` by `velocity` (= (u, v[, w]); each component a
  Variable, float, or callable f(x,y,z))."""

  _label = "AdvectionModel"

  def __init__(self, var, mesh, velocity, cfl=0.8, order=1, scheme="upwind", output=None):
    super().__init__(var, mesh, output)
    vel = tuple(_to_velocity(v, mesh.domain, name=f"vel{i}")
                for i, v in enumerate(velocity))
    self._solver = AdvectionSolver(var, vel=vel, order=order, cfl=cfl, scheme=scheme)


class DiffusionModel(_ExplicitModel):
  """Explicit advection-diffusion of `var`."""

  _label = "DiffusionModel"

  def __init__(self, var, mesh, velocity, Dxx=0.1, Dyy=0.0, Dzz=0.0,
               cfl=0.8, order=2, scheme="upwind", output=None):
    super().__init__(var, mesh, output)
    vel = tuple(_to_velocity(v, mesh.domain, name=f"vel{i}")
                for i, v in enumerate(velocity))
    self._solver = AdvectionDiffusionSolver(var, vel=vel, Dxx=Dxx, Dyy=Dyy, Dzz=Dzz,
                                            order=order, cfl=cfl, scheme=scheme)


# --------------------------------------------------------------------------- #
# implicit (linear-system) models
# --------------------------------------------------------------------------- #
def _make_ls(solver, domain, var, ls_kwargs):
  if solver == "mumps":
    return MUMPSSolver(domain=domain, var=var, **ls_kwargs)
  if solver == "petsc":
    return PETScKrylovSolver(domain=domain, var=var, **ls_kwargs)
  raise ValueError(f"unknown solver '{solver}' (use 'mumps' or 'petsc')")


class PoissonModel:
  """Steady linear (diffusion / Laplace / Poisson) solve for `var`.

  Example
  -------
  P = mesh.field("P", bc={"in": ("dirichlet", 20), "out": ("dirichlet", 0),
                          "upper": ("dirichlet", 0), "bottom": ("dirichlet", 0)})
  PoissonModel(P, mesh).solve()        # P.cell now holds the solution
  """

  def __init__(self, var, mesh, solver="mumps", scheme="diamond", **ls_kwargs):
    self._mesh = mesh
    self._var = var
    ls_kwargs.setdefault("scheme", scheme)
    self._L = _make_ls(solver, mesh.domain, var, ls_kwargs)

  @property
  def solver(self):
    return self._L

  def solve(self):
    self._L()
    return self._var

  def save(self, name=None):
    name = name or self._var.name or "P"
    _save(self._mesh.domain, [self._var], [name], 0, 0, 0, 0, "node")


class DarcyModel:
  """Pressure-driven (Darcy) flow: solve the pressure `var`, derive the velocity
  from its gradient, and transport an optional `tracer` with it.

  Mirrors examples/2D/darcy2d.py. The velocity components (u, v[, w]) are created
  internally and available via `model.velocity`.

  Example
  -------
  P  = mesh.field("P", bc={...})
  ne = mesh.field("ne", init=gaussian)
  DarcyModel(P, mesh, tracer=ne).run(T=0.25, output_every=10)
  """

  def __init__(self, var, mesh, tracer=None, Dxx=0.0, Dyy=0.0, Dzz=0.0,
               order=2, cfl=0.8, flux="upwind", solver="mumps", scheme="diamond",
               output=None, **ls_kwargs):
    self._mesh = mesh
    self._var = var
    self._tracer = tracer
    domain = mesh.domain
    self._dim = domain.dim

    ls_kwargs.setdefault("scheme", scheme)
    ls_kwargs.setdefault("reuse_mtx", True)
    self._L = _make_ls(solver, domain, var, ls_kwargs)

    names = ["u", "v", "w"][:self._dim]
    self._vel = tuple(Variable(domain=domain, name=nm) for nm in names)

    self._transport = None
    if tracer is not None:
      self._transport = AdvectionDiffusionSolver(tracer, vel=self._vel,
                                                 Dxx=Dxx, Dyy=Dyy, Dzz=Dzz,
                                                 order=order, cfl=cfl, scheme=flux)

    if output is None:
      outs = [(var, var.name or "P")] + [(c, c.name) for c in self._vel]
      if tracer is not None:
        outs.append((tracer, tracer.name or "tracer"))
      output = outs
    self._out_vars, self._out_names = _resolve_output(output, var)

  @property
  def velocity(self):
    return self._vel

  @property
  def solver(self):
    return self._L

  def _update_velocity(self):
    P = self._var
    self._L()
    P.update_halo_value()
    P.update_ghost_value()
    P.interpolate_celltonode()
    self._L.compute_Sol_gradient()
    grads = (P.gradfacex, P.gradfacey, P.gradfacez)
    for comp, g in zip(self._vel, grads):
      comp.face[:] = g[:]
      comp.interpolate_facetocell()

  def run(self, T, output_every=50, output_mode="node"):
    if self._transport is None:
      raise ValueError("DarcyModel.run needs a tracer; for a steady solve use solve()")
    domain = self._mesh.domain
    time, niter, miter = 0.0, 1, 0

    if RANK == 0:
      print(f"[DarcyModel] T={T}  output_every={output_every}")

    while time < T:
      self._update_velocity()
      d_t = self._transport.stepper()
      time += d_t
      self._transport.compute_fluxes()
      self._transport.compute_new_val()
      if niter == 1 or niter % output_every == 0:
        _save(domain, self._out_vars, self._out_names, d_t, time, niter, miter, output_mode)
        miter += 1
      niter += 1

    if RANK == 0:
      print(f"[DarcyModel] done — {niter - 1} iters, t={time:.6f}")

  def solve(self):
    """Steady solve: pressure + derived velocity (no transport)."""
    self._update_velocity()
    return self._var, self._vel

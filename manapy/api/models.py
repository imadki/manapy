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

  def run(self, T, output_every=50, output_mode="node", exact=None):
    """Advance to time T.

    exact : optional callable ``f(x, y, z, t)`` returning the exact cell values.
            When given, an ``<var>_exact`` field is maintained, saved next to the
            solution in every output frame, and the final L2/L-infinity errors are
            printed and stored on ``self.l2_error`` / ``self.linf_error``.
            (Use output_mode="cell" with exact -- the reference field has no BCs.)
    """
    domain = self._mesh.domain
    solver = self._solver
    time, niter, miter = 0.0, 1, 0

    out_vars, out_names = list(self._out_vars), list(self._out_names)
    exact_var = None
    self.l2_error = self.linf_error = None
    if exact is not None:
      base = self._var.name or "u"
      exact_var = self._mesh.field(base + "_exact")
      cc = domain.cells.center

      def _refresh_exact(t):
        exact_var.cell[:] = exact(cc[:, 0], cc[:, 1], cc[:, 2], t)

      out_vars = out_vars + [exact_var]
      out_names = out_names + [base + "_exact"]

    if RANK == 0:
      print(f"[{self._label}] T={T}  output_every={output_every}"
            + ("  (+exact)" if exact is not None else ""))

    while time < T:
      d_t = solver.stepper()
      time += d_t
      solver.compute_fluxes()
      solver.compute_new_val()
      if niter == 1 or niter % output_every == 0:
        if exact_var is not None:
          _refresh_exact(time)
        _save(domain, out_vars, out_names, d_t, time, niter, miter, output_mode)
        miter += 1
      niter += 1

    if exact_var is not None:
      _refresh_exact(time)
      err = np.asarray(self._var.cell) - np.asarray(exact_var.cell)
      vol = np.asarray(domain.cells.volume)
      num = COMM.allreduce(float(np.sum(vol * err * err)), op=MPI.SUM)
      den = COMM.allreduce(float(np.sum(vol)), op=MPI.SUM)
      self.linf_error = COMM.allreduce(float(np.max(np.abs(err))), op=MPI.MAX)
      self.l2_error = (num / den) ** 0.5

    if RANK == 0:
      print(f"[{self._label}] done — {niter - 1} iters, t={time:.6f}")
      if exact_var is not None:
        print(f"[{self._label}] vs exact @ t={time:.4f} :  "
              f"L2 = {self.l2_error:.3e}   Linf = {self.linf_error:.3e}")


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


class BurgersModel(_ExplicitModel):
  """Explicit nonlinear (viscous) Burgers: u_t + div(u^2/2) = nu*lap(u).

  The field advects itself (no prescribed velocity); `nu` is the isotropic
  viscosity (nu=0 -> inviscid). Order 2 is MUSCL with a slope limiter.

  Example
  -------
  u = mesh.field("u", init=lambda x, y, z: np.where(x < 0.25, 1.0, 0.0),
                 bc={"in": ("dirichlet", 1), "out": ("dirichlet", 0),
                     "upper": "neumann", "bottom": "neumann"}, limiter="vanalbada")
  BurgersModel(u, mesh, nu=0.01, order=2).run(T=1.0)
  """

  _label = "BurgersModel"

  def __init__(self, var, mesh, nu=0.0, cfl=0.4, order=2, scheme="rusanov", output=None):
    super().__init__(var, mesh, output)
    # Local import keeps the (numba-compiling) burgers module off the api import path.
    from manapy.solvers.burgers.system import BurgersSolver
    self._solver = BurgersSolver(var, nu=nu, order=order, cfl=cfl, scheme=scheme)


class MultilayerSWModel(_ExplicitModel):
  """Variable-density multilayer shallow water for dense (brine) plumes.

  `layers` is a list of dicts ``{'h','hu','hv','s'}`` of Variables, ordered
  bottom -> top (index 0 = densest, at the bed). `rho` is the per-layer density.
  Inter-layer baroclinic coupling is carried by a per-layer effective bed; v2
  turbulent entrainment (``entrain=True``) dilutes the dense current.

  ``scheme='srnh'`` is the well-balanced subcritical default; ``scheme='hllc'`` is
  robust through the sonic point Fr=1 -- use it for transcritical plunging plumes.

  Example
  -------
  NEU = {"in": "neumann", "out": "neumann", "upper": "neumann", "bottom": "neumann"}
  layers = [{'h': mesh.field("h1", init=h1_0, bc=NEU), 'hu': mesh.field("hu1", bc=NEU),
             'hv': mesh.field("hv1", bc=NEU), 's': mesh.field("s1", init=h1_0, bc=NEU)},
            {'h': mesh.field("h2", init=h2_0, bc=NEU), 'hu': mesh.field("hu2", bc=NEU),
             'hv': mesh.field("hv2", bc=NEU), 's': mesh.field("s2", bc=NEU)}]
  MultilayerSWModel(layers, mesh, rho=[1035, 1000], Z=Z, scheme="hllc",
                    entrain=True, Mann=0.01).run(T=3.0, output_every=200, output_mode="cell")
  """

  _label = "MultilayerSWModel"

  def __init__(self, layers, mesh, rho, Z=None, grav=9.81, cfl=0.8, order=1, Mann=0.0,
               scheme="srnh", entrain=False, E0=0.075, a_par=718.0, n_par=2.4,
               rho0=None, cap_frac=0.2, output=None):
    if output is None:
      output = []
      for k, lay in enumerate(layers):
        output.append((lay['h'], f"h{k + 1}"))
        output.append((lay['s'], f"s{k + 1}"))
    super().__init__(layers[0]['h'], mesh, output)
    # Local import keeps the (numba-compiling) multilayer module off the api import path.
    from manapy.solvers.multilayer.system import MultilayerSWSolver
    self._solver = MultilayerSWSolver(layers, rho=rho, Z=Z, grav=grav, cfl=cfl, order=order,
                                      Mann=Mann, scheme=scheme, entrain=entrain, E0=E0,
                                      a_par=a_par, n_par=n_par, rho0=rho0, cap_frac=cap_frac)
    self._layers = layers


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

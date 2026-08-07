from functools import partial
from manapy.backends.ManapyArray import Device, ManapyArray
from manapy.backends.config import ManapyConfig
from manapy.compute._compute import _Compute

class _Solvers_common:
  """Device-agnostic entry points for the advection-solver kernels.

  Same contract as `VariableCompute` and `BoundaryCompute`: the compiled
  kernels take numpy arrays (CPU build) or CuPy arrays (CUDA build) and never
  see a ManapyArray:

      r  -> cpu_r  / gpu_r    read-only:  sync in if stale, other side stays valid
      rw -> cpu_rw / gpu_rw   read-write: sync in if stale, other side invalidated
      w  -> cpu_w  / gpu_w    write-only: NO transfer, other side invalidated

  `__init__` resolves the CPU/CUDA and 2D/3D kernel plus the matching accessor
  triple once and binds them with `functools.partial` (see `VariableCompute`
  for why partial rather than a lambda), so callers keep the same positional
  argument list the raw kernels take.

  The wrappers are positional-only in practice: one wrapper serves both the 2D
  and the 3D convective kernel, so keyword calls must not be used. Scalars
  (`cfl`, `dtime`, `order`, `scheme`, `dim`) are passed straight through --
  only the array arguments must be ManapyArray.

  `update_new_value` and the two `initialisation_gaussian_*` are not advec
  kernels: they live in solvers.utils and are shared by every solver, and are
  exposed here because the advection solver's setup and time loop need them.
  The Gaussian ones are CPU-only and so always run through the CPU accessors
  (see `__init__`).
  """

  def __init__(self, config: ManapyConfig, dim: int):
    self.config = config
    self.dim = dim
    self.compute = _Compute.getComputeInstance(config)

    if dim not in (2, 3):
      raise ValueError(f"dim must be 2 or 3, got {dim}")

    if config.device == Device.CUDA:
      acc = (ManapyArray.gpu_r, ManapyArray.gpu_rw, ManapyArray.gpu_w)
      k_update_new_value = self.compute.update_new_value_cuda
    else:
      acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
      k_update_new_value = self.compute.update_new_value



    self.update_new_value = partial(
      _Solvers_common.update_new_value, k_update_new_value, acc)

    # Cpu Only
    cpu_acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
    self.initialisation_gaussian_2d = partial(
      _Solvers_common.initialisation_gaussian_2d,
      self.compute.initialisation_gaussian_2d, cpu_acc)
    self.initialisation_gaussian_3d = partial(
      _Solvers_common.initialisation_gaussian_3d,
      self.compute.initialisation_gaussian_3d, cpu_acc)


  @staticmethod
  def update_new_value(kernel, acc, ne_c, rez_ne, dissip_ne, src_ne, dtime,
                       cell_volume):
    """solvers.utils.update_new_value: forward-Euler update of a cell field,
    ne_c += dtime * ((rez + dissip) / volume + src). `ne_c` is accumulated
    into, not overwritten, so it is read-write."""
    r, rw, w = acc
    kernel(
      rw(ne_c), r(rez_ne), r(dissip_ne), r(src_ne), dtime, r(cell_volume)
    )

  @staticmethod
  def initialisation_gaussian_2d(kernel, acc, ne, u, v, P, cell_center, Pinit):
    """solvers.utils.initialisation_gaussian_2d: Gaussian bump initial
    condition -- ne = Gaussian centred at (0.2, 0.2), u = v = 0 and
    P = Pinit * (0.5 - x).

    The kernel loops over every cell of `cell_center` and assigns (never
    accumulates) each output, so `ne`, `u`, `v` and `P` are all write-only.
    They must be at least as long as `cell_center` -- the loop bound comes from
    `cell_center`, not from the outputs."""
    r, rw, w = acc
    kernel(w(ne), w(u), w(v), w(P), r(cell_center), Pinit)

  @staticmethod
  def initialisation_gaussian_3d(kernel, acc, ne, u, v, w_, P, cell_center,
                                 Pinit):
    """solvers.utils.initialisation_gaussian_3d: 3D counterpart of
    initialisation_gaussian_2d -- Gaussian centred at (0.2, 0.25, 0.45),
    u = v = w = 0 and P = Pinit * (0.5 - x). Same write-only outputs, plus the
    z velocity (named `w_` here so it does not shadow the `w` accessor)."""
    r, rw, w = acc
    kernel(w(ne), w(u), w(v), w(w_), w(P), r(cell_center), Pinit)

from functools import partial
from manapy.backends.ManapyArray import Device, ManapyArray
from manapy.backends.config import ManapyConfig
from manapy.compute._compute import _Compute

class VariableCompute:
  """Device-agnostic entry points for the per-variable kernels.

  The compiled kernels take numpy arrays (CPU build) or CuPy arrays (CUDA
  build); they never see a ManapyArray. Each kernel is exposed here as a
  *static* wrapper that unwraps every argument through the ManapyArray
  interface with the intent the kernel actually has for it:

      r  -> cpu_r  / gpu_r    read-only:  sync in if stale, other side stays valid
      rw -> cpu_rw / gpu_rw   read-write: sync in if stale, other side invalidated
      w  -> cpu_w  / gpu_w    write-only: NO transfer, other side invalidated

  `w` is only legal when the kernel writes EVERY element -- it hands out an
  uninitialised buffer otherwise. Kernels looping over the full extent of their
  output (facetocell, center_to_vertex, cell_gradient, the limiters) qualify;
  kernels writing through a gather list of face ids (celltoface,
  face_gradient) touch only part of the array and must use `rw`.

  `__init__` resolves the CPU/CUDA and 2D/3D kernel plus the matching accessor
  triple once and binds them, so callers keep calling e.g.
  ``compute.cell_gradient(...)`` with the same positional argument list as
  before. All array arguments must be ManapyArray.

  The binding is done with `functools.partial`, which freezes the leading
  arguments of a function and returns a callable taking the rest. Each wrapper
  is declared as ``(kernel, acc, <kernel args...>)`` and

      self.facetocell = partial(VariableCompute.facetocell, k_facetocell, acc)

  pins the first two, so `self.facetocell` behaves as if its signature were
  ``(u_face, u_c, cell_faceid, dim)`` -- the device/dim choice is resolved once
  here instead of on every call. A closure
  (``lambda *a: VariableCompute.facetocell(k, acc, *a)``) would do the same,
  but `partial` prepends the frozen arguments in C (no extra Python frame) and
  is picklable, which a lambda is not -- and manapy runs under MPI.

  The wrappers are positional-only in practice: the 2D and 3D bindings do not
  always agree on parameter names (``face_halofid`` vs ``face_haloid``,
  ``d_periodicfaces`` vs ``d_periodicboundaryfaces``), so a single wrapper
  serves both and keyword calls must not be used.
  """

  def __init__(self, config: ManapyConfig, dim: int):
    self.config = config
    self.dim = dim
    self.compute = _Compute.getComputeInstance(config)

    if config.device == Device.CUDA:
      acc = (ManapyArray.gpu_r, ManapyArray.gpu_rw, ManapyArray.gpu_w)
      k_facetocell = self.compute.facetocell_cuda
      k_celltoface = self.compute.celltoface_cuda
      if dim == 2:
        k_interp = self.compute.center_to_vertex_2d_cuda
        k_face_gradient = self.compute.face_gradient_2d_cuda
        k_cell_gradient = self.compute.cell_gradient_2d_cuda
        k_barthlimiter = self.compute.barthlimiter_2d_cuda
        k_vanalbadalimiter = self.compute.vanalbadalimiter_2d_cuda
      else:
        k_interp = self.compute.center_to_vertex_3d_cuda
        k_face_gradient = self.compute.face_gradient_3d_cuda
        k_cell_gradient = self.compute.cell_gradient_3d_cuda
        k_barthlimiter = self.compute.barthlimiter_3d_cuda
        k_vanalbadalimiter = self.compute.vanalbadalimiter_3d_cuda
    else:
      acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
      k_facetocell = self.compute.facetocell
      k_celltoface = self.compute.celltoface
      if dim == 2:
        k_interp = self.compute.center_to_vertex_2d
        k_face_gradient = self.compute.face_gradient_2d
        k_cell_gradient = self.compute.cell_gradient_2d
        k_barthlimiter = self.compute.barthlimiter_2d
        k_vanalbadalimiter = self.compute.vanalbadalimiter_2d
      else:
        k_interp = self.compute.center_to_vertex_3d
        k_face_gradient = self.compute.face_gradient_3d
        k_cell_gradient = self.compute.cell_gradient_3d
        k_barthlimiter = self.compute.barthlimiter_3d
        k_vanalbadalimiter = self.compute.vanalbadalimiter_3d

    self.facetocell = partial(VariableCompute.facetocell, k_facetocell, acc)
    self.celltoface = partial(VariableCompute.celltoface, k_celltoface, acc)
    self.interp = partial(VariableCompute.interp, k_interp, acc)
    self.face_gradient = partial(VariableCompute.face_gradient, k_face_gradient, acc)
    self.cell_gradient = partial(VariableCompute.cell_gradient, k_cell_gradient, acc)
    # Both limiters have the same signature and the same read/write intents.
    self.barthlimiter = partial(VariableCompute.limiter, k_barthlimiter, acc)
    self.vanalbadalimiter = partial(VariableCompute.limiter, k_vanalbadalimiter, acc)

  # ------------------------------------------------------------------ kernels

  @staticmethod
  def facetocell(kernel, acc, u_face, u_c, cell_faceid, dim):
    """Face field -> cell field. `u_c` is written for every cell."""
    r, rw, w = acc
    kernel(r(u_face), w(u_c), r(cell_faceid), dim)

  @staticmethod
  def celltoface(kernel, acc, u_cell, u_face, u_ghost, u_halo, face_cellid,
                 face_halofid, d_innerfaces, d_boundaryfaces, d_halofaces):
    """Cell field -> face field. `u_face` is written only at the gathered face
    ids (inner / halo / boundary), so it is read-write, not write-only."""
    r, rw, w = acc
    kernel(
      r(u_cell), rw(u_face), r(u_ghost), r(u_halo), r(face_cellid),
      r(face_halofid), r(d_innerfaces), r(d_boundaryfaces), r(d_halofaces)
    )

  @staticmethod
  def interp(kernel, acc, w_c, w_ghost, w_halo, w_haloghost, cell_center,
             halo_centvol, node_cellid, ghost_info_flt, ghost_ext_info_flt,
             node_ghostid, node_haloghostid, node_periodicid, node_halonid,
             nodes, node_oldname, node_R_x, node_R_y, node_R_z, node_lambda_x,
             node_lambda_y, node_lambda_z, node_number, cell_shift, w_n,
             ghost_faceid):
    """center_to_vertex_2d/3d: cell field -> node field. `w_n` is written for
    every node."""
    r, rw, w = acc
    kernel(
      r(w_c), r(w_ghost), r(w_halo), r(w_haloghost), r(cell_center),
      r(halo_centvol), r(node_cellid), r(ghost_info_flt), r(ghost_ext_info_flt),
      r(node_ghostid), r(node_haloghostid), r(node_periodicid), r(node_halonid),
      r(nodes), r(node_oldname), r(node_R_x), r(node_R_y), r(node_R_z),
      r(node_lambda_x), r(node_lambda_y), r(node_lambda_z), r(node_number),
      r(cell_shift), w(w_n), r(ghost_faceid)
    )

  @staticmethod
  def cell_gradient(kernel, acc, w_c, w_ghost, w_halo, w_haloghost, cell_center,
                    cell_cellnid, ghost_info_flt, ghost_ext_info_flt,
                    cell_ghostnid, cell_haloghostnid, cell_halonid, cells,
                    cell_periodicfid, node_periodicid, node_oldname,
                    halo_centvol, cell_shift, w_x, w_y, w_z, ghost_faceid):
    """cell_gradient_2d/3d: least-squares gradient at the cell centres.
    `w_x`/`w_y`/`w_z` are written for every cell (the 2D kernel sets w_z to 0
    explicitly), so all three are write-only."""
    r, rw, w = acc
    kernel(
      r(w_c), r(w_ghost), r(w_halo), r(w_haloghost), r(cell_center),
      r(cell_cellnid), r(ghost_info_flt), r(ghost_ext_info_flt),
      r(cell_ghostnid), r(cell_haloghostnid), r(cell_halonid), r(cells),
      r(cell_periodicfid), r(node_periodicid), r(node_oldname),
      r(halo_centvol), r(cell_shift), w(w_x), w(w_y), w(w_z), r(ghost_faceid)
    )

  @staticmethod
  def face_gradient(kernel, acc, w_c, w_ghost, w_halo, w_node, face_cellid,
                    faces, face_halofid, face_air_diamond, face_normal, face_f1,
                    face_f2, face_f3, face_f4, wx_face, wy_face, wz_face,
                    d_innerfaces, d_halofaces, dirichletfaces, neumann,
                    d_periodicfaces):
    """face_gradient_2d/3d: gradient at the face midpoints. The outputs are
    written only at the gathered face ids -- and the 2D kernel does not touch
    `wz_face` at all -- so all three are read-write."""
    r, rw, w = acc
    kernel(
      r(w_c), r(w_ghost), r(w_halo), r(w_node), r(face_cellid), r(faces),
      r(face_halofid), r(face_air_diamond), r(face_normal), r(face_f1),
      r(face_f2), r(face_f3), r(face_f4), rw(wx_face), rw(wy_face),
      rw(wz_face), r(d_innerfaces), r(d_halofaces), r(dirichletfaces),
      r(neumann), r(d_periodicfaces)
    )

  @staticmethod
  def limiter(kernel, acc, w_c, w_ghost, w_halo, w_x, w_y, w_z, psi,
              face_cellid, cell_faceid, face_name, face_haloid, cell_center,
              face_center):
    """barthlimiter_2d/3d and vanalbadalimiter_2d/3d: same signature and same
    intents. `psi` is written for every cell (the 2D kernels ignore `w_z`)."""
    r, rw, w = acc
    kernel(
      r(w_c), r(w_ghost), r(w_halo), r(w_x), r(w_y), r(w_z), w(psi),
      r(face_cellid), r(cell_faceid), r(face_name), r(face_haloid),
      r(cell_center), r(face_center)
    )


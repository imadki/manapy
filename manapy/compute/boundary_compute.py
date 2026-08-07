from functools import partial
from manapy.backends.ManapyArray import Device, ManapyArray
from manapy.backends.config import ManapyConfig
from manapy.compute._compute import _Compute


class BoundaryCompute:
  def __init__(self, config: ManapyConfig, dim: int, BCtype: str):
    self.config = config
    self.dim = dim
    self.BCtype = BCtype
    self.compute = _Compute.getComputeInstance(config)

    if config.device == Device.CUDA:
      acc = (ManapyArray.gpu_r, ManapyArray.gpu_rw, ManapyArray.gpu_w)
      if BCtype == "dirichlet":
        k_ghost = self.compute.ghost_value_dirichlet_cuda
        k_haloghost = self.compute.haloghost_value_dirichlet_cuda
      elif BCtype == "neumann" or BCtype == "periodic":
        k_ghost = self.compute.ghost_value_neumann_cuda
        k_haloghost = self.compute.haloghost_value_neumann_cuda
      elif BCtype == "neumannNH":
        k_ghost = self.compute.ghost_value_neumannNH_cuda
        k_haloghost = self.compute.haloghost_value_neumannNH_cuda
      elif BCtype == "nonslip":
        k_ghost = self.compute.ghost_value_nonslip_cuda
        k_haloghost = self.compute.haloghost_value_nonslip_cuda
      elif BCtype == "slip":
        if dim == 2:
          k_ghost = self.compute.ghost_value_slip_2d_cuda
          k_haloghost = self.compute.haloghost_value_slip_2d_cuda
        else:
          k_ghost = self.compute.ghost_value_slip_3d_cuda
          k_haloghost = self.compute.haloghost_value_slip_3d_cuda
      else:
        raise ValueError(f"unknown BCtype: {BCtype}")
    else:
      acc = (ManapyArray.cpu_r, ManapyArray.cpu_rw, ManapyArray.cpu_w)
      if BCtype == "dirichlet":
        k_ghost = self.compute.ghost_value_dirichlet
        k_haloghost = self.compute.haloghost_value_dirichlet
      elif BCtype == "neumann" or BCtype == "periodic":
        k_ghost = self.compute.ghost_value_neumann
        k_haloghost = self.compute.haloghost_value_neumann
      elif BCtype == "neumannNH":
        k_ghost = self.compute.ghost_value_neumannNH
        k_haloghost = self.compute.haloghost_value_neumannNH
      elif BCtype == "nonslip":
        k_ghost = self.compute.ghost_value_nonslip
        k_haloghost = self.compute.haloghost_value_nonslip
      elif BCtype == "slip":
        if dim == 2:
          k_ghost = self.compute.ghost_value_slip_2d
          k_haloghost = self.compute.haloghost_value_slip_2d
        else:
          k_ghost = self.compute.ghost_value_slip_3d
          k_haloghost = self.compute.haloghost_value_slip_3d
      else:
        raise ValueError(f"unknown BCtype: {BCtype}")

    # The four scalar conditions share one signature, so one wrapper serves
    # them all. Slip does not: it takes 2 or 3 velocity components, so it
    # needs its own wrapper per dimension, not just its own kernel.
    if BCtype != "slip":
      w_ghost = BoundaryCompute.ghost_value
      w_haloghost = BoundaryCompute.haloghost_value
    elif dim == 2:
      w_ghost = BoundaryCompute.slip_ghost_2d
      w_haloghost = BoundaryCompute.slip_haloghost_2d
    else:
      w_ghost = BoundaryCompute.slip_ghost_3d
      w_haloghost = BoundaryCompute.slip_haloghost_3d

    self.ghost = partial(w_ghost, k_ghost, acc)
    self.haloghost = partial(w_haloghost, k_haloghost, acc)

  # ------------------------------------------------------------------ kernels

  @staticmethod
  def ghost_value(kernel, acc, value, w_ghost, face_cellid, bc_faces, cst,
                  face_dist_ortho):
    """ghost_value_dirichlet/_neumann/_neumannNH/_nonslip: set the ghost value
    behind every face of `bc_faces`.

    `value` is the prescribed per-face value for dirichlet and the cell field
    `w_c` for the other kinds -- the same slot in every signature. `cst` and
    `face_dist_ortho` are read by neumannNH only, `face_cellid` by everything
    except dirichlet; the unused ones stay in the signature for parity."""
    r, rw, w = acc
    kernel(
      r(value), rw(w_ghost), r(face_cellid), r(bc_faces), r(cst),
      r(face_dist_ortho)
    )

  @staticmethod
  def haloghost_value(kernel, acc, value, w_haloghost, node_haloghostid,
                      ghost_ext_info_int, ghost_ext_info_flt, BCindex,
                      d_halonodes, cst):
    """haloghost_value_dirichlet/_neumann/_neumannNH/_nonslip: set the halo
    ghosts tagged `BCindex` that hang off a node of `d_halonodes`.

    `value` is indexed per halo ghost for dirichlet and is the halo cell field
    `w_halo` for the other kinds -- two different index spaces (sizehaloghost
    vs nbhalos) in the same slot. `BCindex` is a scalar, not an array."""
    r, rw, w = acc
    kernel(
      r(value), rw(w_haloghost), r(node_haloghostid), r(ghost_ext_info_int),
      r(ghost_ext_info_flt), BCindex, r(d_halonodes), r(cst)
    )

  @staticmethod
  def slip_ghost_2d(kernel, acc, u_c, v_c, u_ghost, v_ghost, face_cellid,
                    bc_faces, normal):
    """ghost_value_slip_2d: free-slip reflection of the velocity behind every
    face of `bc_faces`. Coupled: both components are needed together."""
    r, rw, w = acc
    kernel(
      r(u_c), r(v_c), rw(u_ghost), rw(v_ghost), r(face_cellid), r(bc_faces),
      r(normal)
    )

  @staticmethod
  def slip_ghost_3d(kernel, acc, u_c, v_c, w_c, u_ghost, v_ghost, w_ghost,
                    face_cellid, bc_faces, normal):
    """ghost_value_slip_3d: 3D counterpart of slip_ghost_2d."""
    r, rw, w = acc
    kernel(
      r(u_c), r(v_c), r(w_c), rw(u_ghost), rw(v_ghost), rw(w_ghost),
      r(face_cellid), r(bc_faces), r(normal)
    )

  @staticmethod
  def slip_haloghost_2d(kernel, acc, u_halo, v_halo, u_haloghost, v_haloghost,
                        node_haloghostid, ghost_ext_info_int,
                        ghost_ext_info_flt, BCindex, d_halonodes):
    """haloghost_value_slip_2d: free-slip reflection on the halo ghosts tagged
    `BCindex`. No `cst` argument, unlike the scalar haloghost kernels."""
    r, rw, w = acc
    kernel(
      r(u_halo), r(v_halo), rw(u_haloghost), rw(v_haloghost),
      r(node_haloghostid), r(ghost_ext_info_int), r(ghost_ext_info_flt),
      BCindex, r(d_halonodes)
    )

  @staticmethod
  def slip_haloghost_3d(kernel, acc, u_halo, v_halo, w_halo, u_haloghost,
                        v_haloghost, w_haloghost, node_haloghostid,
                        ghost_ext_info_int, ghost_ext_info_flt, BCindex,
                        d_halonodes):
    """haloghost_value_slip_3d: 3D counterpart of slip_haloghost_2d."""
    r, rw, w = acc
    kernel(
      r(u_halo), r(v_halo), r(w_halo), rw(u_haloghost), rw(v_haloghost),
      rw(w_haloghost), r(node_haloghostid), r(ghost_ext_info_int),
      r(ghost_ext_info_flt), BCindex, r(d_halonodes)
    )

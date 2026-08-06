#pragma once

#include <cuda_runtime_api.h>

#include "array_view.hpp"
#include "precision.hpp"

// GPU entry points for the boundary-condition kernels. Each mirrors its CPU
// twin in boundary_compute.hpp, but every ArrayView points at device (CuPy)
// memory. They launch one thread per entry in the gather list (bc_faces for the
// ghost kernels, d_halonodes for the halo-ghost ones) over the given stream and
// write the ghost arrays in place. Asynchronous: the caller must synchronize and
// check cudaGetLastError(). Declared with cudaStream_t (this header pulls in the
// CUDA runtime) so the C++ binding TUs can call them without seeing the kernels
// themselves; the kernels live in gpu/<kernel>_cuda.cu.

// Scalar conditions on local boundary faces; kernels in gpu/ghost_value_cuda.cu.
void launch_ghost_value_dirichlet(ArrayView<const real_t, 1> value,
                                  ArrayView<real_t, 1> w_ghost,
                                  ArrayView<const index_t, 2> face_cellid,
                                  ArrayView<const index_t, 1> bc_faces,
                                  ArrayView<const real_t, 1> cst,
                                  ArrayView<const real_t, 1> face_dist_ortho,
                                  cudaStream_t stream);

void launch_ghost_value_neumann(ArrayView<const real_t, 1> w_c,
                                ArrayView<real_t, 1> w_ghost,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const index_t, 1> bc_faces,
                                ArrayView<const real_t, 1> cst,
                                ArrayView<const real_t, 1> face_dist_ortho,
                                cudaStream_t stream);

void launch_ghost_value_neumannNH(ArrayView<const real_t, 1> w_c,
                                  ArrayView<real_t, 1> w_ghost,
                                  ArrayView<const index_t, 2> face_cellid,
                                  ArrayView<const index_t, 1> bc_faces,
                                  ArrayView<const real_t, 1> cst,
                                  ArrayView<const real_t, 1> face_dist_ortho,
                                  cudaStream_t stream);

void launch_ghost_value_nonslip(ArrayView<const real_t, 1> w_c,
                                ArrayView<real_t, 1> w_ghost,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const index_t, 1> bc_faces,
                                ArrayView<const real_t, 1> cst,
                                ArrayView<const real_t, 1> face_dist_ortho,
                                cudaStream_t stream);


// Scalar conditions on halo ghosts; kernels in gpu/haloghost_value_cuda.cu.
// Dirichlet's first array is the prescribed per-halo-ghost value array, not the
// halo cell field the other three take (see boundary_compute.hpp).
void launch_haloghost_value_dirichlet(
    ArrayView<const real_t, 1> value_haloghost,
    ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream);

void launch_haloghost_value_neumann(
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream);

void launch_haloghost_value_neumannNH(
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream);

void launch_haloghost_value_nonslip(
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, ArrayView<const real_t, 1> cst,
    cudaStream_t stream);


// Free-slip (vector) conditions; kernels in gpu/<kernel>_cuda.cu.
void launch_ghost_value_slip_2d(ArrayView<const real_t, 1> u_c,
                                ArrayView<const real_t, 1> v_c,
                                ArrayView<real_t, 1> u_ghost,
                                ArrayView<real_t, 1> v_ghost,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const index_t, 1> bc_faces,
                                ArrayView<const real_t, 2> normal,
                                cudaStream_t stream);

void launch_ghost_value_slip_3d(ArrayView<const real_t, 1> u_c,
                                ArrayView<const real_t, 1> v_c,
                                ArrayView<const real_t, 1> w_c,
                                ArrayView<real_t, 1> u_ghost,
                                ArrayView<real_t, 1> v_ghost,
                                ArrayView<real_t, 1> w_ghost,
                                ArrayView<const index_t, 2> face_cellid,
                                ArrayView<const index_t, 1> bc_faces,
                                ArrayView<const real_t, 2> normal,
                                cudaStream_t stream);

void launch_haloghost_value_slip_2d(
    ArrayView<const real_t, 1> u_halo, ArrayView<const real_t, 1> v_halo,
    ArrayView<real_t, 1> u_haloghost, ArrayView<real_t, 1> v_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, cudaStream_t stream);

void launch_haloghost_value_slip_3d(
    ArrayView<const real_t, 1> u_halo, ArrayView<const real_t, 1> v_halo,
    ArrayView<const real_t, 1> w_halo, ArrayView<real_t, 1> u_haloghost,
    ArrayView<real_t, 1> v_haloghost, ArrayView<real_t, 1> w_haloghost,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> ghost_ext_info_int,
    ArrayView<const real_t, 2> ghost_ext_info_flt, index_t BCindex,
    ArrayView<const index_t, 1> d_halonodes, cudaStream_t stream);

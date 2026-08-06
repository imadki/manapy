#pragma once

#include <cuda_runtime_api.h>

#include "array_view.hpp"
#include "precision.hpp"

// GPU entry point for the 2D least-squares cell gradient. Mirrors
// cell_gradient_2d (variable_compute.hpp) but every ArrayView points at
// device (CuPy) memory. Launches one thread per cell over the given stream and
// writes w_x/w_y/w_z in place. Asynchronous: the caller must synchronize and
// check cudaGetLastError(). Declared with cudaStream_t (this header pulls in
// the CUDA runtime) so the C++ binding TU can call it without seeing the
// kernel itself; the kernel lives in gpu/cell_gradient_2d_cuda.cu.
void launch_cell_gradient_2d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const index_t, 2> cell_cellnid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> cell_ghostnid,
    ArrayView<const index_t, 2> cell_haloghostnid,
    ArrayView<const index_t, 2> cell_halonid,
    ArrayView<const index_t, 2> cells,
    ArrayView<const index_t, 2> node_periodicid,
    ArrayView<const index_t, 1> node_oldname,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_x,
    ArrayView<real_t, 1> w_y, ArrayView<real_t, 1> w_z,
    ArrayView<const index_t, 1> ghost_faceid, cudaStream_t stream);


// GPU entry point for the 2D center-to-vertex interpolation. Mirrors
// center_to_vertex_2d (variable_compute.hpp) but every ArrayView
// points at device (CuPy) memory. Launches one thread per node over the given
// stream and writes w_n in place. Asynchronous: the caller must synchronize and
// check cudaGetLastError(). Declared with cudaStream_t (this header pulls in the
// CUDA runtime) so the C++ binding TU can call it without seeing the kernel
// itself; the kernel lives in gpu/center_to_vertex_2d_cuda.cu.
void launch_center_to_vertex_2d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> node_cellid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> node_ghostid,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> node_periodicid,
    ArrayView<const index_t, 2> node_halonid, ArrayView<const real_t, 2> nodes,
    ArrayView<const index_t, 1> node_oldname,
    ArrayView<const real_t, 1> node_R_x, ArrayView<const real_t, 1> node_R_y,
    ArrayView<const real_t, 1> node_lambda_x,
    ArrayView<const real_t, 1> node_lambda_y,
    ArrayView<const index_t, 1> node_number,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_n,
    ArrayView<const index_t, 1> ghost_faceid, cudaStream_t stream);


// GPU entry point for the 2D face gradient. Mirrors face_gradient_2d
// (variable_compute.hpp) but every ArrayView points at device (CuPy)
// memory. Launches one thread per entry in each of the five face-index lists
// over the given stream and writes wx_face/wy_face in place. Asynchronous: the
// caller must synchronize and check cudaGetLastError(). Declared with
// cudaStream_t (this header pulls in the CUDA runtime) so the C++ binding TU
// can call it without seeing the kernels themselves; the kernels live in
// gpu/face_gradient_2d_cuda.cu.
void launch_face_gradient_2d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_halofid,
    ArrayView<const real_t, 1> face_airDiamond,
    ArrayView<const real_t, 2> face_f1, ArrayView<const real_t, 2> face_f2,
    ArrayView<const real_t, 2> face_f3, ArrayView<const real_t, 2> face_f4,
    ArrayView<real_t, 1> wx_face, ArrayView<real_t, 1> wy_face,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> dirichletfaces,
    ArrayView<const index_t, 1> neumann,
    ArrayView<const index_t, 1> d_periodicfaces, cudaStream_t stream);


// GPU entry point for the 2D Barth-Jespersen slope limiter. Mirrors
// barthlimiter_2d (variable_compute.hpp) but every ArrayView points at
// device (CuPy) memory. Launches one thread per cell over the given stream and
// writes psi in place. Asynchronous: the caller must synchronize and check
// cudaGetLastError(). Declared with cudaStream_t (this header pulls in the
// CUDA runtime) so the C++ binding TU can call it without seeing the kernel
// itself; the kernel lives in gpu/barthlimiter_2d_cuda.cu.
void launch_barthlimiter_2d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<real_t, 1> psi,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> cell_faceid,
    ArrayView<const index_t, 1> face_name, ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 2> cell_center, ArrayView<const real_t, 2> face_center,
    cudaStream_t stream);


// GPU entry point for the 2D van Albada / Venkatakrishnan slope limiter.
// Mirrors vanalbadalimiter_2d (variable_compute.hpp) but every ArrayView
// points at device (CuPy) memory. Launches one thread per cell over the given
// stream and writes psi in place. Asynchronous: the caller must synchronize
// and check cudaGetLastError(). Declared with cudaStream_t (this header pulls
// in the CUDA runtime) so the C++ binding TU can call it without seeing the
// kernel itself; the kernel lives in gpu/vanalbadalimiter_2d_cuda.cu.
void launch_vanalbadalimiter_2d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<real_t, 1> psi,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> cell_faceid,
    ArrayView<const index_t, 1> face_name, ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 2> cell_center, ArrayView<const real_t, 2> face_center,
    cudaStream_t stream);


// GPU entry point for the 3D face gradient. Mirrors face_gradient_3d
// (variable_compute.hpp) but every ArrayView points at device (CuPy) memory.
// Launches one thread per entry in each of the five face-index lists over the
// given stream and writes wx_face/wy_face/wz_face in place. Asynchronous: the
// caller must synchronize and check cudaGetLastError(). Declared with
// cudaStream_t (this header pulls in the CUDA runtime) so the C++ binding TU
// can call it without seeing the kernels themselves; the kernels live in
// gpu/face_gradient_3d_cuda.cu.
void launch_face_gradient_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_node,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> faces,
    ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 1> face_air_diamond,
    ArrayView<const real_t, 2> face_normal, ArrayView<const real_t, 2> face_f1,
    ArrayView<const real_t, 2> face_f2, ArrayView<real_t, 1> wx_face,
    ArrayView<real_t, 1> wy_face, ArrayView<real_t, 1> wz_face,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_halofaces,
    ArrayView<const index_t, 1> dirichletfaces,
    ArrayView<const index_t, 1> neumann,
    ArrayView<const index_t, 1> d_periodicboundaryfaces, cudaStream_t stream);


// GPU entry point for the 3D least-squares cell gradient. Mirrors
// cell_gradient_3d (variable_compute.hpp) but every ArrayView points at
// device (CuPy) memory. Launches one thread per cell over the given stream and
// writes w_x/w_y/w_z in place. Asynchronous: the caller must synchronize and
// check cudaGetLastError(). Declared with cudaStream_t (this header pulls in
// the CUDA runtime) so the C++ binding TU can call it without seeing the
// kernel itself; the kernel lives in gpu/cell_gradient_3d_cuda.cu.
void launch_cell_gradient_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const index_t, 2> cell_cellnid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> cell_ghostnid,
    ArrayView<const index_t, 2> cell_haloghostnid,
    ArrayView<const index_t, 2> cell_halonid,
    ArrayView<const index_t, 2> cell_periodicfid,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_x,
    ArrayView<real_t, 1> w_y, ArrayView<real_t, 1> w_z,
    ArrayView<const index_t, 1> ghost_faceid, cudaStream_t stream);


// GPU entry point for the 3D center-to-vertex interpolation. Mirrors
// center_to_vertex_3d (variable_compute.hpp) but every ArrayView points at
// device (CuPy) memory. Launches one thread per node over the given stream and
// writes w_n in place. Asynchronous: the caller must synchronize and check
// cudaGetLastError(). Declared with cudaStream_t (this header pulls in the
// CUDA runtime) so the C++ binding TU can call it without seeing the kernel
// itself; the kernel lives in gpu/center_to_vertex_3d_cuda.cu.
void launch_center_to_vertex_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_haloghost,
    ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> halo_centvol,
    ArrayView<const index_t, 2> node_cellid,
    ArrayView<const real_t, 2> ghost_info_flt,
    ArrayView<const real_t, 2> ghost_ext_info_flt,
    ArrayView<const index_t, 2> node_ghostid,
    ArrayView<const index_t, 2> node_haloghostid,
    ArrayView<const index_t, 2> node_periodicid,
    ArrayView<const index_t, 2> node_halonid, ArrayView<const real_t, 2> nodes,
    ArrayView<const index_t, 1> node_oldname,
    ArrayView<const real_t, 1> node_R_x, ArrayView<const real_t, 1> node_R_y,
    ArrayView<const real_t, 1> node_R_z, ArrayView<const real_t, 1> node_lambda_x,
    ArrayView<const real_t, 1> node_lambda_y,
    ArrayView<const real_t, 1> node_lambda_z,
    ArrayView<const index_t, 1> node_number,
    ArrayView<const real_t, 2> cell_shift, ArrayView<real_t, 1> w_n,
    ArrayView<const index_t, 1> ghost_faceid, cudaStream_t stream);


// GPU entry point for the 3D Barth-Jespersen slope limiter. Mirrors
// barthlimiter_3d (variable_compute.hpp) but every ArrayView points at
// device (CuPy) memory. Launches one thread per cell over the given stream and
// writes psi in place. Asynchronous: the caller must synchronize and check
// cudaGetLastError(). Declared with cudaStream_t (this header pulls in the
// CUDA runtime) so the C++ binding TU can call it without seeing the kernel
// itself; the kernel lives in gpu/barthlimiter_3d_cuda.cu.
void launch_barthlimiter_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<real_t, 1> psi, ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 2> cell_faceid, ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> face_haloid, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center, cudaStream_t stream);


// GPU entry point for the 3D van Albada / Venkatakrishnan slope limiter.
// Mirrors vanalbadalimiter_3d (variable_compute.hpp) but every ArrayView
// points at device (CuPy) memory. Launches one thread per cell over the given
// stream and writes psi in place. Asynchronous: the caller must synchronize
// and check cudaGetLastError(). Declared with cudaStream_t (this header pulls
// in the CUDA runtime) so the C++ binding TU can call it without seeing the
// kernel itself; the kernel lives in gpu/vanalbadalimiter_3d_cuda.cu.
void launch_vanalbadalimiter_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<real_t, 1> psi, ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 2> cell_faceid, ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> face_haloid, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center, cudaStream_t stream);


// GPU entry point for face-to-cell averaging. Mirrors facetocell
// (variable_compute.hpp) but every ArrayView points at device (CuPy) memory.
// Launches one thread per cell over the given stream and writes u_c in place.
// Asynchronous: the caller must synchronize and check cudaGetLastError().
// Declared with cudaStream_t (this header pulls in the CUDA runtime) so the
// C++ binding TU can call it without seeing the kernel itself; the kernel
// lives in gpu/facetocell_cuda.cu.
void launch_facetocell(ArrayView<const real_t, 1> u_face,
                        ArrayView<const index_t, 2> cell_faceid,
                        ArrayView<real_t, 1> u_c, cudaStream_t stream);


// GPU entry point for cell-to-face averaging. Mirrors celltoface
// (variable_compute.hpp) but every ArrayView points at device (CuPy) memory.
// Launches one thread per entry in each of the three face-index lists over
// the given stream and writes u_face in place. Asynchronous: the caller must
// synchronize and check cudaGetLastError(). Declared with cudaStream_t (this
// header pulls in the CUDA runtime) so the C++ binding TU can call it without
// seeing the kernels themselves; the kernels live in gpu/celltoface_cuda.cu.
void launch_celltoface(
    ArrayView<const real_t, 1> u_cell, ArrayView<real_t, 1> u_face,
    ArrayView<const real_t, 1> u_ghost, ArrayView<const real_t, 1> u_halo,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 1> face_halofid,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_halofaces, cudaStream_t stream);

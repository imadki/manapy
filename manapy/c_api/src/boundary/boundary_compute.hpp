#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// ---------------------------------------------------------------------------
// Scalar boundary conditions on local boundary faces (translations of
// ghost_value_dirichlet / _neumann / _neumannNH / _nonslip in to_convert.py).
// CPU entry points: loop over bc_faces and call ghost_value_face
// (common/ghost_value_common.hpp) on the host.
//
// bc_faces is a gather list of face indices, not a full-range counter: it holds
// the ids of the faces carrying this boundary condition. w_ghost is written in
// place at those ids. cst / face_dist_ortho are only read by the neumannNH
// variant and face_cellid only by the non-Dirichlet ones; they stay in every
// signature for parity with the Python API.
// ---------------------------------------------------------------------------

// w_ghost(i) = value(i) -- imposed value on the face.
void ghost_value_dirichlet(ArrayView<const real_t, 1> value,
                           ArrayView<real_t, 1> w_ghost,
                           ArrayView<const index_t, 2> face_cellid,
                           ArrayView<const index_t, 1> bc_faces,
                           ArrayView<const real_t, 1> cst,
                           ArrayView<const real_t, 1> face_dist_ortho);

// w_ghost(i) = w_c(face_cellid(i, 0)) -- zero normal gradient.
void ghost_value_neumann(ArrayView<const real_t, 1> w_c,
                         ArrayView<real_t, 1> w_ghost,
                         ArrayView<const index_t, 2> face_cellid,
                         ArrayView<const index_t, 1> bc_faces,
                         ArrayView<const real_t, 1> cst,
                         ArrayView<const real_t, 1> face_dist_ortho);

// w_ghost(i) = w_c(face_cellid(i, 0)) + cst(i) * face_dist_ortho(i) --
// inhomogeneous Neumann, i.e. an imposed normal gradient cst.
void ghost_value_neumannNH(ArrayView<const real_t, 1> w_c,
                           ArrayView<real_t, 1> w_ghost,
                           ArrayView<const index_t, 2> face_cellid,
                           ArrayView<const index_t, 1> bc_faces,
                           ArrayView<const real_t, 1> cst,
                           ArrayView<const real_t, 1> face_dist_ortho);

// w_ghost(i) = -w_c(face_cellid(i, 0)) -- odd reflection, so the field
// vanishes at the wall (no-slip on a velocity component).
void ghost_value_nonslip(ArrayView<const real_t, 1> w_c,
                         ArrayView<real_t, 1> w_ghost,
                         ArrayView<const index_t, 2> face_cellid,
                         ArrayView<const index_t, 1> bc_faces,
                         ArrayView<const real_t, 1> cst,
                         ArrayView<const real_t, 1> face_dist_ortho);



// ---------------------------------------------------------------------------
// Scalar boundary conditions on the halo ghosts hanging off halo nodes
// (translations of haloghost_value_dirichlet / _neumann / _neumannNH /
// _nonslip in to_convert.py). CPU entry points: loop over d_halonodes and call
// haloghost_value_node (common/haloghost_value_common.hpp) on the host.
//
// d_halonodes is a gather list of node indices. For each of those nodes every
// halo ghost listed in node_haloghostid whose boundary tag
// (ghost_ext_info_int(ghost_id, 1)) equals BCindex is updated, so unrelated
// boundaries sharing the node are left alone. The last column of
// node_haloghostid holds the number of valid entries in that row.
// w_haloghost is written in place.
//
// A halo ghost is reachable from every node of its face, so several entries of
// d_halonodes can land on the same ghost. Every kernel here writes a value that
// depends only on ghost_id (never on the node), so the outcome does not depend
// on the visit order -- which is what makes the GPU twins deterministic.
// ---------------------------------------------------------------------------

// w_haloghost(g) = value_haloghost(g) -- imposed value. Note this first array
// is the prescribed per-halo-ghost value array, not the halo cell field the
// other three take.
void haloghost_value_dirichlet(ArrayView<const real_t, 1> value_haloghost,
                               ArrayView<real_t, 1> w_haloghost,
                               ArrayView<const index_t, 2> node_haloghostid,
                               ArrayView<const index_t, 2> ghost_ext_info_int,
                               ArrayView<const real_t, 2> ghost_ext_info_flt,
                               index_t BCindex,
                               ArrayView<const index_t, 1> d_halonodes,
                               ArrayView<const real_t, 1> cst);

// w_haloghost(g) = w_halo(ghost_ext_info_int(g, 0)) -- zero normal gradient.
void haloghost_value_neumann(ArrayView<const real_t, 1> w_halo,
                             ArrayView<real_t, 1> w_haloghost,
                             ArrayView<const index_t, 2> node_haloghostid,
                             ArrayView<const index_t, 2> ghost_ext_info_int,
                             ArrayView<const real_t, 2> ghost_ext_info_flt,
                             index_t BCindex,
                             ArrayView<const index_t, 1> d_halonodes,
                             ArrayView<const real_t, 1> cst);

// w_haloghost(g) = w_halo(...) + cst(g) * 2*|ghost_ext_info_flt(g, 0)| --
// imposed normal gradient cst, which is indexed per ghost.
void haloghost_value_neumannNH(ArrayView<const real_t, 1> w_halo,
                               ArrayView<real_t, 1> w_haloghost,
                               ArrayView<const index_t, 2> node_haloghostid,
                               ArrayView<const index_t, 2> ghost_ext_info_int,
                               ArrayView<const real_t, 2> ghost_ext_info_flt,
                               index_t BCindex,
                               ArrayView<const index_t, 1> d_halonodes,
                               ArrayView<const real_t, 1> cst);

// w_haloghost(g) = -w_halo(ghost_ext_info_int(g, 0)) -- odd reflection.
void haloghost_value_nonslip(ArrayView<const real_t, 1> w_halo,
                             ArrayView<real_t, 1> w_haloghost,
                             ArrayView<const index_t, 2> node_haloghostid,
                             ArrayView<const index_t, 2> ghost_ext_info_int,
                             ArrayView<const real_t, 2> ghost_ext_info_flt,
                             index_t BCindex,
                             ArrayView<const index_t, 1> d_halonodes,
                             ArrayView<const real_t, 1> cst);



// ---------------------------------------------------------------------------
// Free-slip (slip wall): a vector boundary condition, so unlike the scalar ones
// above it takes all velocity components together plus the face normal
// (translations of ghost_value_slip_2d/_3d and haloghost_value_slip_2d/_3d in
// to_convert.py). The velocity is reflected across the face,
//     U_ghost = U_c - 2 (U_c . n) n,
// so the normal component vanishes at the wall and the tangential one is
// preserved. The normal is normalised internally, so it works whether it is
// unit or area-scaled.
// ---------------------------------------------------------------------------

// CPU entry point: loops over bc_faces calling ghost_value_slip_2d_face
// (common/ghost_value_slip_2d_common.hpp). u_ghost/v_ghost written in place.
void ghost_value_slip_2d(ArrayView<const real_t, 1> u_c,
                         ArrayView<const real_t, 1> v_c,
                         ArrayView<real_t, 1> u_ghost,
                         ArrayView<real_t, 1> v_ghost,
                         ArrayView<const index_t, 2> face_cellid,
                         ArrayView<const index_t, 1> bc_faces,
                         ArrayView<const real_t, 2> normal);

// CPU entry point: loops over bc_faces calling ghost_value_slip_3d_face
// (common/ghost_value_slip_3d_common.hpp).
void ghost_value_slip_3d(ArrayView<const real_t, 1> u_c,
                         ArrayView<const real_t, 1> v_c,
                         ArrayView<const real_t, 1> w_c,
                         ArrayView<real_t, 1> u_ghost,
                         ArrayView<real_t, 1> v_ghost,
                         ArrayView<real_t, 1> w_ghost,
                         ArrayView<const index_t, 2> face_cellid,
                         ArrayView<const index_t, 1> bc_faces,
                         ArrayView<const real_t, 2> normal);

// CPU entry point: loops over d_halonodes calling
// haloghost_value_slip_2d_node (common/haloghost_value_slip_2d_common.hpp),
// which updates every halo ghost of that node tagged BCindex.
void haloghost_value_slip_2d(ArrayView<const real_t, 1> u_halo,
                             ArrayView<const real_t, 1> v_halo,
                             ArrayView<real_t, 1> u_haloghost,
                             ArrayView<real_t, 1> v_haloghost,
                             ArrayView<const index_t, 2> node_haloghostid,
                             ArrayView<const index_t, 2> ghost_ext_info_int,
                             ArrayView<const real_t, 2> ghost_ext_info_flt,
                             index_t BCindex,
                             ArrayView<const index_t, 1> d_halonodes);

// CPU entry point: loops over d_halonodes calling
// haloghost_value_slip_3d_node (common/haloghost_value_slip_3d_common.hpp).
void haloghost_value_slip_3d(ArrayView<const real_t, 1> u_halo,
                             ArrayView<const real_t, 1> v_halo,
                             ArrayView<const real_t, 1> w_halo,
                             ArrayView<real_t, 1> u_haloghost,
                             ArrayView<real_t, 1> v_haloghost,
                             ArrayView<real_t, 1> w_haloghost,
                             ArrayView<const index_t, 2> node_haloghostid,
                             ArrayView<const index_t, 2> ghost_ext_info_int,
                             ArrayView<const real_t, 2> ghost_ext_info_flt,
                             index_t BCindex,
                             ArrayView<const index_t, 1> d_halonodes);

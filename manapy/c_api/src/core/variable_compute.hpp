#pragma once

#include "array_view.hpp"
#include "precision.hpp"

// Least-squares gradient of a cell-centred field on a 2D unstructured mesh,
// including ghost, periodic and halo contributions (translation of
// _cell_gradient_2d in to_convert.py). CPU entry point: loops over every cell
// and calls cell_gradient_2d_element (common/cell_gradient_2d_common.hpp) on
// the host.
//
// All matrices are C-contiguous; the last column of each connectivity matrix
// (*_nid, cells, node_periodicid) holds the number of valid entries in that
// row. nbelement (the number of cells to process) is w_c's extent.
void cell_gradient_2d(
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
    ArrayView<const index_t, 1> ghost_faceid);



// Distance-weighted interpolation of a cell-centred field onto the mesh
// vertices on a 2D unstructured mesh, including ghost, halo, halo-ghost and
// periodic contributions (translation of _centertovertex_2d in to_convert.py).
// CPU entry point: loops over every node and calls center_to_vertex_2d_node
// (common/center_to_vertex_2d_common.hpp) on the host.
//
// All matrices are C-contiguous; the last column of each connectivity matrix
// (node_*id, node_periodicid) holds the number of valid entries in that row.
// The number of nodes to process is nodes' first extent. w_n is written in
// place.
void center_to_vertex_2d(
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
    ArrayView<const index_t, 1> ghost_faceid);



// Green-Gauss-style gradient of a cell-centred field at each face midpoint on
// a 2D unstructured mesh (translation of _face_gradient_2d in
// to_convert.py). CPU entry point: dispatches each of the five face index
// lists (interior, periodic, halo, Dirichlet, Neumann) to
// face_gradient_2d_face (common/face_gradient_2d_common.hpp) on the host.
//
// d_innerfaces, d_periodicfaces, d_halofaces, dirichletfaces and neumann are
// gather lists of face indices, not full-range counters: each holds the ids
// of the faces belonging to that category. wx_face/wy_face are written in
// place at those ids.
void face_gradient_2d(
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
    ArrayView<const index_t, 1> d_periodicfaces);



// Barth-Jespersen slope limiter on a 2D unstructured mesh (translation of
// _barthlimiter_2d in to_convert.py). CPU entry point: loops over every cell
// and calls barthlimiter_2d_cell (common/barthlimiter_2d_common.hpp) on the
// host.
//
// The last column of cell_faceid holds the number of valid entries in that
// row. nbelement (the number of cells to process) is w_c's extent. psi is
// written in place.
void barthlimiter_2d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<real_t, 1> psi,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> cell_faceid,
    ArrayView<const index_t, 1> face_name, ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 2> cell_center, ArrayView<const real_t, 2> face_center);



// van Albada / Venkatakrishnan slope limiter on a 2D unstructured mesh
// (translation of _vanalbadalimiter_2d in to_convert.py). Same
// neighbourhood-min/max structure as barthlimiter_2d, but smooth: the
// per-face factor is phi(y) = (y^2+2y)/(y^2+y+2) instead of min(1,y). CPU
// entry point: loops over every cell and calls vanalbadalimiter_2d_cell
// (common/vanalbadalimiter_2d_common.hpp) on the host.
//
// The last column of cell_faceid holds the number of valid entries in that
// row. nbelement (the number of cells to process) is w_c's extent. psi is
// written in place.
void vanalbadalimiter_2d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<real_t, 1> psi,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 2> cell_faceid,
    ArrayView<const index_t, 1> face_name, ArrayView<const index_t, 1> face_haloid,
    ArrayView<const real_t, 2> cell_center, ArrayView<const real_t, 2> face_center);



// Gradient of a cell-centred field at each face midpoint on a 3D unstructured
// mesh (translation of _face_gradient_3d in to_convert.py). CPU entry point:
// dispatches each of the five face index lists (interior, periodic, halo,
// Dirichlet, Neumann) to face_gradient_3d_face
// (common/face_gradient_3d_common.hpp) on the host.
//
// d_innerfaces, d_periodicboundaryfaces, d_halofaces, dirichletfaces and
// neumann are gather lists of face indices, not full-range counters: each
// holds the ids of the faces belonging to that category. faces holds up to 4
// node ids per row with the valid count (3 or 4) in the last column.
// wx_face/wy_face/wz_face are written in place at those ids.
void face_gradient_3d(
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
    ArrayView<const index_t, 1> d_periodicboundaryfaces);



// Least-squares gradient of a cell-centred field on a 3D unstructured mesh,
// including ghost, periodic and halo contributions (translation of
// _cell_gradient_3d in to_convert.py). CPU entry point: loops over every cell
// and calls cell_gradient_3d_element (common/cell_gradient_3d_common.hpp) on
// the host.
//
// All matrices are C-contiguous; the last column of each connectivity matrix
// (*_nid, cell_periodicfid) holds the number of valid entries in that row.
// nbelement (the number of cells to process) is w_c's extent.
void cell_gradient_3d(
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
    ArrayView<const index_t, 1> ghost_faceid);



// Distance-weighted interpolation of a cell-centred field onto the mesh
// vertices on a 3D unstructured mesh, including ghost, halo, halo-ghost and
// periodic contributions (translation of _centertovertex_3d in
// to_convert.py). CPU entry point: loops over every node and calls
// center_to_vertex_3d_node (common/center_to_vertex_3d_common.hpp) on the
// host.
//
// All matrices are C-contiguous; the last column of each connectivity matrix
// (node_*id, node_periodicid) holds the number of valid entries in that row.
// The number of nodes to process is nodes' first extent. w_n is written in
// place.
void center_to_vertex_3d(
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
    ArrayView<const index_t, 1> ghost_faceid);



// Barth-Jespersen slope limiter on a 3D unstructured mesh (translation of
// _barthlimiter_3d in to_convert.py). CPU entry point: loops over every cell
// and calls barthlimiter_3d_cell (common/barthlimiter_3d_common.hpp) on the
// host.
//
// The last column of cell_faceid holds the number of valid entries in that
// row. nbelement (the number of cells to process) is w_c's extent. psi is
// written in place.
void barthlimiter_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<real_t, 1> psi, ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 2> cell_faceid, ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> face_haloid, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center);



// van Albada / Venkatakrishnan slope limiter on a 3D unstructured mesh
// (translation of _vanalbadalimiter_3d in to_convert.py). Same
// neighbourhood-min/max structure as barthlimiter_3d, but smooth: the
// per-face factor is phi(y) = (y^2+2y)/(y^2+y+2) instead of min(1,y). CPU
// entry point: loops over every cell and calls vanalbadalimiter_3d_cell
// (common/vanalbadalimiter_3d_common.hpp) on the host.
//
// The last column of cell_faceid holds the number of valid entries in that
// row. nbelement (the number of cells to process) is w_c's extent. psi is
// written in place.
void vanalbadalimiter_3d(
    ArrayView<const real_t, 1> w_c, ArrayView<const real_t, 1> w_ghost,
    ArrayView<const real_t, 1> w_halo, ArrayView<const real_t, 1> w_x,
    ArrayView<const real_t, 1> w_y, ArrayView<const real_t, 1> w_z,
    ArrayView<real_t, 1> psi, ArrayView<const index_t, 2> face_cellid,
    ArrayView<const index_t, 2> cell_faceid, ArrayView<const index_t, 1> face_name,
    ArrayView<const index_t, 1> face_haloid, ArrayView<const real_t, 2> cell_center,
    ArrayView<const real_t, 2> face_center);



// Face-to-cell averaging on a 2D/3D unstructured mesh (translation of
// facetocell in to_convert.py). CPU entry point: loops over every cell and
// calls facetocell_cell (common/facetocell_common.hpp) on the host.
//
// The last column of cell_faceid holds the number of valid entries in that
// row. nbelement (the number of cells to process) is u_c's extent. u_c is
// written in place.
void facetocell(ArrayView<const real_t, 1> u_face,
                 ArrayView<const index_t, 2> cell_faceid, ArrayView<real_t, 1> u_c);



// Cell-to-face averaging on a 2D/3D unstructured mesh (translation of
// celltoface in to_convert.py). CPU entry point: dispatches each of the three
// face index lists (interior, halo, boundary) to celltoface_face
// (common/celltoface_common.hpp) on the host.
//
// d_innerfaces, d_boundaryfaces and d_halofaces are gather lists of face
// indices, not full-range counters: each holds the ids of the faces
// belonging to that category. u_face is written in place at those ids.
void celltoface(
    ArrayView<const real_t, 1> u_cell, ArrayView<real_t, 1> u_face,
    ArrayView<const real_t, 1> u_ghost, ArrayView<const real_t, 1> u_halo,
    ArrayView<const index_t, 2> face_cellid, ArrayView<const index_t, 1> face_halofid,
    ArrayView<const index_t, 1> d_innerfaces,
    ArrayView<const index_t, 1> d_boundaryfaces,
    ArrayView<const index_t, 1> d_halofaces);

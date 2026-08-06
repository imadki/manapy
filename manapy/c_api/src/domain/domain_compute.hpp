#pragma once

#include <cstdint>

#include "array_view.hpp"
#include "precision.hpp"

// Declarations for domain kernels (mesh connectivity/geometry, ported from
// src/domain/to_convert.py — see src/domain/Steps.md for the batch plan).
// CPU-only: unlike src/core, there is no matching .cuh/.cu pair, so no
// MANAPY_COMPUTE_HOST_DEVICE split between a shared element function and a
// host loop — each kernel's cpu/<kernel>_cpu.cpp holds the whole
// implementation.
//
// All matrices are C-contiguous; the last column of a connectivity matrix
// (cells, node_cellid, cell_cellnid, phy_faces, ...) holds the number of
// valid entries in that row, matching to_convert.py's `array[-1]`
// convention.

// Batch 1: node/cell connectivity ------------------------------------------

// _count_max_node_cellid: increments res[node] once for every cell that
// lists `node` among its nodes. Called before create_node_cellid to size
// node_cellid's row width (max(res) + 1 slots).
void count_max_node_cellid(ArrayView<const index_t, 2> cells,
                            ArrayView<index_t, 1> res);

// _create_node_cellid: for each node, the sorted (ascending) list of cells
// that reference it. node_cellid must already be zeroed; its last column is
// used as a running write cursor / final count.
void create_node_cellid(ArrayView<const index_t, 2> cells,
                         ArrayView<index_t, 2> node_cellid);

// _get_cell_nb_phyid: increments cell_nb_phyid[cell] once per physical face
// that has one of `cell`'s nodes on it (each cell counted at most once per
// physical face). i_visited is scratch, sized to the number of cells;
// its contents on entry are irrelevant, sized/typed so a physical-face
// index never collides with a stale value.
void get_cell_nb_phyid(ArrayView<const index_t, 2> phy_faces,
                        ArrayView<const index_t, 2> node_cellid,
                        ArrayView<index_t, 1> i_visited,
                        ArrayView<index_t, 1> cell_nb_phyid);

// _count_max_cell_cellnid: for each cell, the number of distinct
// node-neighboring cells (excluding itself); returns the maximum across all
// cells, used to size cell_cellnid's row width before create_cell_cellnid.
// i_visited is scratch, sized to the number of cells.
index_t count_max_cell_cellnid(ArrayView<const index_t, 2> cells,
                                ArrayView<const index_t, 2> node_cellid,
                                ArrayView<index_t, 1> i_visited);

// _create_cell_cellnid: node-adjacency between cells. For each cell `i` and
// each node-neighboring cell `nc` reached through one of `i`'s nodes, records
// `i` into cell_cellnid[nc] (skipping a duplicate only when it would repeat
// the immediately preceding entry — matches the Python original exactly).
// cell_cellnid must already be zeroed.
void create_cell_cellnid(ArrayView<const index_t, 2> cells,
                          ArrayView<const index_t, 2> node_cellid,
                          ArrayView<index_t, 2> cell_cellnid);

// Batch 2: face/cell topology -----------------------------------------------

// _create_info: builds faces, cell->face, face->cell and cell->neighbor-cell
// (by shared face) tables in one pass over `cells`.
//
// tmp_cell_faces/tmp_size_info are scratch, overwritten every cell (sized
// (max_nb_faces, max_nb_face_nodes) and (max_nb_faces + 1,)).
// tmp_cell_faces_map is NOT scratch despite the name: it persists across the
// whole call, one row per cell, recording faces this cell has already
// contributed so the neighbor cell that shares them can find their id
// instead of creating a duplicate (sized (nb_cells, 2 * max_nb_faces + 1) --
// columns [0, nb_faces) hold neighbor cell ids, [nb_faces, 2*nb_faces) the
// matching face ids, and the last column a running count).
//
// faces, cell_faceid, face_cellid and cell_cellfid are outputs; cell_faceid
// and cell_cellfid must already be zeroed. faces_counter is a single-element
// in/out counter (must start at 0).
void create_info(ArrayView<const index_t, 2> cells,
                  ArrayView<const index_t, 2> node_cellid,
                  ArrayView<const std::int8_t, 1> cell_type,
                  ArrayView<index_t, 2> tmp_cell_faces,
                  ArrayView<index_t, 1> tmp_size_info,
                  ArrayView<index_t, 2> tmp_cell_faces_map,
                  ArrayView<index_t, 2> faces,
                  ArrayView<index_t, 2> cell_faceid,
                  ArrayView<index_t, 2> face_cellid,
                  ArrayView<index_t, 2> cell_cellfid,
                  ArrayView<index_t, 1> faces_counter);

// Batch 3: 3D face geometry --------------------------------------------------

// _compute_face_info_2d: measure (length), center and outward normal of
// every 2D face (an edge between two nodes). The normal is oriented away
// from face_cellid(i, 0) (the face's left/owning cell).
void compute_face_info_2d(ArrayView<const index_t, 2> faces,
                           ArrayView<const real_t, 2> nodes,
                           ArrayView<const index_t, 2> face_cellid,
                           ArrayView<const real_t, 2> cell_center,
                           ArrayView<real_t, 1> face_measure,
                           ArrayView<real_t, 2> face_center,
                           ArrayView<real_t, 2> face_normal);

// _compute_face_info_3d: measure (area), center, outward normal, tangent
// and binormal of every 3D face (a triangle or quad, faces(i, -1) vertices).
// A quad is treated as two triangles sharing the diagonal (p0,p1,p2) and
// (p0,p3,p2). Like compute_face_info_2d, the normal is oriented away from
// face_cellid(i, 0); face_normal and face_binormal both carry the 0.5
// factor from the shared-triangle decomposition (see to_convert.py).
void compute_face_info_3d(ArrayView<const index_t, 2> faces,
                           ArrayView<const real_t, 2> nodes,
                           ArrayView<const index_t, 2> face_cellid,
                           ArrayView<const real_t, 2> cell_center,
                           ArrayView<real_t, 1> face_measure,
                           ArrayView<real_t, 2> face_center,
                           ArrayView<real_t, 2> face_normal,
                           ArrayView<real_t, 2> face_tangent,
                           ArrayView<real_t, 2> face_binormal);

// Batch 4: ghost / boundary cell tables --------------------------------------

// _create_bf_cellid: for each physical (boundary) face, the local cell it
// belongs to and that cell's local face index. `intersect` is scratch,
// sized 2 (see intersect_face_nodes in domain_helpers.hpp). Throws if a
// physical face can't be resolved to a cell/face (usually a malformed
// mesh).
void create_bf_cellid(ArrayView<const index_t, 2> phy_faces,
                       ArrayView<const index_t, 2> node_cellid,
                       ArrayView<const index_t, 1> phyid_to_faceid,
                       ArrayView<const index_t, 2> cell_faceid,
                       ArrayView<index_t, 1> intersect,
                       ArrayView<index_t, 2> bf_cellid);

// _create_ghost_info: per boundary cell (bf_cellid, from create_bf_cellid),
// the mirrored "ghost" cell center reflected across its boundary face, plus
// bookkeeping (gamma weight, face center/normal, old name, global id).
// dim selects the gamma formula (2 or 3). ghost_info_flt columns: [0:3)
// ghost center, [3] gamma, [4:7) face center, [7:10) face normal.
// ghost_info_int columns: [0] cell id, [1] face index in the cell, [2] face
// old name, [3] cell global id (only if cell_loctoglob is non-empty), [4]
// face id. A row whose bf_cellid(i, 0) == -1 is a periodic face (no
// boundary cell to mirror): only ghost_info_int(i, 0) is written, to -1,
// so create_ghost_tables can skip it; the rest of that row is left
// untouched.
void create_ghost_info(ArrayView<const index_t, 2> bf_cellid,
                        ArrayView<const real_t, 2> cell_center,
                        ArrayView<const index_t, 2> cell_faceid,
                        ArrayView<const index_t, 1> cell_loctoglob,
                        ArrayView<const index_t, 2> faces,
                        ArrayView<const real_t, 2> nodes,
                        ArrayView<const index_t, 1> face_oldname,
                        ArrayView<const real_t, 2> face_normal,
                        ArrayView<const real_t, 2> face_center,
                        ArrayView<const real_t, 1> face_measure,
                        ArrayView<index_t, 2> ghost_info_int,
                        ArrayView<real_t, 2> ghost_info_flt, index_t dim);

// _create_ghost_tables: node_ghostid/cell_ghostid, indices into
// ghost_info_int for the ghost cells neighboring each node/cell.
// ghost_i_visited is scratch, sized to the number of cells. A row with
// ghost_info_int(i, 0) == -1 (periodic face, see create_ghost_info) is
// skipped.
void create_ghost_tables(ArrayView<const index_t, 2> ghost_info_int,
                          ArrayView<const index_t, 2> faces,
                          ArrayView<const index_t, 2> cell_faceid,
                          ArrayView<const index_t, 2> node_cellid,
                          ArrayView<index_t, 1> ghost_i_visited,
                          ArrayView<index_t, 2> node_ghostid,
                          ArrayView<index_t, 2> cell_ghostid);

// _count_max_bcell_halophyid: for each boundary cell (indexed via
// b_ncellid), the number of distinct halo-physical-face ids touching its
// nodes; returns the maximum, used to size bcell_halophyid's row width
// before create_bcell_halophyid. i_visited is scratch, sized to
// node_halophyid's node range.
index_t
count_max_bcell_halophyid(ArrayView<const index_t, 2> cells,
                           ArrayView<const index_t, 1> b_ncellid,
                           ArrayView<const index_t, 2> node_halophyid,
                           ArrayView<index_t, 1> i_visited);

// _create_bcell_halophyid: bcell_halophyid(i) = [cell global id, halo-phy-id,
// ..., count]. i_visited is scratch, same sizing as
// count_max_bcell_halophyid's.
void create_bcell_halophyid(ArrayView<const index_t, 2> cells,
                             ArrayView<const index_t, 1> b_ncellid,
                             ArrayView<const index_t, 2> node_halophyid,
                             ArrayView<index_t, 1> i_visited,
                             ArrayView<index_t, 2> bcell_halophyid);

// _get_max_b_ncellid: number of distinct cells touching any node in
// b_nodeid; returns the count, used to size b_ncellid before
// create_b_ncellid. b_visited is scratch, sized to the number of cells,
// zeroed on entry.
index_t get_max_b_ncellid(ArrayView<const index_t, 1> b_nodeid,
                           ArrayView<const index_t, 2> node_cellid,
                           ArrayView<std::int8_t, 1> b_visited);

// _create_b_ncellid: the distinct cells touching any node in b_nodeid,
// written into b_ncellid. b_visited is scratch, same sizing as
// get_max_b_ncellid's (pass a freshly-zeroed buffer, not the one already
// consumed by get_max_b_ncellid).
void create_b_ncellid(ArrayView<const index_t, 1> b_nodeid,
                       ArrayView<const index_t, 2> node_cellid,
                       ArrayView<std::int8_t, 1> b_visited,
                       ArrayView<index_t, 1> b_ncellid);

// Batch 5: halo ghost tables --------------------------------------------------

// _create_halo_ghost_tables: unpacks the flat-encoded `cell_halophyid` /
// `node_halophyid` (format: [id1, size1, val1_1, ..., val1_size1, id2,
// size2, ...] -- NOT the 2D per-row tables Batch 4 uses under the same
// names) into cell_haloghostid/node_haloghostid, each row's entries being
// indices into ext_ghost_info_int (i.e. halo-physical-face ids). While
// walking node_halophyid, also patches ext_ghost_info_int(phy_id, 0) in
// place: it's resolved here to the local halo-cell index found via
// search_halo_cell (domain_helpers.hpp), using ext_ghost_info_int(phy_id, 2)
// (a global cell id) as the lookup key.
void create_halo_ghost_tables(ArrayView<index_t, 2> ext_ghost_info_int,
                               ArrayView<const index_t, 1> node_halophyid,
                               ArrayView<const index_t, 1> cell_halophyid,
                               ArrayView<const index_t, 2> node_haloid,
                               ArrayView<const index_t, 2> halo_halosext,
                               ArrayView<index_t, 2> cell_haloghostid,
                               ArrayView<index_t, 2> node_haloghostid);

// Batch 6: parallel cell-cell connectivity -----------------------------------

// _create_cellfid: cell->neighbor-cell (by shared face) table, computed
// directly per cell without building the faces/cell_faceid/face_cellid
// tables create_info does. Parallelized with OpenMP (`#pragma omp parallel
// for` in the .cpp, silently sequential if built without OpenMP): each
// cell only ever writes cell_cellfid(i, ...) and only reads const arrays,
// so cells are independent -- matching the Python original's
// numba.prange. Scratch (bounded by the max cell faces/face nodes any
// create_cell_faces cell type needs) is allocated fresh per cell inside
// the loop rather than taken as a parameter, so there's nothing shared to
// race on. cell_cellfid must already be zeroed.
void create_cellfid(ArrayView<const index_t, 2> cells,
                     ArrayView<const index_t, 2> node_cellid,
                     ArrayView<const std::int8_t, 1> cell_type,
                     ArrayView<index_t, 2> cell_cellfid);

// Batch 7: face naming --------------------------------------------------------

// _define_node_oldname: for each node touched by a physical face, the
// smallest physical-face name among all physical faces touching it (0
// means "not yet set" as well as "not a boundary node", matching the
// Python original's `== 0` check).
void define_node_oldname(ArrayView<const index_t, 2> phy_faces,
                          ArrayView<const index_t, 1> phy_faces_name,
                          ArrayView<index_t, 1> node_oldname);

// _define_face_name: resolves every face to its physical-face id (via
// Batch 0's get_phyid) and propagates that physical face's name. A face on
// a halo boundary (face_haloid(i) != -1, only checked if face_haloid is
// non-empty) is always named 10, overriding the physical name. phy_faces is
// non-const for the same reason get_phyid's is (see domain_helpers.hpp).
void define_face_name(ArrayView<index_t, 2> phy_faces,
                       ArrayView<const index_t, 1> phy_faces_name,
                       ArrayView<const index_t, 2> faces,
                       ArrayView<const index_t, 2> node_phyfaceid,
                       ArrayView<const index_t, 1> face_haloid,
                       ArrayView<index_t, 1> face_oldname,
                       ArrayView<index_t, 1> face_name,
                       ArrayView<index_t, 1> phyid_to_faceid,
                       ArrayView<index_t, 1> face_to_phyid);

// Batch 8: halo cells ---------------------------------------------------------

// _create_halo_cells: unpacks the flat-encoded `node_halos` ([node_id,
// halo_id, node_id, halo_id, ...]) into node_haloid (node_haloid must
// already be zeroed), then for every face intersects its nodes'
// node_haloid rows (via Batch 0's intersect_common) to find the single halo
// cell the face borders, if any (-1 written to face_haloid otherwise), then
// for every cell collects the union of its nodes' halo neighbors into
// cell_halonid (deduplicated via Batch 0's is_in_array; must already be
// zeroed). b_visited is scratch, sized to node_haloid's halo-id range.
void create_halo_cells(ArrayView<const index_t, 2> cells,
                        ArrayView<const index_t, 2> faces,
                        ArrayView<const index_t, 1> node_halos,
                        ArrayView<index_t, 2> node_haloid,
                        ArrayView<std::int8_t, 1> b_visited,
                        ArrayView<index_t, 2> cell_halonid,
                        ArrayView<index_t, 1> face_haloid);

// Batch 9: face-gradient diamond geometry -------------------------------------

// _face_gradient_info_2d: per-face "diamond scheme" geometry for a
// Green-Gauss-style gradient at face midpoints on a 2D mesh -- the same
// f1..f4/param1..param4/air_diamond quantities src/core's face_gradient_2d
// consumes (see src/core/common/face_gradient_2d_common.hpp), but computed
// here from raw mesh data (nodes, cell centers, ghost/halo/periodic info)
// rather than taken as precomputed input. v_2 (the "right" point of the
// diamond) is cell_center(c_right) for an interior face (face_name == 0),
// shifted by cell_shift for a periodic face (11/22/33/44), halo_centvol for
// a halo face (face_name == 10), or ghost_info_flt for a physical
// (boundary) face (face_to_phyid(i) != -1) -- if none apply, v_2 stays 0,
// matching the Python original (no explicit "else" there).
void face_gradient_info_2d(ArrayView<const index_t, 2> face_cellid,
                            ArrayView<const index_t, 2> faces,
                            ArrayView<const index_t, 1> face_to_phyid,
                            ArrayView<const real_t, 2> ghost_info_flt,
                            ArrayView<const index_t, 1> face_name,
                            ArrayView<const real_t, 2> face_normal,
                            ArrayView<const real_t, 2> cell_center,
                            ArrayView<const real_t, 2> halo_centvol,
                            ArrayView<const index_t, 1> face_haloid,
                            ArrayView<const real_t, 2> nodes,
                            ArrayView<real_t, 1> face_air_diamond,
                            ArrayView<real_t, 1> face_param1,
                            ArrayView<real_t, 1> face_param2,
                            ArrayView<real_t, 1> face_param3,
                            ArrayView<real_t, 1> face_param4,
                            ArrayView<real_t, 2> face_f1,
                            ArrayView<real_t, 2> face_f2,
                            ArrayView<real_t, 2> face_f3,
                            ArrayView<real_t, 2> face_f4,
                            ArrayView<const real_t, 2> cell_shift);

// _face_gradient_info_3d: the 3D counterpart of face_gradient_info_2d (also
// feeds src/core's face_gradient_3d). Unlike the 2D version, v_2 has no
// implicit zero fallback: an unresolved face_name/face_to_phyid, or a
// degenerate (zero-area) diamond, throws -- matching the Python original's
// explicit RuntimeErrors.
void face_gradient_info_3d(ArrayView<const index_t, 2> face_cellid,
                            ArrayView<const index_t, 2> faces,
                            ArrayView<const index_t, 1> face_to_phyid,
                            ArrayView<const real_t, 2> ghost_info_flt,
                            ArrayView<const index_t, 1> face_name,
                            ArrayView<const real_t, 2> face_normal,
                            ArrayView<const real_t, 2> cell_center,
                            ArrayView<const real_t, 2> halo_centvol,
                            ArrayView<const index_t, 1> face_haloid,
                            ArrayView<const real_t, 2> nodes,
                            ArrayView<real_t, 1> face_air_diamond,
                            ArrayView<real_t, 1> face_param1,
                            ArrayView<real_t, 1> face_param2,
                            ArrayView<real_t, 1> face_param3,
                            ArrayView<real_t, 2> face_f1,
                            ArrayView<real_t, 2> face_f2,
                            ArrayView<const real_t, 2> cell_shift);

// Batch 10: FV face geometry --------------------------------------------------

// _fv_face_geometry: per-face coefficients for a finite-volume-style
// gradient scheme (distinct from Batch 9's diamond scheme): fv_coeff
// (|n|^2 / |n.d|) scales the normal-direction term; fv_corrx/y/z is the
// non-orthogonal correction vector n - (|n|^2/(n.d)) d; fv_weight_left is
// the left-cell interpolation weight for reconstructing a face value from
// its neighbors. d is the vector from the left cell center to a "right"
// point resolved via face_name the same way as face_gradient_info_2d/3d
// (interior/periodic/halo) -- but with no ghost/face_to_phyid branch: a
// face matching none of those names falls through with the right point
// left at the face's own center and fv_weight_left forced to 1. Throws if
// n.d == 0 (face normal orthogonal to the left-to-right direction).
void fv_face_geometry(ArrayView<const index_t, 2> face_cellid,
                       ArrayView<const index_t, 1> face_name,
                       ArrayView<const real_t, 2> face_normal,
                       ArrayView<const real_t, 2> face_center,
                       ArrayView<const index_t, 1> face_haloid,
                       ArrayView<const real_t, 2> cell_center,
                       ArrayView<const real_t, 2> halo_centvol,
                       ArrayView<const real_t, 2> cell_shift,
                       ArrayView<real_t, 1> fv_coeff,
                       ArrayView<real_t, 1> fv_corrx,
                       ArrayView<real_t, 1> fv_corry,
                       ArrayView<real_t, 1> fv_corrz,
                       ArrayView<real_t, 1> fv_weight_left);

// Batch 11: node-based least-squares variables --------------------------------

// _variables_2d: per-node least-squares gradient-interpolation weights.
// For every node, accumulates a 2x2 moment matrix (I_xx, I_yy, I_xy) and a
// moment vector (node_R_x, node_R_y) from every neighboring cell center,
// ghost, periodic image, halo-ghost and halo cell relative to that node,
// then solves the resulting 2x2 system in closed form for
// node_lambda_x/y -- the weights src/core's center_to_vertex_2d uses to
// interpolate a cell field onto the mesh vertices. The periodic branch
// (node_oldname(i) >= 11) applies the FULL cell_shift vector to every
// node_periodicid partner cell -- each partner already carries its own
// correctly-signed shift (zero on the components it isn't periodic in,
// see pair_periodic_faces), so this one branch also images a corner node's
// partners from more than one periodic direction correctly, without
// picking an axis based on node_oldname. node_R_x, node_R_y and
// node_number are accumulators: they must start zeroed. Throws if the
// moment matrix is singular (D == 0).
void variables_2d(ArrayView<const real_t, 2> cell_center,
                   ArrayView<const index_t, 2> node_cellid,
                   ArrayView<const index_t, 2> node_haloid,
                   ArrayView<const index_t, 2> node_ghostid,
                   ArrayView<const index_t, 2> node_haloghostid,
                   ArrayView<const index_t, 2> node_periodicid,
                   ArrayView<const real_t, 2> nodes,
                   ArrayView<const index_t, 1> node_oldname,
                   ArrayView<const real_t, 2> ghost_info_flt,
                   ArrayView<const real_t, 2> ext_ghost_info_flt,
                   ArrayView<const real_t, 2> halo_centvol,
                   ArrayView<real_t, 1> node_R_x, ArrayView<real_t, 1> node_R_y,
                   ArrayView<real_t, 1> node_lambda_x,
                   ArrayView<real_t, 1> node_lambda_y,
                   ArrayView<index_t, 1> node_number,
                   ArrayView<const real_t, 2> cell_shift);

// _variables_3d: the 3D counterpart of variables_2d, accumulating a 3x3
// moment matrix (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz) and moment vector
// (node_R_x/y/z), then solving via the closed-form cofactor/adjugate
// expressions for a 3x3 system, for node_lambda_x/y/z. Same unified
// periodic branch as variables_2d (node_oldname(i) >= 11, full cell_shift
// vector per node_periodicid partner), extended to the third periodic-name
// pair (55/66, z-direction shift). node_R_x/y/z and node_number are
// accumulators: they must start zeroed. Throws if the moment matrix is
// singular (D == 0).
void variables_3d(ArrayView<const real_t, 2> cell_center,
                   ArrayView<const index_t, 2> node_cellid,
                   ArrayView<const index_t, 2> node_haloid,
                   ArrayView<const index_t, 2> node_ghostid,
                   ArrayView<const index_t, 2> node_haloghostid,
                   ArrayView<const index_t, 2> node_periodicid,
                   ArrayView<const real_t, 2> nodes,
                   ArrayView<const index_t, 1> node_oldname,
                   ArrayView<const real_t, 2> ghost_info_flt,
                   ArrayView<const real_t, 2> ext_ghost_info_flt,
                   ArrayView<const real_t, 2> halo_centvol,
                   ArrayView<real_t, 1> node_R_x, ArrayView<real_t, 1> node_R_y,
                   ArrayView<real_t, 1> node_R_z,
                   ArrayView<real_t, 1> node_lambda_x,
                   ArrayView<real_t, 1> node_lambda_y,
                   ArrayView<real_t, 1> node_lambda_z,
                   ArrayView<index_t, 1> node_number,
                   ArrayView<const real_t, 2> cell_shift);

// Batch 12: misc geometry -----------------------------------------------------

// _create_normal_face_of_cell: outward-oriented copy of face_normal for
// every (cell, local face) pair -- cell_nf(i, j, :) is face_normal(fid, :)
// flipped if needed so it points away from cell i's center, where
// fid = cell_faceid(i, j). (distance_2d was dropped from this port's
// scope.)
void create_normal_face_of_cell(ArrayView<const real_t, 2> cell_center,
                                 ArrayView<const real_t, 2> face_center,
                                 ArrayView<const index_t, 2> cell_faceid,
                                 ArrayView<const real_t, 2> face_normal,
                                 ArrayView<real_t, 3> cell_nf);

// _dist_ortho_function_2d: per-face orthogonal distance used by some FV
// diffusion schemes. d_boundaryfaces/d_innerfaces are gather lists of face
// indices (not full-range counters), matching the d_innerfaces/etc.
// convention in src/core's face_gradient_2d. For a boundary face, it's
// twice the distance from the owning cell center to its orthogonal
// projection onto the face-normal line through the face center; for an
// interior face, it's the sum of that projection distance from each side's
// cell center. Only face_dist_ortho(bf) for bf in d_boundaryfaces/
// d_innerfaces is written -- not the full array.
void dist_ortho_function_2d(ArrayView<const index_t, 1> d_innerfaces,
                             ArrayView<const index_t, 1> d_boundaryfaces,
                             ArrayView<const index_t, 2> face_cellid,
                             ArrayView<const real_t, 2> cell_center,
                             ArrayView<const real_t, 2> face_center,
                             ArrayView<const real_t, 2> face_normal,
                             ArrayView<real_t, 1> face_dist_ortho);

// Post-port addition: periodic boundary connectivity ------------------------
// Not part of to_convert.py's original 39; added later to build
// cell_shift/node_periodicid (consumed by variables_2d/variables_3d) in C++
// instead of Python. See Steps.md's "Post-port addition" section.

// _pair_periodic_faces: same-rank periodic face pairing. Matches faces
// tagged name_lo (owner shift +L on component saxis) to name_hi (shift -L)
// by their transverse coordinate(s) taxis0[,taxis1], and wires
// face_cellid(.,1) + cell_shift. Matching is sort-based (no dict): a unique
// integer key is built from the rounded transverse coords, both sides are
// sorted by that key, and paired in order. Returns nlo on success, -1 if
// the two sides have different counts (cross-rank leftover / malformed),
// -2 if a transverse key has no match.
index_t pair_periodic_faces(ArrayView<const index_t, 1> face_name,
                             ArrayView<const real_t, 2> face_center,
                             ArrayView<index_t, 2> face_cellid,
                             ArrayView<real_t, 2> cell_shift,
                             ArrayView<const real_t, 1> cmin, index_t name_lo,
                             index_t name_hi, index_t taxis0, index_t taxis1,
                             index_t saxis, real_t L, real_t dtol);

// _node_periodic_bits: per-node bitmask of the periodic boundaries the node
// lies on, taken from the periodic faces it is a vertex of. This is what
// lets an EDGE/CORNER node be matched in every periodic direction it
// touches (node_oldname carries only one tag and cannot express that).
// Bits: 1=11(x-lo) 2=22(x-hi) 4=33(y-hi) 8=44(y-lo) 16=55(z-lo) 32=66(z-hi).
void node_periodic_bits(ArrayView<const index_t, 2> faces,
                         ArrayView<const index_t, 1> face_name,
                         ArrayView<index_t, 1> node_bits);

// _accum_periodic_dir: for ONE periodic axis, match the boundary nodes
// carrying lo_bit to those carrying hi_bit by their transverse
// coordinate(s), and APPEND each side's partner cells into
// node_periodicid (per-node running counter node_fill). Called once per
// periodic axis, so an edge/corner node (which carries several bits, see
// node_periodic_bits) accumulates partners from every direction. Tolerant
// two-pointer merge over the two sides' sorted keys: unmatched nodes
// (cross-rank) are silently left unpaired here, for the halo branch to
// handle.
void accum_periodic_dir(ArrayView<const index_t, 1> node_bits,
                         ArrayView<const real_t, 2> nodes,
                         ArrayView<const index_t, 2> node_cellid,
                         ArrayView<index_t, 2> node_periodicid,
                         ArrayView<index_t, 1> node_fill,
                         ArrayView<const real_t, 1> cmin, index_t lo_bit,
                         index_t hi_bit, index_t taxis0, index_t taxis1,
                         real_t dtol);

// Cell centroids and areas/volumes -----------------------------------------
//
// Ported from the manapy c_api's compute_cell_center_volume.cpp. Element
// geometry belongs with the rest of the mesh geometry here rather than with
// the partitioner that first needed it; src/partitioning compiles this same
// translation unit for the two halo_* entry points, which it calls while
// building each subdomain's halo tables (they are not bound to Python).
//
// Supported shapes are the ones CELL_TYPE enumerates: triangle/quad in 2D,
// tetrahedron/pyramid/hexahedron in 3D. The element count comes from the row
// data, not the row width -- `cells`' last column holds the node count, and
// halo_halosext's rows are [global_id, node0, ..., nodeN, count], so their
// vertex count is that last column minus one.
//
// Intermediate arithmetic is done in double regardless of real_t: these
// centroid sums and signed-volume determinants lose significance badly in
// float32, and the cost is negligible next to the array traffic.

// Centroid and area of every 2D cell. Writes cell_area (n_cells) and
// cell_center (n_cells, >=2) in place.
void compute_cell_center_area_2d(ArrayView<const index_t, 2> cells,
                                  ArrayView<const real_t, 2> nodes,
                                  ArrayView<real_t, 1> cell_area,
                                  ArrayView<real_t, 2> cell_center);

// Centroid and volume of every 3D cell. Writes cell_volume (n_cells) and
// cell_center (n_cells, >=3) in place.
void compute_cell_center_volume_3d(ArrayView<const index_t, 2> cells,
                                    ArrayView<const real_t, 2> nodes,
                                    ArrayView<real_t, 1> cell_volume,
                                    ArrayView<real_t, 2> cell_center);

// Same as compute_cell_center_area_2d, for exterior halo cells: reads the
// [global_id, nodes..., count] rows of halo_halosext and writes the packed
// halo_centvol (n_halos, 4) = [center_x, center_y, center_z, area].
void compute_halo_cell_center_area_2d(ArrayView<const index_t, 2> halo_halosext,
                                       ArrayView<const real_t, 2> nodes,
                                       ArrayView<real_t, 2> halo_centvol);

// Same as compute_cell_center_volume_3d, for exterior halo cells; writes
// halo_centvol (n_halos, 4) = [center_x, center_y, center_z, volume].
void compute_halo_cell_center_volume_3d(
    ArrayView<const index_t, 2> halo_halosext,
    ArrayView<const real_t, 2> nodes, ArrayView<real_t, 2> halo_centvol);

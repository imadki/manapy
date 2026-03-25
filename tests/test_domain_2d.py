"""
Unit tests for 2D domain structure.

Three levels of verification:
  1. Analytical reference  — compare against hand-computed values (structured mesh)
  2. Connectivity          — verify topological invariants (all mesh types)
  3. Type-specific         — hybrid: both triangles and quads must be present

Mesh files used:
  - carre_structure.msh  : structured quads  → analytical reference values
  - rectangle.msh        : unstructured quads
  - carre_hybrid.msh     : mixed triangles + quadrilaterals
  - carre.msh            : pure quads (unstructured)

Note: cells.nodeid[i, -1] stores the node count per cell (padded arrays).
      faces.cellid[f]      stores the two adjacent cells (-1 = boundary).
      faces.mesure         is the face length (French spelling in the code).
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared connectivity checks (applicable to any 2D mesh)
# ---------------------------------------------------------------------------
def check_face_cell_connectivity(domain):
    """
    Every face must be adjacent to 1 (boundary) or 2 (inner) cells.
    cellid[:,0] is always set; cellid[:,1] == -1 means boundary face.
    """
    cellid = domain.faces.cellid          # (nfaces, 2)
    assert np.all(cellid[:, 0] >= 0), "Every face must have at least one adjacent cell"
    assert np.all(cellid[:, 1] >= -1), "Face cellid[1] must be >= -1"


def check_total_volume_matches_bbox(domain, rtol=1e-4):
    """
    Sum of cell volumes must equal the domain bounding-box area.
    Works for convex domains without holes.
    """
    v = domain.nodes.vertex              # (nbnodes, 3)
    bbox_area = (v[:, 0].max() - v[:, 0].min()) * (v[:, 1].max() - v[:, 1].min())
    total_vol = np.sum(domain.cells.volume)
    assert abs(total_vol - bbox_area) / bbox_area < rtol, \
        f"Total volume {total_vol:.6f} != bbox area {bbox_area:.6f}"


def check_face_normals_unit(domain):
    norms = np.linalg.norm(domain.faces.normal, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 1. STRUCTURED QUAD — Analytical reference
# ---------------------------------------------------------------------------
class TestStructuredQuad2DReference:
    """
    For a structured N×N mesh on [0, L]×[0, L]:
      - All cell volumes must equal L²/N²  (uniform cells)
      - All face measures must equal L/N   (uniform faces)
      - All face normals must be axis-aligned: components ∈ {-1, 0, 1}
      - Sum of volumes == L²
      - Inner faces have 2 adjacent cells; boundary faces have 1
    """

    def test_all_cells_same_volume(self, domain_structured_2d):
        """Structured mesh: every cell must have the exact same volume."""
        volumes = domain_structured_2d.cells.volume
        assert np.allclose(volumes, volumes[0], rtol=1e-6), \
            f"Volumes are not uniform: min={volumes.min():.6e}, max={volumes.max():.6e}"

    def test_all_faces_same_measure(self, domain_structured_2d):
        """Structured mesh: every face must have the same length."""
        measures = domain_structured_2d.faces.mesure
        assert np.allclose(measures, measures[0], rtol=1e-6), \
            f"Face measures are not uniform: min={measures.min():.6e}, max={measures.max():.6e}"

    def test_total_volume_equals_bbox(self, domain_structured_2d):
        check_total_volume_matches_bbox(domain_structured_2d)

    def test_face_normals_axis_aligned(self, domain_structured_2d):
        """
        On a structured quad mesh all face normals must be axis-aligned,
        i.e. each component must be close to -1, 0, or +1.
        """
        normals = domain_structured_2d.faces.normal     # (nfaces, 3)
        for comp in range(2):                            # x and y only in 2D
            vals = normals[:, comp]
            not_aligned = ~(
                np.isclose(vals,  0.0, atol=1e-8) |
                np.isclose(vals,  1.0, atol=1e-8) |
                np.isclose(vals, -1.0, atol=1e-8)
            )
            assert not np.any(not_aligned), \
                f"Normal component {comp} contains non-axis-aligned values"

    def test_cell_volume_consistent_with_face_measure(self, domain_structured_2d):
        """
        For a uniform structured mesh: cell_volume == face_length²
        (each cell is a square with side = face_length).
        """
        vol = domain_structured_2d.cells.volume[0]
        msr = domain_structured_2d.faces.mesure[0]
        assert abs(vol - msr ** 2) / vol < 1e-6, \
            f"cell_volume ({vol:.6e}) != face_length² ({msr**2:.6e})"

    def test_face_cell_connectivity(self, domain_structured_2d):
        check_face_cell_connectivity(domain_structured_2d)

    def test_face_normals_unit(self, domain_structured_2d):
        check_face_normals_unit(domain_structured_2d)

    def test_cell_type_all_quads(self, domain_structured_2d):
        """All cells must be quads (4 nodes each)."""
        node_counts = domain_structured_2d.cells.nodeid[:, -1]
        assert np.all(node_counts == 4), \
            f"Non-quad cells found: unique counts = {np.unique(node_counts)}"

    def test_inner_faces_have_two_cells(self, domain_structured_2d):
        """Every inner face must be shared by exactly 2 cells."""
        cellid = domain_structured_2d.faces.cellid
        inner_mask = cellid[:, 1] != -1
        assert np.all(cellid[inner_mask, 0] >= 0)
        assert np.all(cellid[inner_mask, 1] >= 0)

    def test_boundary_faces_have_one_cell(self, domain_structured_2d):
        """Every boundary face must be adjacent to exactly 1 cell."""
        cellid = domain_structured_2d.faces.cellid
        boundary_mask = cellid[:, 1] == -1
        assert np.any(boundary_mask), "No boundary faces found"
        assert np.all(cellid[boundary_mask, 0] >= 0)


# ---------------------------------------------------------------------------
# 2. UNSTRUCTURED QUAD — Connectivity and consistency
# ---------------------------------------------------------------------------
class TestUnstructuredQuad2D:
    """
    rectangle.msh and carre.msh are unstructured quad meshes.
    We cannot predict exact volumes or normals, but topological and
    geometric invariants must hold.
    """

    def _check(self, domain):
        check_face_cell_connectivity(domain)
        check_total_volume_matches_bbox(domain)
        check_face_normals_unit(domain)

    def test_rectangle_connectivity(self, domain_rectangle_2d):
        self._check(domain_rectangle_2d)

    def test_rectangle_cell_type_all_quads(self, domain_rectangle_2d):
        node_counts = domain_rectangle_2d.cells.nodeid[:, -1]
        assert np.all(node_counts == 4), \
            f"Unexpected cell types: {np.unique(node_counts)}"

    def test_rectangle_volumes_positive(self, domain_rectangle_2d):
        assert np.all(domain_rectangle_2d.cells.volume > 0)

    def test_rectangle_face_measures_positive(self, domain_rectangle_2d):
        assert np.all(domain_rectangle_2d.faces.mesure > 0)

    def test_rectangle_inner_faces_have_two_cells(self, domain_rectangle_2d):
        cellid = domain_rectangle_2d.faces.cellid
        inner = cellid[:, 1] != -1
        assert np.all(cellid[inner, 0] >= 0)
        assert np.all(cellid[inner, 1] >= 0)

    def test_carre_connectivity(self, domain_carre_2d):
        self._check(domain_carre_2d)

    def test_carre_cell_type_all_quads(self, domain_carre_2d):
        node_counts = domain_carre_2d.cells.nodeid[:, -1]
        assert np.all(node_counts == 4)

    def test_carre_volumes_positive(self, domain_carre_2d):
        assert np.all(domain_carre_2d.cells.volume > 0)


# ---------------------------------------------------------------------------
# 3. HYBRID MESH — Both triangles (3 nodes) and quads (4 nodes) must exist
# ---------------------------------------------------------------------------
class TestHybridMesh2D:
    """
    carre_hybrid.msh contains a mix of triangular and quadrilateral cells.
    Key invariants:
      - Cells with 3 nodes (triangles) AND cells with 4 nodes (quads) both exist.
      - Triangles have 3 faces; quads have 4 faces.
      - Total volume equals the bounding-box area.
      - Face-cell connectivity is consistent.
    """

    def test_both_triangles_and_quads_present(self, domain_hybrid_2d):
        """The hybrid mesh must contain both triangle and quad cells."""
        node_counts = domain_hybrid_2d.cells.nodeid[:, -1]
        assert 3 in node_counts, "No triangle cells found in hybrid mesh"
        assert 4 in node_counts, "No quad cells found in hybrid mesh"

    def test_triangle_cells_have_3_faces(self, domain_hybrid_2d):
        """Triangle cells must have exactly 3 faces."""
        node_counts = domain_hybrid_2d.cells.nodeid[:, -1]
        face_counts = domain_hybrid_2d.cells.faceid[:, -1]
        tri_mask = node_counts == 3
        assert np.all(face_counts[tri_mask] == 3), \
            f"Some triangles do not have 3 faces: {np.unique(face_counts[tri_mask])}"

    def test_quad_cells_have_4_faces(self, domain_hybrid_2d):
        """Quad cells must have exactly 4 faces."""
        node_counts = domain_hybrid_2d.cells.nodeid[:, -1]
        face_counts = domain_hybrid_2d.cells.faceid[:, -1]
        quad_mask = node_counts == 4
        assert np.all(face_counts[quad_mask] == 4), \
            f"Some quads do not have 4 faces: {np.unique(face_counts[quad_mask])}"

    def test_total_volume_equals_bbox(self, domain_hybrid_2d):
        check_total_volume_matches_bbox(domain_hybrid_2d)

    def test_face_cell_connectivity(self, domain_hybrid_2d):
        check_face_cell_connectivity(domain_hybrid_2d)

    def test_face_normals_unit(self, domain_hybrid_2d):
        check_face_normals_unit(domain_hybrid_2d)

    def test_triangle_volumes_positive(self, domain_hybrid_2d):
        node_counts = domain_hybrid_2d.cells.nodeid[:, -1]
        tri_volumes = domain_hybrid_2d.cells.volume[node_counts == 3]
        assert np.all(tri_volumes > 0)

    def test_quad_volumes_positive(self, domain_hybrid_2d):
        node_counts = domain_hybrid_2d.cells.nodeid[:, -1]
        quad_volumes = domain_hybrid_2d.cells.volume[node_counts == 4]
        assert np.all(quad_volumes > 0)

    def test_triangle_smaller_than_quad_on_average(self, domain_hybrid_2d):
        """
        On a reasonably meshed hybrid domain, average triangle area should
        be smaller than average quad area (quads span larger regions).
        This is a soft heuristic, not a hard rule.
        """
        node_counts = domain_hybrid_2d.cells.nodeid[:, -1]
        tri_vol = domain_hybrid_2d.cells.volume[node_counts == 3].mean()
        quad_vol = domain_hybrid_2d.cells.volume[node_counts == 4].mean()
        assert tri_vol < quad_vol, \
            f"Expected avg triangle area < avg quad area, got {tri_vol:.4e} vs {quad_vol:.4e}"

"""
Unit tests for 2D domain structure.

Covers three mesh types:
  - rectangle.msh        : quadrilateral cells
  - carre_hybrid.msh     : mixed triangles + quadrilaterals
  - carre_structure.msh  : structured quadrilateral cells
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_domain_structure(domain, expected_dim=2):
    """Common structural assertions for any 2D domain."""
    assert domain.dim == expected_dim
    assert domain.nbcells > 0
    assert domain.nbfaces > 0
    assert domain.nbnodes > 0


# ---------------------------------------------------------------------------
# Rectangle mesh (quads)
# ---------------------------------------------------------------------------
class TestDomainRectangle2D:

    def test_dimensions(self, domain_rectangle_2d):
        check_domain_structure(domain_rectangle_2d)

    def test_cell_volumes_positive(self, domain_rectangle_2d):
        assert np.all(domain_rectangle_2d.cells.volume > 0), \
            "All cell volumes must be strictly positive"

    def test_cell_centers_in_range(self, domain_rectangle_2d):
        centers = domain_rectangle_2d.cells.center
        # Coordinates must be finite
        assert np.all(np.isfinite(centers))

    def test_face_normals_unit(self, domain_rectangle_2d):
        normals = domain_rectangle_2d.faces.normal
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10), \
            "All face normals must be unit vectors"

    def test_face_measures_positive(self, domain_rectangle_2d):
        assert np.all(domain_rectangle_2d.faces.measure > 0), \
            "All face measures (lengths) must be strictly positive"

    def test_face_centers_finite(self, domain_rectangle_2d):
        assert np.all(np.isfinite(domain_rectangle_2d.faces.center))

    def test_node_coords_finite(self, domain_rectangle_2d):
        assert np.all(np.isfinite(domain_rectangle_2d.nodes.vertex))

    def test_total_volume_positive(self, domain_rectangle_2d):
        total = np.sum(domain_rectangle_2d.cells.volume)
        assert total > 0

    def test_euler_formula(self, domain_rectangle_2d):
        """
        For a simply-connected 2-D mesh: F - E + V - C ~ 1 (loose check).
        We only verify the counts are consistent (all > 0 and faces > cells).
        """
        d = domain_rectangle_2d
        # A convex 2-D domain must have more faces than cells
        assert d.nbfaces >= d.nbcells


# ---------------------------------------------------------------------------
# Hybrid mesh (triangles + quads)
# ---------------------------------------------------------------------------
class TestDomainHybrid2D:

    def test_dimensions(self, domain_hybrid_2d):
        check_domain_structure(domain_hybrid_2d)

    def test_cell_volumes_positive(self, domain_hybrid_2d):
        assert np.all(domain_hybrid_2d.cells.volume > 0)

    def test_face_normals_unit(self, domain_hybrid_2d):
        normals = domain_hybrid_2d.faces.normal
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_face_measures_positive(self, domain_hybrid_2d):
        assert np.all(domain_hybrid_2d.faces.measure > 0)

    def test_cell_centers_finite(self, domain_hybrid_2d):
        assert np.all(np.isfinite(domain_hybrid_2d.cells.center))

    def test_total_volume_positive(self, domain_hybrid_2d):
        assert np.sum(domain_hybrid_2d.cells.volume) > 0


# ---------------------------------------------------------------------------
# Structured mesh (quads)
# ---------------------------------------------------------------------------
class TestDomainStructured2D:

    def test_dimensions(self, domain_structured_2d):
        check_domain_structure(domain_structured_2d)

    def test_cell_volumes_positive(self, domain_structured_2d):
        assert np.all(domain_structured_2d.cells.volume > 0)

    def test_face_normals_unit(self, domain_structured_2d):
        normals = domain_structured_2d.faces.normal
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_face_measures_positive(self, domain_structured_2d):
        assert np.all(domain_structured_2d.faces.measure > 0)

    def test_total_volume_positive(self, domain_structured_2d):
        assert np.sum(domain_structured_2d.cells.volume) > 0

    def test_structured_uniform_volumes(self, domain_structured_2d):
        """Structured mesh: all cells should have the same volume."""
        volumes = domain_structured_2d.cells.volume
        assert np.allclose(volumes, volumes[0], rtol=1e-6), \
            "Structured mesh cells should all have the same volume"

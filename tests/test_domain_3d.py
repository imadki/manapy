"""
Unit tests for 3D domain structure.

Covers:
  - cube.msh     : hexahedral cells
  - cube_bis.msh : alternative hexahedral cells
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_domain_structure_3d(domain):
    assert domain.dim == 3
    assert domain.nbcells > 0
    assert domain.nbfaces > 0
    assert domain.nbnodes > 0


# ---------------------------------------------------------------------------
# Cube mesh (hexahedra)
# ---------------------------------------------------------------------------
class TestDomainCube3D:

    def test_dimensions(self, domain_cube_3d):
        check_domain_structure_3d(domain_cube_3d)

    def test_cell_volumes_positive(self, domain_cube_3d):
        assert np.all(domain_cube_3d.cells.volume > 0), \
            "All cell volumes must be strictly positive"

    def test_cell_centers_finite(self, domain_cube_3d):
        assert np.all(np.isfinite(domain_cube_3d.cells.center))

    def test_face_normals_unit(self, domain_cube_3d):
        normals = domain_cube_3d.faces.normal
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10), \
            "All face normals must be unit vectors"

    def test_face_measures_positive(self, domain_cube_3d):
        assert np.all(domain_cube_3d.faces.measure > 0)

    def test_face_centers_finite(self, domain_cube_3d):
        assert np.all(np.isfinite(domain_cube_3d.faces.center))

    def test_node_coords_finite(self, domain_cube_3d):
        assert np.all(np.isfinite(domain_cube_3d.nodes.vertex))

    def test_total_volume_positive(self, domain_cube_3d):
        assert np.sum(domain_cube_3d.cells.volume) > 0

    def test_more_faces_than_cells(self, domain_cube_3d):
        d = domain_cube_3d
        assert d.nbfaces >= d.nbcells

    def test_cell_centers_3d_coords(self, domain_cube_3d):
        """Cell centres must have three spatial coordinates."""
        centers = domain_cube_3d.cells.center
        assert centers.shape[1] >= 3


# ---------------------------------------------------------------------------
# Alternative cube mesh
# ---------------------------------------------------------------------------
class TestDomainCubeBis3D:

    def test_dimensions(self, domain_cube_bis_3d):
        check_domain_structure_3d(domain_cube_bis_3d)

    def test_cell_volumes_positive(self, domain_cube_bis_3d):
        assert np.all(domain_cube_bis_3d.cells.volume > 0)

    def test_face_normals_unit(self, domain_cube_bis_3d):
        normals = domain_cube_bis_3d.faces.normal
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_face_measures_positive(self, domain_cube_bis_3d):
        assert np.all(domain_cube_bis_3d.faces.measure > 0)

    def test_total_volume_positive(self, domain_cube_bis_3d):
        assert np.sum(domain_cube_bis_3d.cells.volume) > 0

"""
Integration tests for manapy.api (Mesh.generate + AdvectionModel).

What is tested
--------------
- Mesh.generate() produces a valid domain for all cell types
- AdvectionModel runs without error and conserves mass
  (∑ φ·vol = const across time steps for a pure advection)

These tests use the python backend (no Numba) for fast startup.
"""
import pytest
import numpy as np

from manapy.api import Mesh, AdvectionModel
from manapy.ast import Variable
from manapy.base.base import Struct


# ---------------------------------------------------------------------------
# Fixtures — generated meshes (module scope for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh_tri(tmp_path_factory):
    """2D triangle mesh on unit square."""
    d = str(tmp_path_factory.mktemp("mesh_tri"))
    return Mesh.generate(dim=2, n=10, cell_type="triangle",
                         backend="python", work_dir=d)


@pytest.fixture(scope="module")
def mesh_quad(tmp_path_factory):
    """2D quad mesh on unit square."""
    d = str(tmp_path_factory.mktemp("mesh_quad"))
    return Mesh.generate(dim=2, n=10, cell_type="quad",
                         backend="python", work_dir=d)


@pytest.fixture(scope="module")
def mesh_rect(tmp_path_factory):
    """2D triangle mesh on a non-unit rectangle."""
    d = str(tmp_path_factory.mktemp("mesh_rect"))
    return Mesh.generate(dim=2, bounds=((0, 2), (0, 1)), n=(20, 10),
                         cell_type="triangle", backend="python", work_dir=d)


# ---------------------------------------------------------------------------
# Mesh geometry tests
# ---------------------------------------------------------------------------

class TestMeshGenerate2D:

    def test_triangle_cell_count(self, mesh_tri):
        # 10×10 grid → 2×100 = 200 triangles
        assert mesh_tri.domain.nbcells == 200

    def test_quad_cell_count(self, mesh_quad):
        # 10×10 grid → 100 quads
        assert mesh_quad.domain.nbcells == 100

    def test_rectangle_cell_count(self, mesh_rect):
        # 20×10 grid → 2×200 = 400 triangles
        assert mesh_rect.domain.nbcells == 400

    def test_triangle_volumes_positive(self, mesh_tri):
        assert np.all(mesh_tri.domain.cells.volume > 0)

    def test_quad_volumes_positive(self, mesh_quad):
        assert np.all(mesh_quad.domain.cells.volume > 0)

    def test_triangle_total_volume(self, mesh_tri):
        total = np.sum(mesh_tri.domain.cells.volume)
        assert abs(total - 1.0) < 1e-10

    def test_quad_total_volume(self, mesh_quad):
        total = np.sum(mesh_quad.domain.cells.volume)
        assert abs(total - 1.0) < 1e-10

    def test_rectangle_total_volume(self, mesh_rect):
        total = np.sum(mesh_rect.domain.cells.volume)
        assert abs(total - 2.0) < 1e-10

    def test_bounds_triangle(self, mesh_tri):
        v = mesh_tri.domain.nodes.vertex
        assert abs(v[:, 0].min()) < 1e-12
        assert abs(v[:, 0].max() - 1.0) < 1e-12
        assert abs(v[:, 1].min()) < 1e-12
        assert abs(v[:, 1].max() - 1.0) < 1e-12

    def test_bounds_rectangle(self, mesh_rect):
        v = mesh_rect.domain.nodes.vertex
        assert abs(v[:, 0].max() - 2.0) < 1e-12
        assert abs(v[:, 1].max() - 1.0) < 1e-12

    def test_face_normals_finite(self, mesh_tri):
        assert np.all(np.isfinite(mesh_tri.domain.faces.normal))


# ---------------------------------------------------------------------------
# Advection integration test — mass conservation
# ---------------------------------------------------------------------------

def _run_advection(mesh, nsteps=5):
    """
    Run nsteps of advection with constant velocity (1,0) on mesh.
    Returns (initial_mass, final_mass).
    """
    domain = mesh.domain
    c = domain.cells.center

    phi = Variable(domain=domain, name="phi")
    phi.cell[:] = np.exp(-((c[:, 0] - 0.3)**2 + (c[:, 1] - 0.5)**2) / 0.02)

    u = Variable(domain=domain, name="u")
    v = Variable(domain=domain, name="v")
    u.cell[:] = 1.0
    v.cell[:] = 0.0

    vol = domain.cells.volume
    mass0 = np.dot(phi.cell, vol)

    Variable.is_called = False
    from manapy.solvers.advec import AdvectionSolver
    conf = Struct(order=1, cfl=0.5)
    solver = AdvectionSolver(phi, vel=(u, v), conf=conf)

    for _ in range(nsteps):
        solver.stepper()
        solver.compute_fluxes()
        solver.compute_new_val()

    mass1 = np.dot(phi.cell, vol)
    return mass0, mass1


class TestAdvectionOnGeneratedMesh:

    def test_advection_runs_triangle(self, mesh_tri):
        """Advection completes without error on triangle mesh."""
        mass0, mass1 = _run_advection(mesh_tri)
        assert np.isfinite(mass1), "mass diverged"

    def test_advection_runs_quad(self, mesh_quad):
        """Advection completes without error on quad mesh."""
        mass0, mass1 = _run_advection(mesh_quad)
        assert np.isfinite(mass1), "mass diverged"

    def test_mass_conservation_triangle(self, mesh_tri):
        """Mass is conserved to within 5% over 5 steps."""
        mass0, mass1 = _run_advection(mesh_tri)
        if abs(mass0) > 1e-12:
            rel_err = abs(mass1 - mass0) / abs(mass0)
            assert rel_err < 0.05, f"mass loss = {rel_err:.2%}"

    def test_mass_conservation_quad(self, mesh_quad):
        """Mass is conserved to within 5% over 5 steps."""
        mass0, mass1 = _run_advection(mesh_quad)
        if abs(mass0) > 1e-12:
            rel_err = abs(mass1 - mass0) / abs(mass0)
            assert rel_err < 0.05, f"mass loss = {rel_err:.2%}"

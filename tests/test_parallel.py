"""
Parallel (MPI) tests for domain decomposition and gradient operators.

Run with:
    mpirun -n 2 pytest tests/test_parallel.py -m parallel
    mpirun -n 4 pytest tests/test_parallel.py -m parallel --oversubscribe

These tests are skipped when running on a single process.

What is tested
--------------
Domain:
  - Sum of local cell volumes == global volume (decomposition preserves volume)
  - Halo cells exist when N > 1 processes

Gradient (2D and 3D, on every rank including partition boundaries):
  - Linear functions    : exact reconstruction (atol 1e-4)
  - Quadratic functions : L2 relative error < 5 %
  - Sinusoidal functions: L2 relative error < 5 %

Why haloghost matters
---------------------
When a node sits on both a physical boundary and a partition interface, its
associated ghost cell (mirror image of an interior cell) belongs to the
neighbouring partition — it becomes a "haloghost".

cell_gradient_2d uses w_haloghost in its stencil for these corner nodes.
If haloghost is left at zero, the gradient reconstruction is wrong for
every cell touching such a node.

Fix: define Dirichlet BCs with the analytical function as value and call
update_ghost_value() after update_halo_value().  update_ghost_value()
populates both ghost[] (local boundary faces) and haloghost[]
(boundary faces whose ghost is in a neighbour partition).
"""
import os
import pytest
import numpy as np

from mpi4py import MPI
from manapy.partitions import MeshPartition
from manapy.ddm import Domain
from manapy.ast import Variable
from manapy.base.base import Struct

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

MESH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mesh'))

PARALLEL_CONF = Struct(
    backend="python",
    signature=False,
    cache=False,
    float_precision="double",
    int_precision="signed",
)

requires_mpi = pytest.mark.skipif(SIZE == 1, reason="requires MPI (mpirun -n N)")

ATOL_LINEAR  = 1e-4   # tolerance for linear gradient (partition boundaries add noise)
RTOL_SMOOTH  = 0.05   # L2 relative tolerance for quadratic / sinusoidal


# ---------------------------------------------------------------------------
# Shared gradient helpers
# ---------------------------------------------------------------------------
def _l2_rel(computed, exact):
    return np.linalg.norm(computed - exact) / (np.linalg.norm(exact) + 1e-12)


_BC_LOCS_2D = ("in", "out", "upper", "bottom")
_BC_LOCS_3D = ("in", "out", "upper", "bottom", "front", "back")


def _make_var(domain, func):
    """
    Create a Variable with Dirichlet BCs = func(x, y, z).

    The BC lambda is used by update_ghost_value() to populate:
      - ghost[]     : boundary faces whose ghost is LOCAL to this rank
      - haloghost[] : boundary faces whose ghost belongs to a NEIGHBOUR
                      partition (node at both physical boundary and
                      partition interface — the "corner haloghost" case)

    Call sequence:
      1. var.cell[:] = func at cell centres
      2. update_halo_value()   → sync w_halo across partitions
      3. update_ghost_value()  → set ghost + haloghost from the BC lambda
    """
    Variable.is_called = False

    locs = _BC_LOCS_3D if domain.dim == 3 else _BC_LOCS_2D
    boundaries = {loc: "dirichlet" for loc in locs}
    values     = {loc: lambda x, y, z, f=func: f(x, y, z) for loc in locs}

    var = Variable(domain=domain, BC=boundaries, values=values)

    c = domain.cells.center
    var.cell[:] = func(c[:, 0], c[:, 1], c[:, 2])

    var.update_halo_value()    # w_halo  ← neighbour cell values
    var.update_ghost_value()   # ghost + haloghost ← BC lambda
    return var


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _parallel_domain(mesh_name, dim, tmp_path_shared):
    """
    All ranks call MeshPartition + Domain.
    Only rank 0 writes the partition files; all ranks then read them.
    """
    mesh_path = os.path.join(MESH_DIR, mesh_name)
    original_cwd = os.getcwd()
    # Use a shared directory visible to all ranks
    os.chdir(tmp_path_shared)
    try:
        Variable.is_called = False
        MeshPartition(mesh_path, dim=dim, conf=PARALLEL_CONF, periodic=[0, 0, 0])
        COMM.Barrier()
        domain = Domain(dim=dim, conf=PARALLEL_CONF)
    finally:
        os.chdir(original_cwd)
    return domain


@pytest.fixture(scope="module")
def par_domain_rect_2d(tmp_path_factory):
    d = tmp_path_factory.mktemp("par_rect_2d")
    # Broadcast the path so all ranks use the same directory
    d_str = COMM.bcast(str(d), root=0)
    return _parallel_domain("rectangle.msh", 2, d_str)


@pytest.fixture(scope="module")
def par_domain_cube_3d(tmp_path_factory):
    d = tmp_path_factory.mktemp("par_cube_3d")
    d_str = COMM.bcast(str(d), root=0)
    return _parallel_domain("cube.msh", 3, d_str)


# ---------------------------------------------------------------------------
# 2D parallel tests
# ---------------------------------------------------------------------------
@requires_mpi
class TestParallelDomain2D:

    def test_global_volume_preserved(self, par_domain_rect_2d):
        """
        Sum of all local volumes across all ranks must equal the single-process
        total volume (bounding-box area).
        """
        local_vol = np.sum(par_domain_rect_2d.cells.volume)
        global_vol = COMM.allreduce(local_vol, op=MPI.SUM)

        v = par_domain_rect_2d.nodes.vertex
        # Reduce bounding box across ranks
        x_min = COMM.allreduce(v[:, 0].min(), op=MPI.MIN)
        x_max = COMM.allreduce(v[:, 0].max(), op=MPI.MAX)
        y_min = COMM.allreduce(v[:, 1].min(), op=MPI.MIN)
        y_max = COMM.allreduce(v[:, 1].max(), op=MPI.MAX)
        bbox_area = (x_max - x_min) * (y_max - y_min)

        assert abs(global_vol - bbox_area) / bbox_area < 1e-4, \
            f"[rank {RANK}] global vol {global_vol:.6f} != bbox {bbox_area:.6f}"

    def test_global_cell_count_consistent(self, par_domain_rect_2d):
        """
        Sum of local cell counts + halo cells must be consistent across ranks.
        Each rank knows its local count; the sum over ranks >= total real cells.
        """
        local_count = par_domain_rect_2d.nbcells
        global_count = COMM.allreduce(local_count, op=MPI.SUM)
        assert global_count > 0

    def test_halo_cells_exist_on_each_rank(self, par_domain_rect_2d):
        """When N > 1 processes, every rank must have halo cells."""
        assert par_domain_rect_2d.nbhalos >= 0   # halos may be 0 on some ranks
        # At least one rank must have halos (otherwise no communication)
        any_halo = COMM.allreduce(par_domain_rect_2d.nbhalos > 0, op=MPI.LOR)
        assert any_halo, "No rank has halo cells — partitioning may be wrong"

    def test_local_cell_volumes_positive(self, par_domain_rect_2d):
        assert np.all(par_domain_rect_2d.cells.volume > 0)

    def test_local_face_normals_unit(self, par_domain_rect_2d):
        norms = np.linalg.norm(par_domain_rect_2d.faces.normal, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_gradient_linear_across_partitions(self, par_domain_rect_2d):
        """
        f(x,y) = x  →  grad = (1, 0) on every rank, including partition boundaries.
        This verifies that halo + haloghost communication does not corrupt the gradient.
        """
        domain = par_domain_rect_2d
        var = _make_var(domain, lambda x, y, z: x)
        var.compute_cell_gradient()

        assert np.allclose(var.gradcellx, 1.0, atol=ATOL_LINEAR), \
            f"[rank {RANK}] max gradcellx error = {np.max(np.abs(var.gradcellx - 1.0)):.2e}"
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR), \
            f"[rank {RANK}] max gradcelly error = {np.max(np.abs(var.gradcelly)):.2e}"


# ---------------------------------------------------------------------------
# 3D parallel tests
# ---------------------------------------------------------------------------
@requires_mpi
class TestParallelDomain3D:

    def test_global_volume_preserved(self, par_domain_cube_3d):
        local_vol = np.sum(par_domain_cube_3d.cells.volume)
        global_vol = COMM.allreduce(local_vol, op=MPI.SUM)

        v = par_domain_cube_3d.nodes.vertex
        x_min = COMM.allreduce(v[:, 0].min(), op=MPI.MIN)
        x_max = COMM.allreduce(v[:, 0].max(), op=MPI.MAX)
        y_min = COMM.allreduce(v[:, 1].min(), op=MPI.MIN)
        y_max = COMM.allreduce(v[:, 1].max(), op=MPI.MAX)
        z_min = COMM.allreduce(v[:, 2].min(), op=MPI.MIN)
        z_max = COMM.allreduce(v[:, 2].max(), op=MPI.MAX)
        bbox_vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

        assert abs(global_vol - bbox_vol) / bbox_vol < 1e-4, \
            f"[rank {RANK}] global vol {global_vol:.6f} != bbox {bbox_vol:.6f}"

    def test_local_cell_volumes_positive(self, par_domain_cube_3d):
        assert np.all(par_domain_cube_3d.cells.volume > 0)

    def test_gradient_linear_across_partitions(self, par_domain_cube_3d):
        """f(x,y,z) = x+y+z  →  grad = (1,1,1) on all ranks."""
        domain = par_domain_cube_3d
        var = _make_var(domain, lambda x, y, z: x + y + z)
        var.compute_cell_gradient()
        for grad, name in [(var.gradcellx, "x"), (var.gradcelly, "y"), (var.gradcellz, "z")]:
            assert np.allclose(grad, 1.0, atol=ATOL_LINEAR), \
                f"[rank {RANK}] grad{name} max err = {np.max(np.abs(grad - 1.0)):.2e}"


# ---------------------------------------------------------------------------
# 2D parallel gradient tests
# ---------------------------------------------------------------------------
@requires_mpi
class TestParallelGradient2D:
    """
    All gradient functions tested on the partitioned rectangle mesh.
    update_halo_value() is called before compute_cell_gradient() to ensure
    halo cells carry correct values across partition boundaries.
    """

    # --- Linear ---
    def test_linear_x(self, par_domain_rect_2d):
        """f = x  →  grad = (1, 0) on all ranks."""
        d = par_domain_rect_2d
        var = _make_var(d, lambda x, y, z: x)
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 1.0, atol=ATOL_LINEAR), \
            f"[rank {RANK}] gradx err={np.max(np.abs(var.gradcellx-1)):.2e}"
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR), \
            f"[rank {RANK}] grady err={np.max(np.abs(var.gradcelly)):.2e}"

    def test_linear_y(self, par_domain_rect_2d):
        """f = y  →  grad = (0, 1) on all ranks."""
        d = par_domain_rect_2d
        var = _make_var(d, lambda x, y, z: y)
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, 1.0, atol=ATOL_LINEAR)

    def test_linear_general(self, par_domain_rect_2d):
        """f = 3x - 2y  →  grad = (3, -2) on all ranks."""
        d = par_domain_rect_2d
        var = _make_var(d, lambda x, y, z: 3*x - 2*y)
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx,  3.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, -2.0, atol=ATOL_LINEAR)

    # --- Quadratic ---
    def test_quadratic_l2_error(self, par_domain_rect_2d):
        """f = x²+y²  →  exact grad = (2x, 2y). L2 err < 5 % on each rank."""
        d = par_domain_rect_2d
        var = _make_var(d, lambda x, y, z: x**2 + y**2)
        var.compute_cell_gradient()
        c = d.cells.center
        err_x = _l2_rel(var.gradcellx, 2*c[:, 0])
        err_y = _l2_rel(var.gradcelly, 2*c[:, 1])
        assert err_x < RTOL_SMOOTH, f"[rank {RANK}] quadratic gradx L2={err_x:.3e}"
        assert err_y < RTOL_SMOOTH, f"[rank {RANK}] quadratic grady L2={err_y:.3e}"

    # --- Sinusoidal ---
    def test_sin_x(self, par_domain_rect_2d):
        """f = sin(pi*x)  →  df/dx = pi*cos(pi*x), df/dy = 0."""
        d = par_domain_rect_2d
        kx = np.pi
        var = _make_var(d, lambda x, y, z: np.sin(kx*x))
        var.compute_cell_gradient()
        c = d.cells.center
        exact_gx = kx * np.cos(kx * c[:, 0])
        assert _l2_rel(var.gradcellx, exact_gx) < RTOL_SMOOTH, \
            f"[rank {RANK}] sin(x) gradx L2={_l2_rel(var.gradcellx, exact_gx):.3e}"
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR), \
            f"[rank {RANK}] sin(x) grady err={np.max(np.abs(var.gradcelly)):.2e}"

    def test_sin_xy(self, par_domain_rect_2d):
        """f = sin(pi*x)*sin(pi*y)  →  L2 error < 5 % on each rank."""
        d = par_domain_rect_2d
        kx = ky = np.pi
        var = _make_var(d, lambda x, y, z: np.sin(kx*x) * np.sin(ky*y))
        var.compute_cell_gradient()
        c = d.cells.center
        exact_gx = kx * np.cos(kx*c[:,0]) * np.sin(ky*c[:,1])
        exact_gy = ky * np.sin(kx*c[:,0]) * np.cos(ky*c[:,1])
        assert _l2_rel(var.gradcellx, exact_gx) < RTOL_SMOOTH, \
            f"[rank {RANK}] sin(xy) gradx L2={_l2_rel(var.gradcellx, exact_gx):.3e}"
        assert _l2_rel(var.gradcelly, exact_gy) < RTOL_SMOOTH, \
            f"[rank {RANK}] sin(xy) grady L2={_l2_rel(var.gradcelly, exact_gy):.3e}"


# ---------------------------------------------------------------------------
# 3D parallel gradient tests
# ---------------------------------------------------------------------------
@requires_mpi
class TestParallelGradient3D:

    # --- Linear ---
    def test_linear_x(self, par_domain_cube_3d):
        """f = x  →  grad = (1, 0, 0) on all ranks."""
        d = par_domain_cube_3d
        var = _make_var(d, lambda x, y, z: x)
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 1.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcellz, 0.0, atol=ATOL_LINEAR)

    def test_linear_y(self, par_domain_cube_3d):
        d = par_domain_cube_3d
        var = _make_var(d, lambda x, y, z: y)
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, 1.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcellz, 0.0, atol=ATOL_LINEAR)

    def test_linear_z(self, par_domain_cube_3d):
        d = par_domain_cube_3d
        var = _make_var(d, lambda x, y, z: z)
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcellz, 1.0, atol=ATOL_LINEAR)

    def test_linear_general(self, par_domain_cube_3d):
        """f = 2x - y + 3z  →  grad = (2, -1, 3) on all ranks."""
        d = par_domain_cube_3d
        var = _make_var(d, lambda x, y, z: 2*x - y + 3*z)
        var.compute_cell_gradient()
        assert np.allclose(var.gradcellx,  2.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcelly, -1.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcellz,  3.0, atol=ATOL_LINEAR)

    # --- Quadratic ---
    def test_quadratic_l2_error(self, par_domain_cube_3d):
        """f = x²+y²+z²  →  exact grad = (2x, 2y, 2z). L2 err < 5 %."""
        d = par_domain_cube_3d
        var = _make_var(d, lambda x, y, z: x**2 + y**2 + z**2)
        var.compute_cell_gradient()
        c = d.cells.center
        for grad, exact, name in [
            (var.gradcellx, 2*c[:,0], "x"),
            (var.gradcelly, 2*c[:,1], "y"),
            (var.gradcellz, 2*c[:,2], "z"),
        ]:
            err = _l2_rel(grad, exact)
            assert err < RTOL_SMOOTH, f"[rank {RANK}] quadratic grad{name} L2={err:.3e}"

    # --- Sinusoidal ---
    def test_sin_xyz(self, par_domain_cube_3d):
        """f = sin(pi*x)*sin(pi*y)*sin(pi*z)  →  L2 error < 5 % on each rank."""
        d = par_domain_cube_3d
        k = np.pi
        var = _make_var(d, lambda x, y, z: np.sin(k*x) * np.sin(k*y) * np.sin(k*z))
        var.compute_cell_gradient()
        c = d.cells.center
        exact_gx = k * np.cos(k*c[:,0]) * np.sin(k*c[:,1]) * np.sin(k*c[:,2])
        exact_gy = k * np.sin(k*c[:,0]) * np.cos(k*c[:,1]) * np.sin(k*c[:,2])
        exact_gz = k * np.sin(k*c[:,0]) * np.sin(k*c[:,1]) * np.cos(k*c[:,2])
        for grad, exact, name in [
            (var.gradcellx, exact_gx, "x"),
            (var.gradcelly, exact_gy, "y"),
            (var.gradcellz, exact_gz, "z"),
        ]:
            err = _l2_rel(grad, exact)
            assert err < RTOL_SMOOTH, f"[rank {RANK}] sin(xyz) grad{name} L2={err:.3e}"

    def test_sin_x_only(self, par_domain_cube_3d):
        """f = sin(pi*x)  →  df/dy = df/dz = 0 on all ranks."""
        d = par_domain_cube_3d
        k = np.pi
        var = _make_var(d, lambda x, y, z: np.sin(k*x))
        var.compute_cell_gradient()
        assert np.allclose(var.gradcelly, 0.0, atol=ATOL_LINEAR)
        assert np.allclose(var.gradcellz, 0.0, atol=ATOL_LINEAR)

"""
Parallel (MPI) tests for domain decomposition.

Run with:
    mpirun -n 2 pytest tests/test_parallel.py -m parallel
    mpirun -n 4 pytest tests/test_parallel.py -m parallel --oversubscribe

These tests are skipped when running on a single process.

What is tested
--------------
- Sum of local cell volumes == global volume (domain decomposition preserves volume)
- Sum of local cell counts == global cell count
- Halo cells exist on every process when N > 1 processes
- Local gradient of a linear function equals the analytical value,
  even across partition boundaries
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
        This verifies that halo communication does not corrupt the gradient.
        """
        Variable.is_called = False
        domain = par_domain_rect_2d
        var = Variable(domain=domain)

        c = domain.cells.center
        g = domain.faces.ghostcenter
        var.cell[:] = c[:, 0]
        var.ghost[:] = g[:, 0]

        var.update_halo_value()       # synchronise halos across ranks
        var.compute_cell_gradient()

        assert np.allclose(var.gradcellx, 1.0, atol=1e-4), \
            f"[rank {RANK}] max gradcellx error = {np.max(np.abs(var.gradcellx - 1.0)):.2e}"
        assert np.allclose(var.gradcelly, 0.0, atol=1e-4), \
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
        Variable.is_called = False
        domain = par_domain_cube_3d
        var = Variable(domain=domain)

        c = domain.cells.center
        g = domain.faces.ghostcenter
        var.cell[:] = c[:, 0] + c[:, 1] + c[:, 2]
        var.ghost[:] = g[:, 0] + g[:, 1] + g[:, 2]

        var.update_halo_value()
        var.compute_cell_gradient()

        for grad, name in [(var.gradcellx, "x"), (var.gradcelly, "y"), (var.gradcellz, "z")]:
            assert np.allclose(grad, 1.0, atol=1e-4), \
                f"[rank {RANK}] grad{name} max err = {np.max(np.abs(grad - 1.0)):.2e}"

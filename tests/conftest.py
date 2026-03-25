import os
import pytest
import numpy as np

from manapy.partitions import MeshPartition
from manapy.ddm import Domain
from manapy.ast import Variable
from manapy.base.base import Struct

MESH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mesh'))

# Use python backend (no Numba) for fast test startup
TEST_CONF = Struct(
    backend="python",
    signature=False,
    cache=False,
    float_precision="double",
    int_precision="signed",
)


def create_domain(mesh_name, dim, work_dir):
    """
    Create a Domain from a mesh file.
    MeshPartition writes meshesNPROC/ into the CWD, so we temporarily
    change into a dedicated temp directory.
    """
    mesh_path = os.path.join(MESH_DIR, mesh_name)
    original_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        Variable.is_called = False
        MeshPartition(mesh_path, dim=dim, conf=TEST_CONF, periodic=[0, 0, 0])
        domain = Domain(dim=dim, conf=TEST_CONF)
    finally:
        os.chdir(original_cwd)
    return domain


# ---------------------------------------------------------------------------
# Autouse fixture: reset Variable compilation state before every test so
# that 2D and 3D tests can coexist in the same session.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_variable_compiled():
    Variable.is_called = False
    yield
    Variable.is_called = False


# ---------------------------------------------------------------------------
# 2D domain fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def domain_rectangle_2d(tmp_path_factory):
    """2D rectangle mesh — quadrilateral cells."""
    d = tmp_path_factory.mktemp("rectangle_2d")
    return create_domain("rectangle.msh", 2, d)


@pytest.fixture(scope="session")
def domain_carre_2d(tmp_path_factory):
    """2D square mesh — quadrilateral cells."""
    d = tmp_path_factory.mktemp("carre_2d")
    return create_domain("carre.msh", 2, d)


@pytest.fixture(scope="session")
def domain_hybrid_2d(tmp_path_factory):
    """2D hybrid mesh — mixed triangles and quadrilaterals."""
    d = tmp_path_factory.mktemp("hybrid_2d")
    return create_domain("carre_hybrid.msh", 2, d)


@pytest.fixture(scope="session")
def domain_structured_2d(tmp_path_factory):
    """2D structured square mesh — quadrilateral cells."""
    d = tmp_path_factory.mktemp("structured_2d")
    return create_domain("carre_structure.msh", 2, d)


# ---------------------------------------------------------------------------
# 3D domain fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def domain_cube_3d(tmp_path_factory):
    """3D cube mesh — hexahedral cells."""
    d = tmp_path_factory.mktemp("cube_3d")
    return create_domain("cube.msh", 3, d)


@pytest.fixture(scope="session")
def domain_cube_bis_3d(tmp_path_factory):
    """3D alternative cube mesh — hexahedral cells."""
    d = tmp_path_factory.mktemp("cube_bis_3d")
    return create_domain("cube_bis.msh", 3, d)


# ---------------------------------------------------------------------------
# Helper: create a Variable and initialise cell + ghost values from a function
# ---------------------------------------------------------------------------
def make_variable_from_func(domain, func):
    """
    Return a Variable whose cell values (and boundary ghost values) are set
    by evaluating func(x, y[, z]) at each cell / ghost-cell centre.
    This lets gradient tests be exact for polynomial functions.
    """
    var = Variable(domain=domain)
    centers = domain.cells.center          # (ncells, 3)
    ghost_centers = domain.faces.ghostcenter  # (nfaces, dim+1) or (nfaces, 3)

    var.cell[:] = func(centers)
    # ghost centres: use only spatial coords (first `dim` columns)
    var.ghost[:] = func(ghost_centers[:, : domain.dim])
    return var

import pytest
from manapy.testing.TestTables import *
from manapy.testing.test_domain_helper import duplicate_config
from manapy.helpers import get_test_mesh
from base_test_domain import BaseTestDomain

@pytest.mark.parametrize(
  "config",
  duplicate_config([
    {
      "nb_parts": [1, 4],
      "dim": get_test_mesh('smallTriangles.msh')[0],
      "mesh_path": get_test_mesh('smallTriangles.msh')[1],
      "reference_domain_path": SmallTrianglesTables(),
      "partitioning_type": "Partitioning.Par_Nodal",
    },
    {
      "nb_parts": [1, 4, 16],
      "dim": get_test_mesh('triangles.msh')[0],
      "mesh_path": get_test_mesh('triangles.msh')[1],
      "reference_domain_path": TrianglesTables(),
      "partitioning_type": "Partitioning.Par_Nodal",
    },
    {
      "nb_parts": [4, 16],
      "dim": get_test_mesh('triangles.msh')[0],
      "mesh_path": get_test_mesh('triangles.msh')[1],
      "reference_domain_path": TrianglesTables(),
      "partitioning_type": "Partitioning.Par_Dual",
    },
  ]),
  indirect=True
)
class TestTriangles2D(BaseTestDomain):
  pass
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
      "dim": get_test_mesh('smallCuboid.msh')[0],
      "mesh_path": get_test_mesh('smallCuboid.msh')[1],
      "reference_domain_path": SmallCuboidTables(),
      "partitioning_type": "Partitioning.Par_Nodal",
    },
    {
      "nb_parts": [1, 4, 23],
      "dim": get_test_mesh('cuboid.msh')[0],
      "mesh_path": get_test_mesh('cuboid.msh')[1],
      "reference_domain_path": CuboidTables(),
      "partitioning_type": "Partitioning.Par_Nodal",
    }
  ]),
  indirect=True
)
class TestCuboid3D(BaseTestDomain):
  pass
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
      "dim": get_test_mesh('smallHybrid2D.msh')[0],
      "mesh_path": get_test_mesh('smallHybrid2D.msh')[1],
      "reference_domain_path": SmallHybrid2DTables(),
      "partitioning_type": "Partitioning.Par_Nodal",
    },
    {
      "nb_parts": [1, 16, 23],
      "dim": get_test_mesh('hybrid2d.msh')[0],
      "mesh_path": get_test_mesh('hybrid2d.msh')[1],
      "reference_domain_path": "hybrid2d.hd5",
      "partitioning_type": "Partitioning.Par_Nodal",
    },
  ]),
  indirect=True
)
class TestHybrid2D(BaseTestDomain):
  pass

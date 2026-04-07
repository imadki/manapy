import pytest
from manapy.testing.TestTables import *
from manapy.testing.test_domain_helper import duplicate_config
from manapy.helpers import get_test_mesh
from base_test_domain import BaseTestDomain

@pytest.mark.parametrize(
  "config",
  duplicate_config([
    {
      "nb_parts": [1, 4, 16],
      "dim": get_test_mesh('rectangles.msh')[0],
      "mesh_path": get_test_mesh('rectangles.msh')[1],
      "reference_domain_path": RectanglesTables(),
      "partitioning_type": "Partitioning.Par_Nodal",
    },
    {
      "nb_parts": [4, 16, 20],
      "dim": get_test_mesh('rectangles.msh')[0],
      "mesh_path": get_test_mesh('rectangles.msh')[1],
      "reference_domain_path": RectanglesTables(),
      "partitioning_type": "Partitioning.Par_Graph_K_Way",
    },
  ]),
  indirect=True
)
class TestRectangles2D(BaseTestDomain):
  pass
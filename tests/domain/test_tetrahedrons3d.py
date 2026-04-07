import pytest
from manapy.testing.TestTables import *
from manapy.testing.test_domain_helper import duplicate_config
from manapy.helpers import get_test_mesh
from base_test_domain import BaseTestDomain

@pytest.mark.parametrize(
  "config",
  duplicate_config(
    [
      {
        "nb_parts": [1, 4], # There are just 6 cells
        "dim": get_test_mesh('smallTetrahedrons.msh')[0],
        "mesh_path": get_test_mesh('smallTetrahedrons.msh')[1],
        "reference_domain_path": SmallTetrahedronTables(),
        "partitioning_type": "Partitioning.Par_Nodal",
      },
      {
        "nb_parts": [1, 4, 16, 30, 100],
        "dim": get_test_mesh('tetrahedrons.msh')[0],
        "mesh_path": get_test_mesh('tetrahedrons.msh')[1],
        "reference_domain_path": "tetrahedrons.hd5",
        "partitioning_type": "Partitioning.Par_Nodal",
      },
    ]
  ),
  indirect=True
)
class TestTetrahedrons3D(BaseTestDomain):
  pass
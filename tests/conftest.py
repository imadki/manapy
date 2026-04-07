import pytest
from manapy.testing.test_domain_helper import get_local_domains, get_reference_domain

@pytest.fixture(scope="class")
def config(request):
    return request.param

@pytest.fixture(scope="class")
def local_domains(config):
  nb_parts = config["nb_parts"]
  mesh_path = config["mesh_path"]
  dim = config["dim"]
  partitioning_type = config["partitioning_type"]

  return get_local_domains(nb_parts, mesh_path, dim, partitioning_type)

@pytest.fixture(scope="class")
def reference_domain(config, local_domains):
  reference_domain_path = config["reference_domain_path"]
  dim = config["dim"]
  return get_reference_domain(reference_domain_path, dim)

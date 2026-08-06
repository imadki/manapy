import pytest
from manapy.testing.test_domain_helper import get_local_domains, get_reference_domain, make_test_config
from manapy.domain import Domain

@pytest.fixture(scope="class")
def config(request):
    return request.param

@pytest.fixture(scope="session")
def manapy_config():
  """Precision pair + device every domain in the suite is built with.

  Session-scoped: the config only carries the dtypes/device that select the
  compiled kernels, so one instance is shared by every test.
  """
  return make_test_config()

@pytest.fixture(scope="class")
def local_domains(config, manapy_config):
  nb_parts = config["nb_parts"]
  mesh_path = config["mesh_path"]
  dim = config["dim"]
  partitioning_type = config["partitioning_type"]

  return get_local_domains(nb_parts, mesh_path, dim, partitioning_type, manapy_config)

@pytest.fixture(scope="class")
def domain(config, manapy_config):
  mesh_path = config["mesh_path"]
  dim = config["dim"]
  partitioning_type = config["partitioning_type"]

  return Domain.create_domain(mesh_path, dim, manapy_config, partitioning_type)

@pytest.fixture(scope="class")
def reference_domain(config, local_domains):
  reference_domain_path = config["reference_domain_path"]
  dim = config["dim"]
  return get_reference_domain(reference_domain_path, dim)

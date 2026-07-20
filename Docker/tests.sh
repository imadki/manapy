#!/bin/bash
set -e

# Unit and integration test suite (serial). Meshes come from the committed
# fixtures in tests/data/meshes, so no mesh generation is required.
python3 -m pytest tests

# Parallel smoke test: exercise the MPI code paths.
pushd tests
mpirun --allow-run-as-root --use-hwthread-cpus -n 2 python3 mpi_test.py
mpirun --allow-run-as-root --use-hwthread-cpus -n 4 --oversubscribe python3 mpi_test.py
popd

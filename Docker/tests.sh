#!/bin/bash

#tail -f
set -e

python3 -m pytest --color=yes -s

pushd tests
mpirun --allow-run-as-root --use-hwthread-cpus -n 2 python3 mpi_test.py
mpirun --allow-run-as-root --use-hwthread-cpus -n 4 --oversubscribe python3 mpi_test.py

popd

pushd manapy/examples/2D
python3 -c "import manapy; import manapy.ddm; print(manapy.__version__)"
python3 advection2d.py
mpirun --allow-run-as-root --use-hwthread-cpus -n 2 python3 advection2d.py
mpirun --allow-run-as-root --use-hwthread-cpus -n 4 --oversubscribe python3 advection2d.py #oversubscribe option allows you to run more MPI processes than there are available CPU cores on the system.

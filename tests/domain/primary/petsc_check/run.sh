set -e

mpi_args="-n 4"
parent_folder='/media/aben-ham/SSD/aben-ham/work'

mkdir -p new
cd new
mpirun $mpi_args python3 $parent_folder/manapy/manapy/domain/examples/laplacien2d.py
cd ..
mkdir -p old
cd old
mpirun $mpi_args python3 $parent_folder/manapy/manapy/examples/2D/laplacien2d.py
cd ..
python3 compare.py

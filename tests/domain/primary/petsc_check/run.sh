set -e

mpi_args="-n 4 --oversubscribe"

mkdir -p new
cd new
mpirun $mpi_args python3 /media/aben-ham/SSD/aben-ham/work/manapy/manapy/domain/examples/laplacien2d.py
cd ..
mkdir -p old
cd old
mpirun $mpi_args python3 /media/aben-ham/SSD/aben-ham/work/manapy/manapy/examples/2D/laplacien2d.py
cd ..
python3 compare.py

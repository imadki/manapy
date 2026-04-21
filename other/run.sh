set -e

#cd ../manapy/examples/2D
#rm -rf results
#mkdir results
#mpirun -n 10 python3 advecdiff2d.py
#mv vtk_results results/advecdiff2d
#mpirun -n 10 python3 advection2d.py
#mv vtk_results results/advection2d
#mpirun -n 10 python3 burgers2d.py
#mv vtk_results results/burgers2d
#mpirun -n 10 python3 darcy2d.py
#mv vtk_results results/darcy2d
#mpirun -n 1 python3 darcy_with_particles2d.py
#mv vtk_results results/darcy_with_particles2d
#mpirun -n 10 python3 diffusion2d.py
#mv vtk_results results/diffusion2d
#mpirun -n 10 python3 laplacien2d.py
#mv vtk_results results/laplacien2d
#mpirun -n 10 python3 shallow_water2d.py
#mv vtk_results results/shallow_water2d
#exit

cd ../manapy/examples/3D
rm -rf results
mkdir results
mpirun -n 1 python3 advecdiff3d.py
mv vtk_results results/advecdiff3d
mpirun -n 10 python3 advection3d.py
mv vtk_results results/advection3d
mpirun -n 10 python3 burgers3d.py
mv vtk_results results/burgers3d
mpirun -n 10 python3 darcy3d.py
mv vtk_results results/darcy3d
mpirun -n 10 python3 diffusion3d.py
mv vtk_results results/diffusion3d
mpirun -n 10 python3 laplacien3d.py
mv vtk_results results/laplacien3d
exit
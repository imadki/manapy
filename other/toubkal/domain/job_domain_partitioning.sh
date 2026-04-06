#!/bin/bash
#SBATCH --job-name=benchmarks
#SBATCH --output=logs/out_%t.log      # standard output
#SBATCH --error=logs/error_%t.log        # standard error
#SBATCH --mem=150G                     # total memory
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Load modules
module load Anaconda3/2020.11
module load Python/3.8.6-GCCcore-10.2.0
module load OpenMPI/4.1.1-GCC-11.2.0
module load CMake/3.22.1-GCCcore-11.2.0
# module load gmsh

export OMPI_MCA_mtl=^ofi
export OMPI_MCA_btl=self,tcp


echo "Starting job on $(hostname)"
bash ./node_info.sh > node_info.txt
bash benchmark_nb_parts.sh ~/work/meshes 2> /dev/null

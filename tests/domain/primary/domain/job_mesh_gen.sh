#!/bin/bash
#SBATCH --job-name=my_gmsh_job
#SBATCH --output=gmsh_output.log      # standard output
#SBATCH --error=gmsh_error.log        # standard error
#SBATCH --mem=24G                     # total memory
#SBATCH --time=03:00:00
#SBATCH --ntasks=1                    # single task

# Load modules
module load Anaconda3/2020.11
module load Python/3.8.6-GCCcore-10.2.0
module load OpenMPI/4.1.1-GCC-11.2.0
module load CMake/3.22.1-GCCcore-11.2.0
module load gmsh

# Your command here
echo "Starting job on $(hostname)"
bash ./node_info.sh > node_info.txt
bash ./mesh_gen.sh

## Version and hardware used during the simulation

- This is Ginkgo 1.11.0 (develop)
  - running with core module 1.11.0 (develop)
  - the reference module is  1.11.0 (develop)
  - the OpenMP    module is  1.11.0 (develop)
  - the CUDA      module is  1.11.0 (develop)

- nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2023 NVIDIA Corporation
  Built on Tue_Aug_15_22:02:13_PDT_2023
  Cuda compilation tools, release 12.2, V12.2.140
  Build cuda_12.2.r12.2/compiler.33191640_0

- gcc (Ubuntu 14.2.0-19ubuntu2) 14.2.0

- c++ (Ubuntu 14.2.0-19ubuntu2) 14.2.0

- Linux 6.14.0-37-generic

- Ubuntu 25.04

- AMD Ryzen™ 9 9900X × 24

## Ginkgo Build Configuration Used

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_BENCHMARKS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_REFERENCE=ON -DGINKGO_BUILD_OMP=ON -DGINKGO_BUILD_CUDA=ON -DGINKGO_DOC_GENERATE_EXAMPLES=OFF
```

## Build

Download the data from https://drive.google.com/file/d/1mpG9yru_4hid40ovjHQZ0cznAGtOwH0z/view?usp=sharing and unzip it.

```
bash build.sh
```

## How to Run the executable

```
Usage: ./ginkgo_solver [executor] [solver name] [data folder containing A.mtx,b.mtx,x.mtx]
executor: omp, cuda, hip, dpcpp, reference
sover name: bicg ,bicgstab ,cgs ,cbgmres ,gmres ,gcr ,idr, cg
```

```
./ginkgo_solver reference gmres small_data
```

## Data
- There are three folders: **large_data**, **med_data**, and **small_data**. Each folder contains the same files: **A.mtx**, **b.mtx**, and **x_sol.mtx**.

- The data is generated using the same mesh, but with different resolutions, using the **Gmsh** software. The Manapy project reads the mesh and constructs the matrices A and b. The matrix A is stored in COO sparse matrix format, and b is stored as a dense matrix; both are saved using the Matrix Market Exchange format (.mtx).

- The Manapy project also uses the same matrices to compute solutions with the MUMPS and PETSc solvers.

- The file x_sol.mtx contains the reference solution provided by the MUMPS solver.


## Results

- Using `PETSc` solver with gmres solver and bjacobi preconditioner
  - small_data Converged in `50` iterations,  Final residual norm: `9.798227117749697e-07`,  Time `0.0062167` seconds
  - med_data   Converged in `109` iterations,  Final residual norm: `1.4667580799634466e-06`, Time `0.03199` seconds
  - large_data Converged in `2807` iterations, Final residual norm: `4.56774954381256e-06`,   Time `103.415419` seconds

- Using `PETSc` solver with gmres solver and gamg preconditioner
  - small_data Converged in `15` iterations,  Final residual norm: `1.1606543677274363e-06`,  Time `0.014013292000001` seconds
  - med_data   Converged in `16` iterations,  Final residual norm: `1.6172822298056422e-06`,  Time `0.033224825000001` seconds
  - large_data Converged in `20` iterations,  Final residual norm: `1.5869596560474027e-05`,  Time `5.362165622999999 seconds`

- Using `Mumps` solver
  - small_data Final residual norm: `2.201875984842108e-09` Time `0.103708` seconds
  - med_data   Final residual norm: `1.326193161469837e-08` Time `0.389664` seconds
  - large_data Final residual norm: `1.449581096338454e-05` Time `53.53278` seconds

- Using `Ginkgo` with gmres solver only using reference executor
  - small_data Iterations: `457`  Final residual norm: `2.220859960660137e-09` Time: `162855`  microseconds
  - med_data   Iterations: `1156` Final residual norm: `1.435653064783598e-08` Time: `1683813` microseconds
  - large_data No results (Take too much time > 30min).

- Using `Ginkgo` with gmres solver and `jacobi` preconditioner using reference executor
  - small_data Iterations: `402` Final residual norm: `2.144497982031397e-09` Time: `159142` microseconds
  - med_data Diverge
  - large_data Diverge

- Using `Ginkgo` with gmres solver and `par-ilu` preconditioner using reference executor
  - small_data Iterations: 64 Final residual norm: `2.596573735454223e-09` Time: `17720` microseconds
  - med_data -nan
  - large_data -nan

## Problem Description
- Using Ginkgo with the GMRES solver on the reference executor, the solver converges for the small and medium problem sizes but requires a large number of iterations and execution time. For the large dataset, no result is obtained due to excessive execution time.

- When adding the Jacobi preconditioner to GMRES in Ginkgo (still using the reference executor), The solver only converge for small dataset. However, the solver fails to converge for both the medium and large datasets, resulting in divergence.

- adding par-ilu preconditioner to GMRES in Ginkgo (still using the reference executor), The solver only converge for small dataset. both the medium and large datasets, resulting in divergence (-nan).
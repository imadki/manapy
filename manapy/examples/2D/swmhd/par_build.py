#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal parallel build test for CROSS-RANK periodic boundaries.

Just partitions the periodic square mesh and builds the Domain on every rank,
printing per-rank cell / periodic-face / halo-face counts. Success = it builds
without the "no same-rank partner" error and the halo-face count is > 0 on the
ranks that own a cross-rank periodic boundary.
"""
import os
from mpi4py import MPI
from manapy.domain import Domain, Partitioning

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MESH_DIR = os.environ.get(
    'MESH_DIR', os.path.join(BASE_DIR, '..', '..', '..', '..', 'meshes'))
filename = os.environ.get('MESH', 'periodic_square48.msh')
mesh_path = os.path.join(MESH_DIR, filename)

domain = Domain.create_domain(mesh_path, 2, Partitioning.Par_Nodal, recreate=True)

nb = int(domain.nbcells)
nb_periodic = int(len(domain.periodicboundaryfaces))
nb_halo = int(len(domain.halofaces))

# Serialize prints for readability.
for r in range(SIZE):
    if r == RANK:
        print(f"[rank {RANK}/{SIZE}] nbcells={nb}  "
              f"periodicboundaryfaces={nb_periodic}  halofaces={nb_halo}",
              flush=True)
    COMM.Barrier()

tot_cells = COMM.allreduce(nb, op=MPI.SUM)
tot_periodic = COMM.allreduce(nb_periodic, op=MPI.SUM)
tot_halo = COMM.allreduce(nb_halo, op=MPI.SUM)
if RANK == 0:
    print(f"[TOTAL] cells={tot_cells}  periodicboundaryfaces={tot_periodic}  "
          f"halofaces={tot_halo}", flush=True)
    print("BUILD OK", flush=True)

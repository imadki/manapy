// openmpi_test.cpp: minimal OpenMPI communication example
// Compile with: mpic++ -std=c++17 openmpi_test.cpp -o openmpi_test
// Run with:     mpirun -np 4 ./openmpi_test

#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get this process's rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // get total number of processes

    int message = 0;

    if (rank == 0) {
        message = 42;
        std::cout << "Rank 0 sending message = " << message << " to rank 1\n";
        MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }

    if (rank == 1) {
        MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Rank 1 received message = " << message << " from rank 0\n";
    }

    // Now broadcast the message to all ranks
    MPI_Bcast(&message, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << "Rank " << rank << " sees message = " << message << " after broadcast\n";

    MPI_Finalize();
    return 0;
}

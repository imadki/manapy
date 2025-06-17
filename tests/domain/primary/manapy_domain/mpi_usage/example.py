from mpi4py import MPI
import numpy as np

# Initialize the MPI communicator
comm = MPI.COMM_WORLD

# Get the total number of processes
size = comm.Get_size()

# Get the rank (ID) of the current process
rank = comm.Get_rank()

# Each process prints a message
print(f"Hello from process {rank} out of {size} processes")

# Example of sending and receiving messages
if rank == 0:
    data = np.array([1, 2, 3], dtype=np.float32)
    comm.send(data, dest=1, tag=11)
    print("Process 0 sent data to Process 1")
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    print(f"Process 1 received data: {data}")
    print(type(data))

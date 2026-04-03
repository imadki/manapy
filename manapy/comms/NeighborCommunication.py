from mpi4py import MPI
import numpy as np

"""
Define for symmetrical communication.
"""
class NeighborCommunication:
  def __init__(self, neighbors: 'int[:]', send_counts: 'int[:]', send_indices: 'int[:]'):
    self.size = MPI.COMM_WORLD.Get_size()
    self.rank = MPI.COMM_WORLD.Get_rank()
    self.cache = {}
    self.neighbors = None
    self.send_counts = None
    self.send_indices = None
    self.recv_counts = None
    self.recv_total = None
    self.graph_comm = None

    if self.size == 1:
      self.neighbors = [0]
      self.send_counts = np.zeros(1, dtype=np.uint32)
      self.send_indices = np.zeros(1, dtype=np.uint32)
      self.recv_counts = np.zeros(1, dtype=np.uint32)
      self.recv_total = np.zeros(1, dtype=np.uint32)
      self.graph_comm = MPI.COMM_WORLD.Create_dist_graph_adjacent(
        sources=self.neighbors,
        destinations=self.neighbors,
        sourceweights=None, destweights=None
      )
    else:
      # Check of send_counts and send_indices are will construct.
      if np.sum(send_counts) != len(send_indices):
        raise ValueError("Mismatch: send_counts sum != send_data size")

      self.neighbors = neighbors.astype(np.uint32)
      self.send_counts = send_counts.astype(np.uint32)
      self.send_indices = send_indices

      # --- Create graph communicator ---
      self.graph_comm = MPI.COMM_WORLD.Create_dist_graph_adjacent(
        sources=neighbors,
        destinations=neighbors,
        sourceweights=None, destweights=None
      )

      # Creat recv_counts
      rcount = np.zeros(len(neighbors), dtype=np.uint32)
      self.graph_comm.Neighbor_alltoallv(self.send_counts, rcount)
      self.recv_counts = rcount
      self.recv_total = np.sum(rcount)

  def exchange(self, data, recv_buffer=None):
    if self.size == 1 or self.neighbors.shape[0] == 0:
      return np.copy(data)
    # --- MPI datatype ---
    mpi_type = MPI._typedict.get(data.dtype.char)
    if mpi_type is None:
        raise TypeError(f"Unsupported dtype {data.dtype}")

    # --- Ensure contiguous ---
    data = np.ascontiguousarray(data)

    # --- Detect dimensionality ---
    if data.ndim == 1:
        block_size = 1
    elif data.ndim == 2:
        block_size = data.shape[1]
    else:
        raise ValueError("Only 1D or 2D arrays are supported")

    # --- Build send buffer ---
    send_data = data[self.send_indices]

    if (data.ndim, block_size) not in self.cache:
      # --- Convert counts: rows → scalar elements ---
      send_counts = np.array(self.send_counts) * block_size
      recv_counts = np.array(self.recv_counts) * block_size

      # --- Compute displacements ---
      send_displs = np.insert(np.cumsum(send_counts[:-1]), 0, 0)
      recv_displs = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)

      self.cache[(data.ndim, block_size)] = (
        send_counts, send_displs,
        recv_counts, recv_displs
      )

    (send_counts, send_displs, recv_counts, recv_displs) = self.cache[(data.ndim, block_size)]

    if recv_buffer is not None:
      # If you already have a receive buffer
      recv_data = recv_buffer
    else:
      # --- Allocate receive buffer ---
      if block_size == 1:
          recv_data = np.empty(self.recv_total, dtype=data.dtype)
      else:
          recv_data = np.empty((self.recv_total, block_size), dtype=data.dtype)

    # --- Communication ---
    self.graph_comm.Neighbor_alltoallv(
      (send_data, (send_counts, send_displs), mpi_type),
      (recv_data, (recv_counts, recv_displs), mpi_type)
    )

    return recv_data

  def immediate_exchange(self, data, recv_buffer):
    if self.size == 1 or self.neighbors.shape[0] == 0:
      return np.empty_like(data)
    # --- MPI datatype ---
    mpi_type = MPI._typedict.get(data.dtype.char)
    if mpi_type is None:
        raise TypeError(f"Unsupported dtype {data.dtype}")

    # --- Ensure contiguous ---
    data = np.ascontiguousarray(data)

    # --- Detect dimensionality ---
    if data.ndim == 1:
        block_size = 1
    elif data.ndim == 2:
        block_size = data.shape[1]
    else:
        raise ValueError("Only 1D or 2D arrays are supported")

    # --- Build send buffer ---
    send_data = data[self.send_indices]

    if (data.ndim, block_size) not in self.cache:
      # --- Convert counts: rows → scalar elements ---
      send_counts = np.array(self.send_counts) * block_size
      recv_counts = np.array(self.recv_counts) * block_size

      # --- Compute displacements ---
      send_displs = np.insert(np.cumsum(send_counts[:-1]), 0, 0)
      recv_displs = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)

      self.cache[(data.ndim, block_size)] = (
        send_counts, send_displs,
        recv_counts, recv_displs
      )

    (send_counts, send_displs, recv_counts, recv_displs) = self.cache[(data.ndim, block_size)]

    # --- Communication ---
    return self.graph_comm.Ineighbor_alltoallv(
      (send_data, (send_counts, send_displs), mpi_type),
      (recv_buffer, (recv_counts, recv_displs), mpi_type)
    )



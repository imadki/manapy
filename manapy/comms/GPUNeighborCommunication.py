# -*- coding: utf-8 -*-
"""
Echange de halos GPU-aware (multi-rang).

Reprend la logique de manapy.comms.NeighborCommunication.exchange mais :
  - le gather du buffer d'envoi (send_data = data[send_indices]) se fait sur le
    GPU (kernel `define_halosend`) ;
  - les buffers d'envoi/reception passes a MPI sont des tableaux DEVICE (numba
    DeviceNDArray expose __cuda_array_interface__) -> avec un MPI CUDA-aware,
    Neighbor_alltoallv lit/ecrit directement en memoire GPU, sans copie hote.

GPUNeighborCommunication enveloppe la comm hote existante (reutilise graph_comm,
send_indices, recv_counts) et expose la meme API .exchange(data, recv_buffer)
afin d'etre un drop-in pour domain.halo_comm cote Variable.

Limite actuelle : tableaux 1D uniquement (suffisant pour l'advection : cell,
gradcellx/y/z, psi). Le bloc 2D leverait NotImplementedError.
"""
import os
import numpy as np
from mpi4py import MPI

# numba.cuda et manapy.backends.gpu sont importes PARESSEUSEMENT dans les methodes
# GPU ci-dessous. Sinon, un simple `import manapy.domain` (chemin CPU) tire toute
# la pile numba.cuda (~0.6 s solo, bien plus a 112 rangs sur NFS) alors qu'elle
# n'est jamais utilisee. GPUNeighborCommunication n'est instancie qu'en backend GPU.


def _gpu_aware_enabled():
  # GPU-aware MPI : opt-in. Beaucoup de builds MPI ne supportent pas les pointeurs
  # device au runtime (segfault). Par defaut on stage par l'hote (portable).
  return os.environ.get("MANAPY_GPU_AWARE_MPI", "0") not in ("0", "", "false", "False")


def _make_gather_kernel(gpu):
  from numba import cuda
  def kernel_define_halosend(w_c: 'float[:]', w_halosend: 'float[:]', indsend: 'int[:]'):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, w_halosend.shape[0], stride):
      w_halosend[i] = w_c[indsend[i]]
  return gpu.compile_kernel(kernel_define_halosend)


class GPUNeighborCommunication:
  def __init__(self, base, gpu):
    self.base = base
    self.gpu = gpu
    self.graph_comm = base.graph_comm
    self.size = base.size
    self.neighbors = base.neighbors

    self._active = (self.size > 1 and self.neighbors.shape[0] > 0)
    self._gpu_aware = _gpu_aware_enabled()
    self._gather = None
    self._d_send_indices = None
    self._send_buf = {}   # dtype -> device send buffer (taille nsend)
    self._counts = None

    if self._active and self._gpu_aware:
      from numba import cuda
      self._gather = _make_gather_kernel(gpu)
      self._d_send_indices = cuda.to_device(np.ascontiguousarray(base.send_indices).astype(np.int32))
      send_counts = np.asarray(base.send_counts, dtype=np.int32)
      recv_counts = np.asarray(base.recv_counts, dtype=np.int32)
      send_displs = np.insert(np.cumsum(send_counts[:-1]), 0, 0).astype(np.int32)
      recv_displs = np.insert(np.cumsum(recv_counts[:-1]), 0, 0).astype(np.int32)
      self._counts = (send_counts, send_displs, recv_counts, recv_displs)

  def _send_buffer(self, nsend, dtype):
    from numba import cuda
    key = np.dtype(dtype).str
    buf = self._send_buf.get(key)
    if buf is None or buf.size != nsend:
      buf = cuda.device_array(nsend, dtype=dtype)
      self._send_buf[key] = buf
    return buf

  def exchange(self, data, recv_buffer=None):
    if not self._active:
      # mono-rang / sans voisin : rien recu, on rend le buffer inchange
      return recv_buffer

    if data.ndim != 1:
      raise NotImplementedError("GPU halo exchange : seulement 1D pour l'instant")

    if not self._gpu_aware:
      return self._exchange_host_staged(data, recv_buffer)

    from manapy.backends.gpu import GPUArray
    d_data = GPUArray.to_device(data)
    d_recv = GPUArray.to_device(recv_buffer)

    nsend = len(self.base.send_indices)
    d_send = self._send_buffer(nsend, data.dtype)

    # gather GPU : d_send[i] = d_data[send_indices[i]]
    grid, block = self.gpu.get_gpu_params(nsend)
    self._gather[grid, block, self.gpu.stream](d_data, d_send, self._d_send_indices)
    if self.gpu.stream is not None:
      self.gpu.stream.synchronize()

    mpi_type = MPI._typedict.get(data.dtype.char)
    if mpi_type is None:
      raise TypeError(f"dtype non supporte {data.dtype}")
    send_counts, send_displs, recv_counts, recv_displs = self._counts

    # alltoallv GPU-aware : buffers DEVICE passes directement
    self.graph_comm.Neighbor_alltoallv(
      (d_send, (send_counts, send_displs), mpi_type),
      (d_recv, (recv_counts, recv_displs), mpi_type),
    )
    return recv_buffer

  def _exchange_host_staged(self, data, recv_buffer):
    """device->hote, MPI sur hote (comm de base), hote->device. Portable."""
    from numba import cuda
    from manapy.backends.gpu import GPUArray
    d_data = GPUArray.to_device(data)
    stream = self.gpu.stream
    h_data = d_data.copy_to_host(stream=stream)
    if stream is not None:
      stream.synchronize()

    # gather + alltoallv sur hote, via la comm existante
    h_recv = self.base.exchange(np.ascontiguousarray(h_data))

    if recv_buffer is None:
      return h_recv
    # re-upload du resultat dans la copie device du buffer de reception
    d_recv = GPUArray.to_device(recv_buffer)
    cuda.to_device(np.ascontiguousarray(h_recv), to=d_recv, stream=stream)
    if stream is not None:
      stream.synchronize()
    return recv_buffer

  def __getattr__(self, name):
    # delegue tout le reste (immediate_exchange, etc.) a la comm hote
    return getattr(self.base, name)

def all_to_all(w_halosend, taille, scount, rcount, w_halorecv, comm_ptr, mpi_precision):
  s_msg = [w_halosend, (scount), mpi_precision]
  r_msg = [w_halorecv, (rcount), mpi_precision]
  comm_ptr.Neighbor_alltoallv(s_msg, r_msg)

def define_halosend(w_c: 'float[:]', w_halosend: 'float[:]', indsend: 'int32[:]'):
  w_halosend[:] = w_c[indsend[:]]
from scipy.io import mmread
import numpy as np

# Original file
O = mmread("cmake-build-debug/small_data/A.mtx")

# Ginkgo output after reading and parsing the original file
G = mmread("cmake-build-debug/ginkgo_A.mtx")



G_dense = G.toarray()
O_dense = O.toarray()



is_equal = np.allclose(G_dense, O_dense, rtol=1e-6, atol=1e-9)
print(is_equal)
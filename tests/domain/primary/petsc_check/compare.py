import scipy
import numpy as np

n = scipy.io.mmread('./new/x.mtx').flatten()
o = scipy.io.mmread('./old/x.mtx').flatten()

n_n = np.linalg.norm(n)
n_o = np.linalg.norm(o)
print("=> norm_n:", n_n)
print("=> norm_o:", n_o)
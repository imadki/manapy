import numpy as np
import manapy_domain   # after build / install

# toy 4-vertex graph (–1 marks “no neighbour”)
g = np.array([[1, 2, 0, 2],
              [0, 2, 3, 3],
              [0, 1, 0, 2],
              [1, 0, 0, 1]], dtype=np.int64)

cuts, part = manapy_domain.make_n_part(g, 2)
print("edge-cut:", cuts)        # e.g. 2
print("partitions:", part)      # e.g. [0 1 0 1]

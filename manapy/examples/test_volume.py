from mpi4py import MPI
import timeit

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from manapy.partitions import MeshPartition
from manapy.ddm import Domain

from manapy.base.base import Struct
import numpy as np

from scipy.special import factorial


import os

start = timeit.default_timer()

# ... get the mesh directory
try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..', '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
 
filename = "carre_structure.msh"

#File name
filename = os.path.join(MESH_DIR, filename)
dim = 2

###Config###
#backend numba or python
#signature: add types to functions (make them faster) but compilation take time
#cache: avoid recompilation in the next run
running_conf = Struct(backend="numba", signature=True, cache=True)
mesh = MeshPartition(filename, dim=dim, conf=running_conf, periodic=[0,0,0])

#import sys; sys.exit()
#Create the informations about cells, faces and nodes
domain = Domain(dim=dim, conf=running_conf)
faces = domain.faces
cells = domain.cells
halos = domain.halos
nodes = domain.nodes

nbnodes = domain.nbnodes
nbfaces = domain.nbfaces
nbcells = domain.nbcells

def get_simplex_volumes(cells, coors):
    """                                                                                                                                                                                                    
    Get volumes of simplices in nD.                                                                                                                                                                        

    Parameters                                                                                                                                                                                             
    ----------                                                                                                                                                                                             
    cells : array, shape (n, d)                                                                                                                                                                            
        The indices of `n` simplices with `d` vertices into `coors`.                                                                                                                                       
    coors : array                                                                                                                                                                                          
        The coordinates of simplex vertices.                                                                                                                                                               

    Returns                                                                                                                                                                                                
    -------                                                                                                                                                                                                
    volumes : array                                                                                                                                                                                        
        The volumes of the simplices.                                                                                                                                                                      
    """
    scoors = coors[cells]
    deltas = scoors[:, 1:] - scoors[:, :1]
    dim = coors.shape[1]
    volumes = np.linalg.det(deltas) / factorial(dim)

    return volumes

volumes = get_simplex_volumes(cells.nodeid[:,:3], nodes.vertex[:,:-2])
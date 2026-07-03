"""Common linear-solver kernels shared by all schemes.

Auto-split from the former monolithic ls_compute.py: each scheme now lives in
its own module with its own setup(dim) so that only the kernels of the scheme
actually in use get compiled.
"""
from manapy.backends.compile_fun import compile
import numpy as np

### UTILS (dim-agnostic, shared)

def _convert_solution(x1: 'float[:]', x1converted: 'float[:]', cell_tc: 'int[:]', b0Size: 'int'):
  for i in range(b0Size):
    x1converted[i] = x1[cell_tc[i]]

def _rhs_value_dirichlet_node(Pbordnode: 'float[:]', nodes: 'int[:]', value: 'float[:]'):
  for i in nodes:
    Pbordnode[i] = value[i]

def _rhs_value_dirichlet_face(Pbordface: 'float[:]', faces: 'int[:]', value: 'float[:]'):
  for i in faces:
    Pbordface[i] = value[i]

def _gs_convert_solution(start: 'int', stride: 'int', x1: 'float[:]', x1converted: 'float[:]',
                         cell_tc: 'int[:]', b0Size: 'int'):
  for i in range(start, b0Size, stride):
    x1converted[i] = x1[cell_tc[i]]

def _gs_rhs_value_dirichlet_node(start: 'int', stride: 'int', Pbordnode: 'float[:]',
                                 nodes: 'int[:]', value: 'float[:]'):
  for k in range(start, nodes.shape[0], stride):
    i = nodes[k]
    Pbordnode[i] = value[i]

def _gs_rhs_value_dirichlet_face(start: 'int', stride: 'int', Pbordface: 'float[:]',
                                 faces: 'int[:]', value: 'float[:]'):
  for k in range(start, faces.shape[0], stride):
    i = faces[k]
    Pbordface[i] = value[i]

def _gs_set_scalar_at(start: 'int', stride: 'int', target: 'float[:]', idx: 'int[:]', value: 'float'):
  # target[idx[k]] = value (pour le neumann : Pbordnode[neumann_nodes] = 1.)
  for k in range(start, idx.shape[0], stride):
    target[idx[k]] = value


_done = False

def setup(dim):
    """Compile the dim-agnostic helpers shared by every scheme. Idempotent."""
    global _done
    if _done:
        return
    global convert_solution, rhs_value_dirichlet_node, rhs_value_dirichlet_face
    convert_solution = compile(_convert_solution)
    rhs_value_dirichlet_node = compile(_rhs_value_dirichlet_node)
    rhs_value_dirichlet_face = compile(_rhs_value_dirichlet_face)
    _done = True

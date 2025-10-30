import numpy as np
import meshio
from manapy.backends.compile_fun import compile
from manapy.backends.types import FLOAT_TYPE

def _append(cells: 'int32[:, :]', cells_item: 'int32[:, :]', counter: 'int32'):
  for i in range(len(cells_item)):
    cells[counter, 0:len(cells_item[i])] = cells_item[i]
    cells[counter, -1] = len(cells_item[i])
    counter += 1

def _append_1d(arr_dest: 'int32[:]', arr_src: 'int32[:]', counter: 'int32'):
  for i in range(len(arr_src)):
    arr_dest[counter] = arr_src[i]
    counter += 1

append = compile(_append)
append_1d = compile(_append_1d)

class Mesh:
  def __init__(self, mesh_path, dim):
    if not (isinstance(dim, int) and dim == 2 or dim == 3):
      raise ValueError('Invalid dimension')

    mesh, cells_dict, points, cell_data_dict = self._read_mesh(mesh_path)
    cells, cells_type, max_cell_nodeid, max_cell_faceid, max_face_nodeid = self._create_cells(cells_dict, dim)
    phy_faces, phy_faces_name = self._create_phy_faces(cells_dict, cell_data_dict, dim)

    self.mesh = mesh
    self.cells = cells
    self.cells_type = cells_type
    self.max_cell_nodeid = max_cell_nodeid
    self.max_cell_faceid = max_cell_faceid
    self.max_face_nodeid = max_face_nodeid
    self.points = points
    self.phy_faces = phy_faces
    self.phy_faces_name = phy_faces_name
    self.dim = dim

    if len(cells) == 0 or len(points) == 0:
      raise ValueError('Empty mesh')

  def _read_mesh(self, mesh_path):
    mesh = meshio.read(mesh_path)
    MESHIO_VERSION = int(meshio.__version__.split(".")[0])
    if MESHIO_VERSION < 4:
      # print(mesh.cell_data['triangle']['gmsh:physical'])
      # print(mesh.cells['triangle'])
      # need to reverse order of access for compatibility
      cell_data_dict = {}
      for k1 in mesh.cell_data.keys():
        for k2 in mesh.cell_data[k1].keys():
          if cell_data_dict.get(k2) is None:
            cell_data_dict[k2] = {}
          cell_data_dict[k2][k1] = mesh.cell_data[k1][k2]
      cells_dict = mesh.cells
      # raise NotImplementedError
    else:
      # print(mesh.cell_data_dict['gmsh:physical']['triangle'])
      # print(mesh.cells_dict['triangle'])
      cells_dict = mesh.cells_dict
      cell_data_dict = mesh.cell_data_dict
    points = mesh.points
    points = np.array(points, dtype=FLOAT_TYPE)

    return mesh, cells_dict, points, cell_data_dict

  def _create_phy_faces(self, cells_dict, cell_data_dict, dim):
    physicals = cell_data_dict['gmsh:physical']
    physicals_key = ['line']
    if dim == 3:
      physicals_key = ['quad', 'triangle']
    max_nb_face_nodes = 2
    counter = 0

    for k in physicals_key:
      if physicals.get(k) is not None:
        counter += len(physicals[k])
        if k == 'triangle':
          max_nb_face_nodes = max(max_nb_face_nodes, 3)
        elif k == 'quad':
          max_nb_face_nodes = max(max_nb_face_nodes, 4)

    phy_faces = np.zeros(shape=(counter, max_nb_face_nodes + 1), dtype=np.int32)
    phy_faces_name = np.zeros(shape=counter, dtype=np.int32)

    counter = np.int32(0)
    for k in physicals_key:
      if physicals.get(k) is not None:
        cells = np.array(cells_dict[k], dtype=np.int32)
        append(phy_faces, cells, counter)
        physical = np.array(physicals[k], dtype=np.int32)
        append_1d(phy_faces_name, physical, counter)
        counter += len(physicals[k])

    return phy_faces, phy_faces_name

  def _create_cells(self, meshio_mesh_dic, dim):
    # TODO make cell Types global constant
    allowed_cells = ['quad', 'triangle']
    if dim == 3:
      allowed_cells = ['pyramid', 'hexahedron', 'tetra']
    cell_type_dic = {
      "triangle": 1,
      "quad": 2,
      "tetra": 3,
      "hexahedron": 4,
      "pyramid": 5,
    }
    max_cell_nodeid = -1
    max_cell_faceid = -1
    max_face_nodeid = -1
    for item in meshio_mesh_dic.keys():
      if item == 'triangle':
        max_cell_nodeid = max(max_cell_nodeid, 3)
        max_cell_faceid = max(max_cell_faceid, 3)
        max_face_nodeid = max(max_face_nodeid, 2)
      elif item == 'quad':
        max_cell_nodeid = max(max_cell_nodeid, 4)
        max_cell_faceid = max(max_cell_faceid, 4)
        max_face_nodeid = max(max_face_nodeid, 2)
      elif item == 'tetra':
        max_cell_nodeid = max(max_cell_nodeid, 4)
        max_cell_faceid = max(max_cell_faceid, 4)
        max_face_nodeid = max(max_face_nodeid, 3)
      elif item == 'hexahedron':
        max_cell_nodeid = max(max_cell_nodeid, 8)
        max_cell_faceid = max(max_cell_faceid, 6)
        max_face_nodeid = max(max_face_nodeid, 4)
      elif item == 'pyramid':
        max_cell_nodeid = max(max_cell_nodeid, 5)
        max_cell_faceid = max(max_cell_faceid, 5)
        max_face_nodeid = max(max_face_nodeid, 4)

    number_of_cells = 0
    for item in allowed_cells:
      if meshio_mesh_dic.get(item) is not None:
        number_of_cells += len(meshio_mesh_dic[item])

    cells = np.zeros(shape=(number_of_cells, max_cell_nodeid + 1), dtype=np.int32)
    cells_type = np.zeros(shape=number_of_cells, dtype=np.int8)

    counter = np.int32(0)
    for item in allowed_cells:
      if meshio_mesh_dic.get(item) is not None:
        cells_item = np.array(meshio_mesh_dic[item], dtype=np.int32)
        cells_type[counter:counter + len(cells_item)] = cell_type_dic[item]
        append(cells, cells_item, counter)
        counter += len(cells_item)

    return cells, cells_type, max_cell_nodeid, max_cell_faceid, max_face_nodeid
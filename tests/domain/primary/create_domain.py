import numpy as np
import meshio
import time
import warnings
import manapy_domain
from create_domain_utils import (append,
                                  append_1d,
                                  count_max_node_cellid,
                                  create_node_cellid,
                                  count_max_cell_cellnid,
                                  create_cell_cellnid,
                                  create_info,
                                  create_cell_info_2d,
                                  get_max_node_faceid,
                                  get_node_faceid,
                                  define_face_and_node_name,
                                  _intersect_nodes,
                                  create_cell_info_3d,
                                  create_cellfid
                                  )


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
          print(f"{k2} => {k1}")
      cells_dict = mesh.cells
      # raise NotImplementedError
    else:
      # print(mesh.cell_data_dict['gmsh:physical']['triangle'])
      # print(mesh.cells_dict['triangle'])
      cells_dict = mesh.cells_dict
      cell_data_dict = mesh.cell_data_dict
    points = mesh.points

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

    counter = 0
    for k in physicals_key:
      if physicals.get(k) is not None:
        append(phy_faces, cells_dict[k], counter)
        append_1d(phy_faces_name, physicals[k], counter)
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
    cells_type = np.zeros(shape=number_of_cells, dtype=np.uint8)

    counter = 0
    for item in allowed_cells:
      if meshio_mesh_dic.get(item) is not None:
        cells_item = meshio_mesh_dic[item]
        cells_type[counter:counter + len(cells_item)] = cell_type_dic[item]
        append(cells, cells_item, counter)
        counter += len(cells_item)

    return cells, cells_type, max_cell_nodeid, max_cell_faceid, max_face_nodeid

class LocalDomainStruct:
  # TODO create cellfid for partitioning with customization
  def __init__(self):
    self.nodes = None # [[node x, y, z]]
    self.cells = None # [[cells nodes]]
    self.cells_type = None # [cell type]
    self.phy_faces = None # [[physical face nodes]]
    self.phy_faces_name = None # [physical face name]
    self.cell_loctoglob = None # [cell global index]
    self.node_loctoglob = None # [node global index]
    self.halo_neighsub = None # [sub domain id]
    self.node_halos = None # [node, global halo cell index, ...]
    self.dim = None
    self.float_precision = None
    self.max_cell_nodeid = None
    self.max_cell_faceid = None
    self.max_face_nodeid = None



    # To be copied if one partition is specified (recalculated on subdomains)
    self.node_cellid = None
    self.cell_cellnid = None

    ## Temporarily
    self.max_phy_face_nodeid = None
    self.map_cells = {}
    self.map_nodes = {}
    self.map_phy_faces = {}

class Domain:

  def __init__(self, mesh: 'Mesh', float_precision: 'str'):
    if float_precision != 'float32' and float_precision != 'float64':
      raise ValueError('Invalid float precision argument')

    self.nodes = np.array(mesh.points).astype(np.float32)
    self.cells = mesh.cells
    self.cells_type = mesh.cells_type
    self.max_cell_nodeid = mesh.max_cell_nodeid
    self.max_cell_faceid = mesh.max_cell_faceid
    self.max_face_nodeid = mesh.max_face_nodeid
    self.phy_faces = mesh.phy_faces
    self.phy_faces_name = mesh.phy_faces_name
    self.dim = mesh.dim
    self.float_precision = float_precision
    self.nb_nodes = np.int32(len(self.nodes))
    self.nb_cells = np.int32(len(self.cells))


    self.start = time.time()
    print("node_cellid")
    self.node_cellid = self.create_node_cellid(self.cells, self.nb_nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellnid")
    self.cell_cellnid = self.create_cell_cellnid(self.cells, self.node_cellid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellfid")
    self.cell_cellfid = self._create_cellfid(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

  # ###############################
  # ###############################

  @staticmethod
  def create_node_cellid(cells: 'int[:, :]', nb_nodes: 'int'):
    # Count max node cellid
    res = np.zeros(shape=nb_nodes, dtype=np.int32)
    count_max_node_cellid(cells, res)
    max_node_cellid = np.max(res)

    # Create node cellid
    node_cellid = np.zeros(shape=(nb_nodes, max_node_cellid + 1), dtype=np.int32)
    create_node_cellid(cells, node_cellid)
    return node_cellid

  @staticmethod
  def create_cell_cellnid(cells: 'int[:, :]', node_cellid: 'int[:, :]'):
    # Count max cell cellnid
    visited = np.zeros(cells.shape[0], dtype=np.int32)
    max_cell_cellnid = count_max_cell_cellnid(cells, node_cellid, visited)

    # Create cell cellnid
    cell_cellnid = np.zeros(shape=(len(cells), max_cell_cellnid + 1), dtype=np.int32)
    create_cell_cellnid(cells, node_cellid, cell_cellnid)
    return cell_cellnid

  @staticmethod
  def _create_cellfid(
    cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int',
  ):
    nb_cells = len(cells)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=np.int32)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=np.int32)
    cell_cellfid = np.ones(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32) * -1
    cell_cellfid[:, -1] = 0

    create_cellfid(
      cells,
      node_cellid,
      cell_type,
      tmp_cell_faces,
      tmp_size_info,
      cell_cellfid,
    )



    return cell_cellfid

  def create_sub_domains(self, nb_parts: 'int'):
    if nb_parts == 1:
      local_domain = LocalDomainStruct()

      local_domain.nodes = self.nodes
      local_domain.cells = self.cells
      local_domain.cells_type = self.cells_type
      local_domain.phy_faces = self.phy_faces
      local_domain.phy_faces_name = self.phy_faces_name
      local_domain.cell_loctoglob = np.zeros(shape=0, dtype=np.uint32)
      local_domain.node_loctoglob = np.zeros(shape=0, dtype=np.uint32)
      local_domain.halo_neighsub = []
      local_domain.node_halos = []
      local_domain.dim = self.dim
      local_domain.float_precision = self.float_precision
      local_domain.max_cell_nodeid = self.max_cell_nodeid
      local_domain.max_cell_faceid = self.max_cell_faceid
      local_domain.max_face_nodeid = self.max_face_nodeid

      return [local_domain]

    d_cell_cellnid = self.cell_cellnid
    d_node_cellid = self.node_cellid
    d_cells = self.cells
    d_cells_type = self.cells_type
    d_nodes = self.nodes
    d_phy_faces = self.phy_faces
    d_phy_faces_name = self.phy_faces_name

    def get_max_info(cell_type):
      # return (max_cell_faceid, max_face_nodeid, max_cell_nodeid)
      if cell_type == 1: #triangle
        return 3, 2, 3
      elif cell_type == 2: #'quad'
        return 4, 2, 4
      elif cell_type == 3: #'tetra'
        return 4, 3, 4
      elif cell_type == 4: #'hexahedron'
        return 6, 4, 8
      elif cell_type == 5: #'pyramid'
        return 5,4, 5

    print("Create graph")
    # graph = []
    # for item in d_cell_cellnid:
    #   c = item[0:item[-1]]
    #   graph.append(c)

    # print("Partition the graph")
    # options = pymetis.Options()
    # options.__setattr__("contig", 1)
    # cutcount, part_vert = pymetis.part_graph(nparts=nb_parts, adjacency=graph, options=options)

    graph = self.cell_cellfid
    part_vert = manapy_domain.make_n_part(graph, nb_parts)
    print(part_vert)
    print(np.unique(np.array(part_vert)))
    # nb_parts = len(np.unique(part_vert))
    print(f"Number of parts: {nb_parts}")
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    local_domains = [LocalDomainStruct() for _ in range(nb_parts)]

    for p in range(nb_parts):
      local_domains[p].max_cell_nodeid = -1
      local_domains[p].max_phy_face_nodeid = -1
      local_domains[p].map_cells = {}
      local_domains[p].map_nodes = {}
      local_domains[p].map_phy_faces = {}
      local_domains[p].max_cell_faceid = -1
      local_domains[p].max_face_nodeid = -1
      local_domains[p].max_cell_nodeid = -1
      local_domains[p].halo_neighsub = set()
      local_domains[p].node_halos = {}

    print("Create Cell Nodes Parts")
    for i in range(d_cells.shape[0]):
      cell_nodes = d_cells[i]
      p = part_vert[i]

      max_info = get_max_info(d_cells_type[i])
      local_domains[p].max_cell_faceid = max(max_info[0], local_domains[p].max_cell_faceid)
      local_domains[p].max_face_nodeid = max(max_info[1], local_domains[p].max_face_nodeid)
      local_domains[p].max_cell_nodeid = max(max_info[2], local_domains[p].max_cell_nodeid)


      cell_local_index = len(local_domains[p].map_cells)
      local_domains[p].map_cells[i] = cell_local_index
      for j in range(cell_nodes[-1]):
        node = cell_nodes[j]
        if local_domains[p].map_nodes.get(node) is None:
          local_domains[p].map_nodes[node] = len(local_domains[p].map_nodes)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("Create Physical Faces Parts")
    # ##############################
    # PhyFaces
    ################################
    intersect_cell = np.zeros(shape=2, dtype=np.int32)
    nb_phyfaces = 0
    for i in range(d_phy_faces.shape[0]):
      phy_face = d_phy_faces[i]
      # For the “phy_face” to belong to part ‘p’, it must belong to a cell in part ‘p’.
      _intersect_nodes(phy_face, phy_face[-1], d_node_cellid, intersect_cell)
      if intersect_cell[0] != -1:
        p = part_vert[intersect_cell[0]]
        local_domains[p].max_phy_face_nodeid = max(phy_face[-1], local_domains[p].max_phy_face_nodeid)
        local_domains[p].map_phy_faces[i] = len(local_domains[p].map_phy_faces)
        nb_phyfaces += 1
    if nb_phyfaces != len(d_phy_faces):
      warnings.warn(f"Warning not all the physical faces match the domain faces !! {nb_phyfaces} Where the number of physical faces is {len(d_phy_faces)}", category=RuntimeWarning)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    # ##############################
    # Local Domain P
    ################################
    print("Create Local Domains Parts")
    for p in range(nb_parts):
      map_cells = local_domains[p].map_cells
      map_nodes = local_domains[p].map_nodes
      map_phy_faces = local_domains[p].map_phy_faces

      nb_cells = len(map_cells)
      nb_nodes = len(map_nodes)
      nb_phy_faces = len(map_phy_faces)
      max_cell_nodeid = local_domains[p].max_cell_nodeid
      max_phy_face_nodeid = local_domains[p].max_phy_face_nodeid

      int_dtype = d_cells.dtype
      nodes_dtype = d_nodes.dtype

      cells = np.zeros(shape=(nb_cells, max_cell_nodeid + 1), dtype=int_dtype)
      cells_type = np.zeros(shape=nb_cells, dtype=np.int8)
      cell_loctoglob = np.zeros(shape=nb_cells, dtype=int_dtype)

      nodes = np.zeros(shape=(nb_nodes, 3), dtype=nodes_dtype)
      node_loctoglob = np.zeros(shape=nb_nodes, dtype=int_dtype)

      phy_faces = np.zeros(shape=(nb_phy_faces, max_phy_face_nodeid + 1), dtype=int_dtype)
      phy_faces_name = np.zeros(shape=nb_phy_faces, dtype=int_dtype)

      halo_neighsub = set()
      node_halos = []

      # Cells, CellsType, CellsLocToGlob
      for k in map_cells.keys():
        cell_local_index = map_cells[k]
        cells[cell_local_index] = d_cells[k]
        cells_type[cell_local_index] = d_cells_type[k]
        cell_loctoglob[cell_local_index] = k
        # set new node name
        for j in range(cells[cell_local_index, -1]):
          cells[cell_local_index, j] = map_nodes[cells[cell_local_index, j]]

      # Nodes, NodesLocToGlob, HaloNeighSub, NodeHalos
      for k in map_nodes.keys():
        l_index = map_nodes[k]
        nodes[l_index] = d_nodes[k]
        node_loctoglob[l_index] = k

        # Halos
        for j in range(d_node_cellid[k, -1]):
          neighbor_cell = d_node_cellid[k, j]
          neighbor_part = part_vert[neighbor_cell]
          if p != neighbor_part:
            halo_neighsub.add(neighbor_part)
            node_halos.append(l_index)
            node_halos.append(neighbor_cell)

      # PhyFaces, PhyFacesName
      for k in map_phy_faces.keys():
        cell_local_index = map_phy_faces[k]
        phy_faces[cell_local_index] = d_phy_faces[k]
        phy_faces_name[cell_local_index] = d_phy_faces_name[k]
        # set new node name
        for j in range(phy_faces[cell_local_index, -1]):
          phy_faces[cell_local_index, j] = map_nodes[phy_faces[cell_local_index, j]]

      local_domains[p].nodes = nodes
      local_domains[p].cells = cells
      local_domains[p].cells_type = cells_type
      local_domains[p].phy_faces = phy_faces
      local_domains[p].phy_faces_name = phy_faces_name
      local_domains[p].cell_loctoglob = cell_loctoglob
      local_domains[p].node_loctoglob = node_loctoglob
      local_domains[p].halo_neighsub = list(halo_neighsub)
      local_domains[p].node_halos = node_halos
      local_domains[p].dim = self.dim
      local_domains[p].float_precision = self.float_precision
      #max_cell_nodeid Assigned above
      #max_cell_faceid Assigned above
      #max_face_nodeid Assigned above

    print(f"Execution time: {time.time() - self.start:.6f} seconds")
    return local_domains

  def c_create_sub_domains(self, nb_parts: 'int'):
    if nb_parts == 1:
      local_domain = LocalDomainStruct()

      local_domain.nodes = self.nodes
      local_domain.cells = self.cells
      local_domain.cells_type = self.cells_type
      local_domain.phy_faces = self.phy_faces
      local_domain.phy_faces_name = self.phy_faces_name
      local_domain.cell_loctoglob = np.zeros(shape=0, dtype=np.uint32)
      local_domain.node_loctoglob = np.zeros(shape=0, dtype=np.uint32)
      local_domain.halo_neighsub = []
      local_domain.node_halos = []
      local_domain.dim = self.dim
      local_domain.float_precision = self.float_precision
      local_domain.max_cell_nodeid = self.max_cell_nodeid
      local_domain.max_cell_faceid = self.max_cell_faceid
      local_domain.max_face_nodeid = self.max_face_nodeid

      return [local_domain]

    print("Start creating sub domains")
    res = manapy_domain.create_sub_domains(
      self.cell_cellfid,
      self.node_cellid,
      self.cells,
      self.cells_type,
      self.nodes,
      self.phy_faces,
      self.phy_faces_name,
      nb_parts,
    )

    local_domains = []
    for item in res:
      l_domain = LocalDomainStruct()
      l_domain.nodes = item[0]
      l_domain.cells = item[1]
      l_domain.cells_type = item[2]
      l_domain.phy_faces = item[3]
      l_domain.phy_faces_name = item[4]
      l_domain.cell_loctoglob = item[5]
      l_domain.node_loctoglob = item[6]
      l_domain.halo_neighsub = item[7]
      l_domain.node_halos = item[8]
      local_domains.append(l_domain)

    print(f"Execution time: {time.time() - self.start:.6f} seconds")
    return local_domains

class LocalDomain:
  # TODO int32 -> uint32
  def __init__(self, local_domain_struct: 'LocalDomainStruct'):
    self.nodes = local_domain_struct.nodes
    self.cells = local_domain_struct.cells
    self.cells_type = local_domain_struct.cells_type
    self.phy_faces = local_domain_struct.phy_faces
    self.phy_faces_name = local_domain_struct.phy_faces_name
    self.cell_loctoglob = local_domain_struct.cell_loctoglob
    self.node_loctoglob = local_domain_struct.node_loctoglob
    self.halo_neighsub = local_domain_struct.halo_neighsub
    self.node_halos = local_domain_struct.node_halos
    self.dim = local_domain_struct.dim
    self.float_precision = local_domain_struct.float_precision
    self.max_cell_nodeid = local_domain_struct.max_cell_nodeid
    self.max_cell_faceid = local_domain_struct.max_cell_faceid
    self.max_face_nodeid = local_domain_struct.max_face_nodeid
    self.node_cellid = local_domain_struct.node_cellid
    self.cell_cellnid = local_domain_struct.cell_cellnid
    self.nb_nodes = np.uint32(len(self.nodes))
    self.nb_cells = np.uint32(len(self.cells))

    self.start = time.time()
    print("node_cellid")
    self.node_cellid = self._create_node_cellid(self.cells, self.nb_nodes, local_domain_struct.node_cellid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("cell_cellnid")
    self.cell_cellnid = self._create_cell_cellnid(self.cells, self.node_cellid, local_domain_struct.cell_cellnid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("_create_info")
    (
      self.faces,
      self.cell_faceid,
      self.face_cellid,
      self.cell_cellfid
    ) = self._create_info(self.cells, self.node_cellid, self.cells_type, self.max_cell_faceid, self.max_face_nodeid)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    print("Create face and node names")
    (
      self.face_name,
      self.node_name,
      self.node_phyfaceid,
    ) = self._define_face_and_node_name(self.phy_faces, self.phy_faces_name, self.faces)
    print(f"Execution time: {time.time() - self.start:.6f} seconds")

    # (
    #   self.cell_volume,
    #   self.cell_center
    # ) = self._create_cell_info(self.cells, self.nodes)


  def _create_node_cellid(self, cells: 'int[:, :]', nb_nodes: 'int', g_node_cellid):
    if g_node_cellid is not None:
      return g_node_cellid
    return Domain.create_node_cellid(cells, nb_nodes)

  def _create_cell_cellnid(self, cells: 'int[:, :]', node_cellid: 'int[:, :]', g_cell_cellnid):
    if g_cell_cellnid is not None:
      return g_cell_cellnid
    return Domain.create_cell_cellnid(cells, node_cellid)

  @staticmethod
  def _create_info(
          cells: 'int[:, :]',
    node_cellid: 'int[:, :]',
    cell_type: 'int[:]',
    max_cell_faceid: 'int',
    max_face_nodeid: 'int',
  ):
    # ? Create tables
    nb_cells = len(cells)
    # tmp_cell_faces = np.zeros(shape=(nb_cells, max_cell_faceid, max_face_nodeid), dtype=np.int32)
    # tmp_size_info = np.zeros(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces = np.zeros(shape=(max_cell_faceid, max_face_nodeid), dtype=np.int32)
    tmp_size_info = np.zeros(shape=(max_cell_faceid + 1), dtype=np.int32)
    tmp_cell_faces_map = np.zeros(shape=(nb_cells, max_cell_faceid * 2 + 1), dtype=np.int32)
    apprx_nb_faces = nb_cells * max_cell_faceid # TODO ((nb_cells * max_cell_faceid + boundary_faces) / 2)
    faces = np.ones(shape=(apprx_nb_faces, max_face_nodeid + 1), dtype=np.int32) * -1
    faces[:, -1] = 0
    cell_faceid = np.ones(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32) * -1
    cell_faceid[:, -1] = 0
    face_cellid = np.ones(shape=(apprx_nb_faces, 2), dtype=np.int32) * -1
    cell_cellfid = np.ones(shape=(nb_cells, max_cell_faceid + 1), dtype=np.int32) * -1
    cell_cellfid[:, -1] = 0
    faces_counter = np.zeros(shape=1, dtype=np.int32)

    create_info(
      cells,
      node_cellid,
      cell_type,
      tmp_cell_faces,
      tmp_size_info,
      tmp_cell_faces_map,
      faces,
      cell_faceid,
      face_cellid,
      cell_cellfid,
      faces_counter
    )

    faces = faces[:faces_counter[0]]
    face_cellid = face_cellid[:faces_counter[0]]

    return (
      faces,
      cell_faceid,
      face_cellid,
      cell_cellfid
    )

  def _define_face_and_node_name(self,
                                 phy_faces: 'int[:, :]',
                                 phy_faces_name: 'int[:]',
                                 faces: 'int[:, :]',
                                 ):
    nb_nodes = self.nb_nodes
    face_name = np.zeros(shape=faces.shape[0], dtype=np.int32)
    node_name = np.zeros(shape=nb_nodes, dtype=np.int32)

    # count max_node_faceid
    tmp = np.zeros(shape=nb_nodes, dtype=np.int32)
    get_max_node_faceid(phy_faces, tmp)
    max_node_faceid = np.max(tmp)

    # create node_phyfaceid
    node_phyfaceid = np.zeros(shape=(nb_nodes, max_node_faceid + 1), dtype=np.int32)
    get_node_faceid(phy_faces, node_phyfaceid)

    define_face_and_node_name(phy_faces, phy_faces_name, faces, node_phyfaceid, face_name, node_name)
    return (
      face_name,
      node_name,
      node_phyfaceid
    )


  def _create_cell_info(self, cells, nodes):
    nb_cells = len(cells)
    cell_volume = np.ones(shape=nb_cells, dtype=self.float_precision)
    cell_center = np.zeros(shape=(nb_cells, self.dim), dtype=self.float_precision)
    if self.dim == 2:
      create_cell_info_2d(cells, nodes, cell_volume, cell_center)
    else:
      create_cell_info_3d(cells, nodes, cell_volume, cell_center)
    return (
      cell_volume,
      cell_center
    )


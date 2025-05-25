"""
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



"""
"""

def _intersect_nodes(face_nodes: 'int[:]', nb_nodes: 'int', node_cellid: 'int[:, :]',
                     intersect_cell: 'int[:]'):

index = 0

intersect_cell[0] = -1
intersect_cell[1] = -1

cells = node_cellid[face_nodes[0]]
for i in range(cells[-1]):
  intersect_cell[index] = cells[i]
  for j in range(1, nb_nodes):
    if _binary_search(node_cellid[face_nodes[j]], cells[i]) == -1:
      intersect_cell[index] = -1
      break
  if intersect_cell[index] != -1:
    index = index + 1
  if index >= 2:
    return
"""
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
  cutcount, part_vert = manapy_domain.make_n_part(graph, nb_parts)
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

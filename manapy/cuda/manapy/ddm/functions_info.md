# Domain.py

  `_read_partition` 🔁
    read hdf5 file and parse it 
    get self._cells._nodeid (cells_ids -> cell nodes) (len(cells_ids))
        self._nodes._vertex (nodes) (len(nodes))
    
    -> it return other things

  `_define_bounds` 🔁
    define the boundaries of the geometry
    [min(node.v[x]), max(node.v[x])]
    [min(node.v[y]), max(node.v[y])]
    [min(node.v[z]), max(node.v[z])]

    -> return : self._bounds

  `_compute_cells_info` 🔁
    Compute the center and volume of cells
    
    -> uses : Compute_2dcentervolumeOfCell, Compute_3dcentervolumeOfCell
    -> return : self._cells._center, self._cells._volume 

  `_make_neighbors` ✅
    return create_node_cellid
      Section 1: create neighboring cells of a node (cellid)
      Section 2: create neighboring cells if a cell by nodes of the cell.
    return self._nodes._cellid and self._cells._cellnid (node -> ncell || cell -> ncellsbynode)

    it also calculate count_max_neighboring_cells_of_node
    if also calculate the maximum number of neighboring cells per cell's nodes across the mesh
    to create the table (self._nodes._cellid and self._cells._cellnid) inside the function
    
    Redefined in test.py as
    - create_node_neighboring_cells
    - create_cell_neighbors_by_node
    - count_max_neighboring_cells_of_node
    - get_max_neighboring_cells_per_cell_nodes
  

  `_define_eltypes` 🔁
    - self._typeOfCells (type_of_cell (2d'quad, triangle' | 3d'tetra, hexahedron, pyramid') => cells)
    - self._maxcellnodeid Max number of nodes of cell
    - self._nbOfTriangles Number of triangles in the mesh
    - self._nbOfQuad Number of quad in the mesh
    - self._maxfacenid Max number node on faces (always 2 in 2D)
    - self._maxcellfid Max number of faces of cells
    - self._nbOfTetra Number of tetra in the mesh
    - self._nbOfpyra Number of pyramid in the mesh
    - self._nbOfQuad Number of hexahedron in the mesh

  `_create_faces_cons` ✅
    - Create faces
    - Create cells with their corresponding faces. 
    - Create neighboring cells for each face.

    Used:
      create_2dfaces create_3dfaces  ==> self._faces._nodeid
      create_cell_faceid ==> self._cells._faceid
      create_cellsOfFace ==> self._faces._cellid
      create_NeighborCellByFace ==> self._cells._cellfid

    Return:
      self._nbfaces Number of faces in the mesh
      self._faces._nodeid [Face => nodes ids of the face]
      self._cells._faceid [Cell => faces id of the cell]
      self._faces._cellid [Face => neighboring cells of the face]
      self._cells._cellfid [Cell => neighboring cells by face of the cell]


    Redefined in test.py
    - create_info

  


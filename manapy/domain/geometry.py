"""
CFD mesh data structures for an unstructured finite volume solver with MPI support.

This module defines the core topology and geometry containers:
- Cell: control volumes
- Node: mesh vertices
- Face: interfaces between cells (flux computation)
- Halo: inter-process (MPI) exchange of cells
- Ghost: boundary cells

------------------------------------------------------------
Supported cell types

cell_type   nb_nodes   nb_faces   max_face_nodeid
TRIANGLE=1    3          3          2
QUAD=2        4          4          2
TETRA=3       4          4          3
HEXAHEDRON=4  8          6          4
PYRAMID=5     5          5          4

Notes:
- nb_nodes: number of vertices per cell
- nb_faces: number of faces per cell
- max_face_nodeid: maximum number of nodes per face

------------------------------------------------------------
Face definitions (meshio convention)

'triangle':
    {'line': [[0, 1], [1, 2], [2, 0]]}

'rectangle':
    {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]}

'tet':
    {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]}

'hex':
    {'quad': [
        [0, 1, 2, 3],
        [0, 1, 4, 5],
        [1, 2, 5, 6],
        [2, 3, 6, 7],
        [0, 3, 4, 7],
        [4, 5, 6, 7]
    ]}

'pyr':
    {
        'quad': [[0, 1, 2, 3]],
        'tri':  [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]
    }

"""

class Cell:
    """ """
    __slots__ = [
        '_nbcells',        # total number of cells (int)

        '_nodeid',         # connectivity: cell -> node indices
                           # shape: (nb_cells, max_cell_nodeid + 1)
                           # layout:
                           # [
                           #   [node_0, node_1, ..., node_(k-1), k],
                           #   ...
                           # ]
                           # where the last entry is the number of valid nodes

        '_type',           # Cell type
                           # shape int8[nb_cells,]
                           # layout [cell_type1, ...]
                           # where cell_type
                           # TRIANGLE = 1
                           # QUAD = 2
                           # TETRA = 3
                           # HEXAHEDRON = 4
                           # PYRAMID = 5

        '_faceid',         # connectivity: cell -> face indices
                           # shape: (nb_cells, max_cell_faceid + 1)
                           # layout:
                           # [
                           #   [face_0, face_1, ..., face_(k-1), k],
                           #   ...
                           # ]
                           # where the last entry is the number of valid faces

        '_cellfid',        # neighboring cells across faces (face adjacency)
                           # shape: (nb_cells, max_cell_faceid + 1)
                           # layout:
                           # [
                           #   [nbr_cell_0, ..., nbr_cell_(k-1), k],
                           #   ...
                           # ]

        '_cellnid',        # neighboring cells sharing nodes (vertex adjacency)
                           # shape: (nb_cells, max_cell_nodeid + 1)
                           # layout:
                           # [
                           #   [nbr_cell_0, ..., nbr_cell_(k-1), k],
                           #   ...
                           # ]

        '_halonid',        # neighboring cells on other MPI ranks (halo region)
                           # shape: (nb_cells, max_cell_halonid + 1)
                           # layout:
                           # [
                           #   [halo_cell_0, ..., halo_cell_(k-1), k],
                           #   ...
                           # ]
                           # values index into Halo.halosext

        '_ghostnid',       # ghost cells associated with boundaries
                           # shape: (nb_cells, max_cell_ghostnid + 1)
                           # layout:
                           # [
                           #   [ghost_cell_0, ..., ghost_cell_(k-1), k],
                           #   ...
                           # ]
                           # values index into Ghost.info_int / Ghost.info_flt

        '_haloghostnid',   # ghost cells received from other MPI ranks
                           # shape: (nb_cells, max_cell_haloghostnid + 1)
                           # layout:
                           # [
                           #   [halo_ghost_0, ..., halo_ghost_(k-1), k],
                           #   ...
                           # ]
                           # values index into Ghost.ext_info_int / Ghost.ext_info_flt

        '_center',         # cell centers
                           # shape: (nb_cells, 3)
                           # layout: [[x, y, z], ...]

        '_volume',         # cell volume (3D) or area (2D)
                           # shape: (nb_cells,)
                           # layout: [vol_0, vol_1, ...]

        '_nf',             # face normals per cell
                           # shape: (nb_cells, max_cell_faceid, 3)
                           # layout:
                           # [
                           #   [[nx, ny, nz], ...],
                           #   ...
                           # ]

        '_loctoglob',      # local-to-global cell index mapping
                           # shape: (nb_cells,)
                           # layout: [global_id_0, global_id_1, ...]

        '_tc',             # global cell indexing (stored only on rank 0)
                           # shape: (nb_cells_global,)
                           # layout: concatenation of all loctoglob arrays

        # TODO description need verification
        '_periodicnid',    # periodic node mapping
                           # shape: (nb_cells, 2)
                           # layout: [[node, periodic_node], ...]

        '_periodicfid',    # periodic face mapping
                           # shape: (nb_cells,)
                           # layout: [periodic_face_id, ...]

        '_shift',          # periodic translation vectors
                           # shape: (nb_cells, 3)
                           # layout: [[dx, dy, dz], ...]
    ]

    def __init__(self):
        pass
        
    @property
    def nbcells(self):
        return self._nbcells
    
    @property
    def nodeid(self):
        return self._nodeid

    @property
    def type(self):
        return self._type

    @property
    def faceid(self):
        return self._faceid
    
    @property
    def cellfid(self):
        return self._cellfid
    
    @property
    def cellnid(self):
        return self._cellnid
    
    @property
    def halonid(self):
        return self._halonid
    
    @property
    def ghostnid(self):
        return self._ghostnid
    
    @property
    def haloghostnid(self):
        return self._haloghostnid

    @property
    def center(self):
        return self._center
    
    @property
    def volume(self):
        return self._volume
    
    @property
    def nf(self):
        return self._nf

    @property
    def loctoglob(self):
        return self._loctoglob
    
    @property
    def tc(self):
        return self._tc
    
    @property
    def periodicnid(self):
        return self._periodicnid
    
    @property
    def periodicfid(self):
        return self._periodicfid
    
    @property
    def shift(self):
        return self._shift
            
class Node:
    """ """
    __slots__ = [
        '_nbnodes',        # total number of nodes (int)

        '_vertex',         # node coordinates
                           # shape: (nb_nodes, 3)
                           # layout: [[x, y, z], ...]


        '_oldname',        # boundary identifier before partitioning
                           # shape: (nb_nodes,)
                           # layout: [old_name_id, ...]
                           # 0 = interior node
                           # others = physical boundaries (in=1, out=2, bottom=3, upper=4, front=5, back=6)
                           # Note node_name = Min(node neighboring faces boundary identifier)

        '_name',            # boundary identifier after partitioning
                            # shape: (nb_nodes,)
                            # same as _oldname except 10 for nodes at MPI boundary (halo)

        '_cellid',         # connectivity: node -> cells
                           # shape: (nb_nodes, max_node_cellid + 1)
                           # layout:
                           # [
                           #   [cell_0, cell_1, ..., cell_(k-1), k],
                           #   ...
                           # ]
                           # where the last entry is the number of node neighboring cells

        '_ghostid',        # connectivity: node -> ghost cells (boundary)
                           # shape: (nb_nodes, max_node_ghostid + 1)
                           # layout:
                           # [
                           #   [ghost_0, ..., ghost_(k-1), k],
                           #   ...
                           # ]
                           # values index into Ghost.info_int / Ghost.info_ext

        '_haloghostid',    # connectivity: node -> ghost cells from other MPI ranks
                           # shape: (nb_nodes, max_node_haloghostid + 1)
                           # layout:
                           # [
                           #   [halo_ghost_0, ..., halo_ghost_(k-1), k],
                           #   ...
                           # ]
                           # values index into Ghost.ext_info_*

        '_loctoglob',      # local-to-global node index mapping
                           # shape: (nb_nodes,)
                           # layout: [global_id_0, global_id_1, ...]

        '_halonid',        # connectivity: node -> halo cells (MPI neighbors)
                           # shape: (nb_nodes, max_node_halonid + 1)
                           # layout:
                           # [
                           #   [halo_cell_0, ..., halo_cell_(k-1), k],
                           #   ...
                           # ]
                           # values index into Halo.halosext

        # TODO description need verification
        '_R_x',            # accumulated Rx = sum(cell_center.x - node.x)
                           # across node_cellid, node_ghostid, node_periodicid, node_haloid, node_haloghostid
                           # shape: (nb_nodes,)

        '_R_y',            # accumulated Ry = sum(cell_center.y - node.y)
                           # across node_cellid, node_ghostid, node_periodicid, node_haloid, node_haloghostid
                           # shape: (nb_nodes,)

        '_R_z',            # accumulated Rz = sum(cell_center.z - node.z)
                           # across node_cellid, node_ghostid, node_periodicid, node_haloid, node_haloghostid
                           # shape: (nb_nodes,)

        '_number',         # number of contributing neighboring entities
                           # (cells + ghosts + halo + periodic)
                           # shape: (nb_nodes,)

        '_lambda_x',       # reconstruction coefficient (least-squares gradient)
                           # shape: (nb_nodes,)

        '_lambda_y',       # reconstruction coefficient (least-squares gradient)
                           # shape: (nb_nodes,)

        '_lambda_z',       # reconstruction coefficient (least-squares gradient)
                           # shape: (nb_nodes,)

        '_periodicid',     # connectivity: node -> periodic cells
                           # shape: (nb_nodes, max_node_periodicid + 1)
                           # layout:
                           # [
                           #   [periodic_cell_0, ..., periodic_cell_(k-1), k],
                           #   ...
                           # ]
                           # used with shift to reconstruct periodic neighbors
    ]
     
    def __init__(self, nbnodes=None):
        pass
    @property
    def nbnodes(self):
        return self._nbnodes
    
    @property
    def vertex(self):
        return self._vertex
    
    @property
    def name(self):
        return self._name
    
    @property
    def oldname(self):
        return self._oldname
    
    @property
    def cellid(self):
        return self._cellid
    
    @property
    def ghostid(self):
        return self._ghostid
    
    @property
    def haloghostid(self):
        return self._haloghostid
    
    @property
    def loctoglob(self):
        return self._loctoglob
    
    @property
    def halonid(self):
        return self._halonid

    @property
    def periodicid(self):
        return self._periodicid
    
    @property
    def R_x(self):
        return self._R_x
    
    @property
    def R_y(self):
        return self._R_y
    
    @property
    def R_z(self):
        return self._R_z
    
    @property
    def number(self):
        return self._number
    
    @property
    def lambda_x(self):
        return self._lambda_x
    
    @property
    def lambda_y(self):
        return self._lambda_y
    
    @property
    def lambda_z(self):
        return self._lambda_z

class Face:
    """ """
    __slots__ = [
        '_nbfaces',        # total number of faces (int)

        '_nodeid',         # connectivity: face -> nodes
                           # shape: (nb_faces, max_face_nodeid + 1)
                           # layout:
                           # [
                           #   [node_0, node_1, ..., node_(k-1), k],
                           #   ...
                           # ]
                           # where the last entry is the number of valid nodes.

        '_cellid',         # connectivity: face -> adjacent cells
                           # shape: (nb_faces, 2)
                           # layout: [[left_cell, right_cell], ...]
                           # right_cell = -1 if boundary face

        '_name',           # face boundary identifier after partitioning
                           # shape: (nb_faces,)
                           # values:
                           # 0 = interior face
                           # 10 = MPI boundary (halo)
                           # others = physical boundaries (in=1, out=2, bottom=3, upper=4, front=5, back=6)

        '_oldname',        # face boundary identifier before partitioning
                           # shape: (nb_faces,)

        '_normal',         # face normal vector (oriented outward from cellid[:, 0] aka left cell)
                           # shape: (nb_faces, 3)
                           # layout: [[nx, ny, nz], ...]
                           # Note: Looping through cell.faceid does not guarantee that the face normal is outward for the current cell (the cell may be the right cell). use cell._nf

        '_mesure',         # face measure (length in 2D, area in 3D)
                           # shape: (nb_faces,)

        '_center',         # face center coordinates
                           # shape: (nb_faces, 3)
                           # layout: [[x, y, z], ...]

        '_halofid',        # mapping to halo cells
                           # shape: (nb_faces,)
                           # value = index into Halo.halosext, -1 if none

        '_tangent',        # face tangent vector
                           # shape: (nb_faces, 3) in 2D shape is 0

        '_ghost_id',       # mapping to ghost cells
                           # shape: (nb_faces,)
                           # value = index into Ghost.info_* or -1 if none

        # TODO description need verification
        '_param1',         # gradient reconstruction coefficient
                           # shape: (nb_faces,)

        '_param2',         # gradient reconstruction coefficient
                           # shape: (nb_faces,)

        '_param3',         # gradient reconstruction coefficient
                           # shape: (nb_faces,)

        '_param4',         # additional coefficient (used in 2D schemes)
                           # shape: (nb_faces,)

        '_f_1',            # geometric flux vector contribution
                           # shape: (nb_faces, dim)

        '_f_2',            # geometric flux vector contribution
                           # shape: (nb_faces, dim)

        '_f_3',            # geometric flux vector contribution (mainly 2D)
                           # shape: (nb_faces, dim)

        '_f_4',            # geometric flux vector contribution (mainly 2D)
                           # shape: (nb_faces, dim)

        '_airDiamond',     # diamond cell measure used in gradient reconstruction
                           # shape: (nb_faces,)

        '_binormal',       # binormal vector (3D only)
                           # shape: (nb_faces, 3)
                           # computed as:
                           # u = nodes[node_1] - nodes[node_0]
                           # binormal = 0.5 * cross(u, normal)

        '_dist_ortho',  # orthogonal distance associated with each face
                        # shape: (nb_faces,) shape=0 in 2D
                        # definition:
                        # distance measured along the face normal direction
                        # between cell centers and their orthogonal projection onto the face

                        # interior faces:
                        # dist = d(K → projection_on_face_normal) + d(L → projection_on_face_normal)
                        # where K and L are the two adjacent cells

                        # boundary faces:
                        # dist = 2 * d(K → projection_on_face_normal)
                        # (symmetric approximation using a mirrored ghost point)

                        # computed using:
                        # projection = cell_center - ((cell_center - face_center) · normal) * normal
    ]

    def __init__(self):
        pass
        
    @property
    def nbfaces(self):
        return self._nbfaces
    
    @property
    def nodeid(self):
        return self._nodeid
    
    @property
    def cellid(self):
        return self._cellid
    
    @property
    def name(self):
        return self._name
    
    @property
    def oldname(self):
        return self._oldname
    
    @property
    def normal(self):
        return self._normal
    
    @property
    def mesure(self):
        return self._mesure
    
    @property
    def center(self):
        return self._center
    
    @property
    def dist_ortho(self):
        return self._dist_ortho

    @property
    def halofid(self):
        return self._halofid
    
    @property
    def param1(self):
        return self._param1
    
    @property
    def param2(self):
        return self._param2
    
    @property
    def param3(self):
        return self._param3
    
    @property
    def param4(self):
        return self._param4
    
    @property
    def f_1(self):
        return self._f_1
    
    @property
    def f_2(self):
        return self._f_2
    
    @property
    def f_3(self):
        return self._f_3
    
    @property
    def f_4(self):
        return self._f_4
    
    @property
    def airDiamond(self):
        return self._airDiamond

    @property
    def tangent(self):
        return self._tangent
    @property
    def binormal(self):
        return self._binormal

    @property
    def ghost_id(self):
        return self._ghost_id
    
    
    # @property
    # def K(self):
    #     return self._K
        
class Halo:
    """ """
    # More information at LocalDomainStruct.h
    __slots__ = [
        '_nb_halos',      # Number of halos

        '_neigh',        # neighboring partitions and halo exchange info
                         # shape: (2, nb_neighbors) shape=0 if MPI_SIZE=1
                         # layout:
                         # [
                         #   [neighbor_part_id_0, neighbor_part_id_1, ...],
                         #   [nb_halos_to_send_0, nb_halos_to_send_1, ...]
                         # ]
                         # defines communication pattern and grouping in `_halosint`

        '_halosint',     # interior halo cells to send to neighboring partitions
                         # shape: (total_halos_int,) shape=0 if MPI_SIZE=1
                         # layout:
                         # [
                         #   halos_for_P1..., halos_for_P2..., ...
                         # ]
                         # contains local cell indices grouped by neighbor partition
                         # grouping and counts are defined by `_neigh`

        '_halosext',     # exterior halo cells received from neighbors
                         # shape: (nb_halos_ext, max_cell_nodeid + 2) shape=0 if MPI_SIZE=1
                         # layout:
                         # [
                         #   [global_id, node_0, node_1, ..., node_(k-1), k],
                         #   ...
                         # ]
                         # where:
                         # - global_id = global cell index in the global mesh
                         # - last entry = number of nodes
                         # constructed by concatenating neighbors' `_halosint` as describe in `_neigh`

        '_centvol',      # geometric data of exterior halo cells
                         # shape: (nb_halos_ext, 4) shape=0 if MPI_SIZE=1
                         # layout:
                         # [
                         #   [cx, cy, cz, volume/area],
                         #   ...
                         # ]

        '_sizehaloghost' # total number of halo ghost cells
                         # equals len(Ghost.ext_info_flt)
    ]

    def __init__(self):
        pass

    @property
    def nb_halos(self):
        return self._nb_halos

    @property
    def halosint(self):
        return self._halosint
    
    @property
    def halosext(self):
        return self._halosext
    
    @property
    def neigh(self):
        return self._neigh
    
    @property
    def centvol(self):
        return self._centvol

    @property
    def sizehaloghost(self):
        return self._sizehaloghost



class Ghost:
    """ """
    __slots__ = [
        '_nb_ghosts',       # total number of local ghost cells (int)

        '_nb_haloghosts',   # total number of halo ghost cells received from other MPI ranks (int)

        '_info_int',        # integer metadata for local ghost cells
                            # shape: (nb_ghosts, 5)
                            # layout:
                            # [
                            #   [cell_id, local_face_index, face_oldname, cell_global_id, face_id],
                            #   ...
                            # ]
                            # where:
                            # - cell_id = owner cell (local)
                            # - local_face_index = index of the face inside the cell
                            # - face_oldname = boundary identifier before partitioning
                            # - cell_global_id = global index of the owner cell
                            # - face_id = associated face index

        '_info_flt',        # floating-point data for local ghost cells
                            # shape: (nb_ghosts, 10)
                            # layout:
                            # [
                            #   [gx, gy, gz, gamma, fcx, fcy, fcz, nx, ny, nz],
                            #   ...
                            # ]
                            # where:
                            # - (gx, gy, gz) = ghost cell center
                            # - (fcx, fcy, fcz) = face center
                            # - (nx, ny, nz) = face normal
                            # - gamma: geometric interpolation coefficient used in flux reconstruction
                            # shape: (nb_ghosts,) or stored per ghost/face
                            #
                            # definition:
                            # scalar measuring the relative position of the cell center with respect to the face geometry
                            # used to project/interpolate values along the face normal direction
                            #
                            # in 2D:
                            # - u = vector from a face node to the cell center
                            # - v = edge vector of the face
                            # gamma = (u · v) / |face|²
                            # interpretation:
                            # - normalized projection of the cell center onto the face edge direction
                            #
                            # in 3D:
                            # - u = vector from cell center to face center
                            # - n = unit normal vector of the face
                            # gamma = (u · n)
                            # interpretation:
                            # - signed distance from cell center to face along the normal direction

        '_ext_info_int',    # integer metadata for halo ghost cells (received from MPI neighbors)
                            # shape: (nb_haloghosts, 3) or 0 if MPI_SIZE == 1
                            # layout:
                            # [
                            #   [halo_cell, face_oldname, cell_global_id],
                            #   ...
                            # ]
                            # where:
                            # - halo_cell = index into Halo.halosext

        '_ext_info_flt',    # floating-point data for halo ghost cells (received from MPI neighbors)
                            # shape: (nb_haloghosts, 10) or 0 if MPI_SIZE == 1
                            # same layout as `_info_flt`

        '_faceid'           # mapping: ghost cell -> associated face
                            # shape: (nb_ghosts,)
                            # layout: [face_id, ...]
    ]

    def __init__(self):
        pass

    @property
    def nb_ghosts(self):
        return self._nb_ghosts

    @property
    def nb_haloghosts(self):
        return self._nb_haloghosts

    @property
    def info_int(self):
        return self._info_int

    @property
    def info_flt(self):
        return self._info_flt

    @property
    def ext_info_int(self):
        return self._ext_info_int

    @property
    def ext_info_flt(self):
        return self._ext_info_flt

    @property
    def faceid(self):
        return self._faceid
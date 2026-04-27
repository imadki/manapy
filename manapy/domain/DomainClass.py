import os
from manapy.domain.LocalDomainClass import LocalDomain
from manapy.domain.MeshClass import Mesh
from manapy.domain.PartitioningClass import Partitioning
from manapy.domain.geometry import Cell, Node, Halo, Face, Ghost
import shutil
from mpi4py import MPI
import numpy as np
import meshio
import manapy.backends.types as types
from manapy.domain.LocalDomainInterface import LocalDomainInterface

class Domain:
  PartitioningClass = Partitioning
  def __init__(self, local_domain: 'LocalDomain'):
    # Init
    self.rank = local_domain.rank
    self.size = local_domain.size
    self.dim = local_domain.dim
    self.nbnodes = local_domain.nb_nodes
    self.nbcells = local_domain.nb_cells
    self.nbfaces = local_domain.nb_faces
    self.nbhalos = local_domain.halo_halosext.shape[0]
    self.nbghost = local_domain.nb_phy_faces
    self._maxcellfid = local_domain.max_cell_faceid
    self._maxcellnodeid = local_domain.max_cell_nodeid
    self._maxfacenid = local_domain.max_face_nodeid
    self.test = local_domain.test # debug attribute


    self.comm = MPI.COMM_WORLD
    self.vtkprecision = "Float32" if types.FLOAT_TYPE == "float32" else "Float64"
    self._vtkpath = self.get_vtk_path(self.rank)


    self.cells = Cell()
    self.nodes = Node()
    self.faces = Face()
    self.halos = Halo()
    self.ghost = Ghost()

    # Cells
    self.cells._nbcells = local_domain.nb_cells
    self.cells._nodeid = local_domain.cells
    self.cells._faceid = local_domain.cell_faceid
    self.cells._cellfid = local_domain.cell_cellfid
    self.cells._cellnid = local_domain.cell_cellnid
    self.cells._halonid = local_domain.cell_halonid
    self.cells._ghostnid = local_domain.cell_ghostid
    self.cells._haloghostnid = local_domain.cell_haloghostid
    self.cells._center = local_domain.cell_center # dimension always 3D
    self.cells._volume = local_domain.cell_volume
    self.cells._nf = local_domain.cell_nf # dimension always 3D
    self.cells._loctoglob = local_domain.cell_loctoglob
    self.cells._tc = local_domain.cell_tc
    self.cells._periodicfid = local_domain.cell_periodicfid
    self.cells._periodicnid = local_domain.cell_periodicnid
    self.cells._shift = local_domain.cell_shift

    # Nodes
    self.nodes._nbnodes = local_domain.nb_nodes
    self.nodes._vertex = local_domain.nodes # in old domain this has dimension (nb_nodes, 4) -> new domain (nb_nodes, 3) 4 is the node_oldname
    self.nodes._name = local_domain.node_name
    self.nodes._oldname = local_domain.node_oldname
    self.nodes._cellid = local_domain.node_cellid
    self.nodes._ghostid = local_domain.node_ghostid
    self.nodes._haloghostid = local_domain.node_haloghostid
    self.nodes._loctoglob = local_domain.node_loctoglob
    self.nodes._halonid = local_domain.node_haloid
    self.nodes._nparts = None
    self.nodes._periodicid = local_domain.node_periodicid
    self.nodes._R_x = local_domain.node_R_x
    self.nodes._R_y = local_domain.node_R_y
    self.nodes._R_z = local_domain.node_R_z
    self.nodes._number = local_domain.node_number
    self.nodes._lambda_x = local_domain.node_lambda_x
    self.nodes._lambda_y = local_domain.node_lambda_y
    self.nodes._lambda_z = local_domain.node_lambda_z

    # Faces
    self.faces._nbfaces = local_domain.nb_faces
    self.faces._nodeid = local_domain.faces
    self.faces._cellid = local_domain.face_cellid
    self.faces._name = local_domain.face_name
    self.faces._oldname = local_domain.face_oldname
    self.faces._normal = local_domain.face_normal # always 3D
    self.faces._mesure = local_domain.face_measure
    self.faces._center = local_domain.face_center # always 3D
    self.faces._dist_ortho = local_domain.face_dist_ortho # ?? in old domain in 3D it is useless to define dist_ortho with shape nbfaces TODO
    self.faces._oppnodeid = None
    self.faces._halofid = local_domain.face_haloid
    self.faces._param1 = local_domain.face_param1
    self.faces._param2 = local_domain.face_param2
    self.faces._param3 = local_domain.face_param3
    self.faces._param4 = local_domain.face_param4
    self.faces._f_1 = local_domain.face_f1
    self.faces._f_2 = local_domain.face_f2
    self.faces._f_3 = local_domain.face_f3
    self.faces._f_4 = local_domain.face_f4
    self.faces._airDiamond = local_domain.face_air_diamond
    self.faces._tangent = local_domain.face_tangent
    self.faces._binormal = local_domain.face_binormal
    self.faces._ghost_id = local_domain.face_to_phyid

    # Halos
    self.halos._halosext = local_domain.halo_halosext
    self.halos._neigh = local_domain.halo_neighsub
    self.halos._halosint = local_domain.halo_halosint
    self.halos._centvol = local_domain.halo_centvol
    self.halos._sizehaloghost = len(local_domain.ext_ghost_info_flt)
    self.halos._faces = None
    self.halos._nodes = None
    self.halos._requests = None
    self.halo_comm =  local_domain.halo_comm
    self.phy_faces_comm = local_domain.phy_faces_comm

    #Ghost
    self.ghost._info_int = local_domain.ghost_info_int
    self.ghost._info_flt = local_domain.ghost_info_flt
    self.ghost._ext_info_int = local_domain.ext_ghost_info_int
    self.ghost._ext_info_flt = local_domain.ext_ghost_info_flt
    self.ghost._faceid = local_domain.phyid_to_faceid

    # Domain
    self._bounds = local_domain.bounds
    self._BCs = local_domain.BCs
    self._innerfaces = local_domain.innerfaces
    self._infaces = local_domain.infaces
    self._outfaces = local_domain.outfaces
    self._upperfaces = local_domain.upperfaces
    self._bottomfaces = local_domain.bottomfaces
    self._halofaces = local_domain.halofaces
    self._periodicinfaces = local_domain.periodicinfaces
    self._periodicoutfaces = local_domain.periodicoutfaces
    self._periodicupperfaces = local_domain.periodicupperfaces
    self._periodicbottomfaces = local_domain.periodicbottomfaces
    self._boundaryfaces = local_domain.boundaryfaces
    self._periodicboundaryfaces = local_domain.periodicboundaryfaces
    self._innernodes = local_domain.innernodes
    self._innodes = local_domain.innodes
    self._outnodes = local_domain.outnodes
    self._uppernodes = local_domain.uppernodes
    self._bottomnodes = local_domain.bottomnodes
    self._halonodes = local_domain.halonodes
    self._periodicinnodes = local_domain.periodicinnodes
    self._periodicoutnodes = local_domain.periodicoutnodes
    self._periodicuppernodes = local_domain.periodicuppernodes
    self._periodicbottomnodes = local_domain.periodicbottomnodes
    self._boundarynodes = local_domain.boundarynodes
    self._periodicboundarynodes = local_domain.periodicboundarynodes
    self._frontfaces = local_domain.frontfaces
    self._backfaces = local_domain.backfaces
    self._periodicfrontfaces = local_domain.periodicfrontfaces
    self._periodicbackfaces = local_domain.periodicbackfaces
    self._frontnodes = local_domain.frontnodes
    self._backnodes = local_domain.backnodes
    self._periodicfrontnodes = local_domain.periodicfrontnodes
    self._periodicbacknodes = local_domain.periodicbacknodes

    self.BCs = self._BCs
    self.innerfaces = self._innerfaces
    self.infaces = self._infaces
    self.outfaces = self._outfaces
    self.bottomfaces = self._bottomfaces
    self.upperfaces = self._upperfaces
    self.halofaces = self._halofaces
    self.innernodes = self._innernodes
    self.innodes = self._innodes
    self.outnodes = self._outnodes
    self.bottomnodes = self._bottomnodes
    self.uppernodes = self._uppernodes
    self.halonodes = self._halonodes
    self.boundaryfaces = self._boundaryfaces
    self.boundarynodes = self._boundarynodes
    self.periodicboundaryfaces = self._periodicboundaryfaces
    self.periodicboundarynodes = self._periodicboundarynodes

    self.periodicinfaces = self._periodicinfaces
    self.periodicoutfaces = self._periodicoutfaces
    self.periodicupperfaces = self._periodicupperfaces
    self.periodicfrontfaces = self._periodicfrontfaces
    self.frontfaces = self._frontfaces
    self.backfaces = self._backfaces
    self.frontnodes = self._frontnodes
    self.backnodes = self._backnodes
    self._typeOfCells = self._define_eltypes()
    self.bounds = self._bounds


  @staticmethod
  def _all_local_mesh_files_exist(size: int):
    folder_name = f"local_domain_{size}"
    for rank in range(size):
      file_path = os.path.join(folder_name, f"mesh{rank}.hdf5")
      if not os.path.isfile(file_path):
        return False
    return True

  @staticmethod
  def _delete_local_domain_folder(size: int):
    folder_name = f"local_domain_{size}"
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
      shutil.rmtree(folder_name)

  @staticmethod
  def create_domain(mesh_path, dim, partitioning_method, recreate=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
      if recreate == True or not Domain._all_local_mesh_files_exist(size):
        mesh = Mesh(mesh_path, dim)
        partitioner = Partitioning(mesh)
        local_domain_data = partitioner.create_sub_domains()
        LocalDomainInterface.save_local_domains(local_domain_data, size)
        local_domain = LocalDomain(local_domain_data[0], rank, size)
        return Domain(local_domain)
      else:
        try:
          local_domain_struct = LocalDomainInterface.load_and_create(rank, size)
          local_domain = LocalDomain(local_domain_struct, rank, size)
          return Domain(local_domain)
        except Exception as e:
          import traceback
          print(f"[Rank {rank}] failed: {e} {traceback.format_exc()}")
    else:
      if rank == 0:
        print("====> Start <=====")
        try:
          if recreate == True or not Domain._all_local_mesh_files_exist(size):
            print("====> Creating Mesh <=====")
            Domain._delete_local_domain_folder(size)
            mesh = Mesh(mesh_path, dim)
            partitioner = Partitioning(mesh)
            partitioner.set_part_vert(size, partitioning_method)
            local_domains = partitioner.create_sub_domains()
            LocalDomainInterface.save_local_domains(local_domains, size)
            print("====> End <=====")
        except Exception as e:
          import traceback
          print(f"[Rank 0] failed: {e} {traceback.format_exc()}")
          comm.Abort(1)

      comm.Barrier()


      try:
        local_domain_struct = LocalDomainInterface.load_and_create(rank, size)
        local_domain = LocalDomain(local_domain_struct, rank, size)
      except Exception as e:
        import traceback
        print(f"[Rank {rank}] failed: {e} {traceback.format_exc()}")
        comm.Abort(1)

      comm.Barrier()
      return Domain(local_domain)

  ##########################################################################
  ##########################################################################
  ##########################################################################
  ##########################################################################
  ##########################################################################

  @staticmethod
  def get_vtk_path(rank):
    vtkpath = "vtk_results"
    if rank == 0:
      if os.path.exists(vtkpath):
        shutil.rmtree(vtkpath)
      os.mkdir(vtkpath)
    return vtkpath


  def _define_eltypes(self):
    """
    Define the type of cells
    """
    typeOfCells = {}
    # self._maxcellnodeid = max(self.cells.nodeid[:, -1])

    if self.dim == 2:
      nbOfTriangles = len(self.cells.nodeid[self.cells.nodeid[:, -1] == 3])
      nbOfQuad = len(self.cells.nodeid[self.cells.nodeid[:, -1] == 4])

      if nbOfQuad != 0:
        typeOfCells["quad"] = self.cells.nodeid[self.cells.nodeid[:, -1] == 4][:, :4]
      if nbOfTriangles != 0:
        typeOfCells["triangle"] = self.cells.nodeid[self.cells.nodeid[:, -1] == 3][:, :3]

      # self._maxfacenid = 2
      # if self._maxcellnodeid == 3:
      #   self._maxcellfid = 3
      # elif self._maxcellnodeid == 4:
      #   self._maxcellfid = 4

    elif self.dim == 3:

      nbOfTetra = len(self.cells.nodeid[self.cells.nodeid[:, -1] == 4])
      nbOfpyra = len(self.cells.nodeid[self.cells.nodeid[:, -1] == 5])
      nbOfQuad = len(self.cells.nodeid[self.cells.nodeid[:, -1] == 8])

      if nbOfTetra != 0:
        typeOfCells["tetra"] = self.cells.nodeid[self.cells.nodeid[:, -1] == 4][:, :4]
      if nbOfQuad != 0:
        typeOfCells["hexahedron"] = self.cells.nodeid[self.cells.nodeid[:, -1] == 8][:, :8]
      if nbOfpyra != 0:
        typeOfCells["pyramid"] = self.cells.nodeid[self.cells.nodeid[:, -1] == 5][:, :5]

      # if self._maxcellnodeid == 4:
      #   self._maxcellfid = 4
      #   self._maxfacenid = 3
      # elif self._maxcellnodeid == 5:
      #   self._maxcellfid = 4
      #   self._maxfacenid = 5
      # elif self._maxcellnodeid == 8:
      #   self._maxcellfid = 6
      #   self._maxfacenid = 4

    # self._maxcellfid = types.np_int_type(self._maxcellfid)
    # self._maxfacenid = types.np_int_type(self._maxfacenid)

    return typeOfCells


  def save_on_cell(self, dt=0, time=0, niter=0, miter=0, value=None):

    if value is None:
      raise ValueError("value must be given")
    assert len(value) == self.nbcells, 'value size != number of cells'

    elements = self._typeOfCells  # {"quad": self.cells._nodeid}

    points = self.nodes.vertex[:, :3]
    points = np.array(points, dtype=types.np_float_type)


    data = {"w": list(value)}
    # data = {"w": data}

    maxw = max(value)

    integral_maxw = np.zeros(1, dtype=types.np_float_type)

    self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)

    if self.comm.rank == 0:
      print(" **************************** Computing ****************************")
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      print("Iteration = ", niter, "time = ", time, "time step = ", dt)
      print("max w =", integral_maxw[0])

    meshio.write_points_cells(f"{self._vtkpath}/visu" + str(self.comm.rank) + "-" + str(miter) + ".vtu",
                              points, elements, cell_data=data, file_format="vtu")

    if self.comm.rank == 0:
      with open(self._vtkpath + "/visu" + str(miter) + ".pvtu", "w") as text_file:
        text_file.write("<?xml version=\"1.0\"?>\n")
        text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
        text_file.write("<PPoints>\n")
        text_file.write(
          "<PDataArray type=\"" + self.vtkprecision + "\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
        text_file.write("</PPoints>\n")
        text_file.write("<PCells>\n")
        text_file.write("<PDataArray type=\"uint32\" Name=\"connectivity\" format=\"binary\"/>\n")
        text_file.write("<PDataArray type=\"uint32\" Name=\"offsets\" format=\"binary\"/>\n")
        text_file.write("<PDataArray type=\"uint32\" Name=\"types\" format=\"binary\"/>\n")
        text_file.write("</PCells>\n")
        text_file.write("<PCellData Scalars=\"h\">\n")
        text_file.write("<PDataArray type=\"" + self.vtkprecision + "\" Name=\"w\" format=\"binary\"/>\n")
        text_file.write("</PCellData>\n")

        for i in range(self.comm.size):
          name1 = "visu"
          bu1 = [10]
          bu1 = str(i)
          name1 += bu1
          name1 += "-" + str(miter)
          name1 += ".vtu"
          text_file.write("<Piece Source=\"" + str(name1) + "\"/>\n")
        text_file.write("</PUnstructuredGrid>\n")
        text_file.write("</VTKFile>")

  def save_on_node(self, dt=0, time=0, niter=0, miter=0, value=None):

    if value is None:
      raise ValueError("value must be given")
    assert len(value) == self.nbnodes, 'value size != number of nodes'

    elements = self._typeOfCells  # {"quad": self.cells._nodeid}
    points = self.nodes.vertex[:, :3]

    data = {"w": value}

    maxw = max(value)

    integral_maxw = np.zeros(1, dtype=types.np_float_type)

    self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)

    if self.comm.rank == 0:
      print(" **************************** Computing ****************************")
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      print("Iteration = ", niter, "time = ", time, "time step = ", dt)
      print("max w =", integral_maxw[0])

    meshio.write_points_cells(f"{self._vtkpath}/visu" + str(self.comm.rank) + "-" + str(miter) + ".vtu",
                              points, elements, point_data=data, file_format="vtu")

    if self.comm.rank == 0:
      with open(self._vtkpath + "/visu" + str(miter) + ".pvtu", "w") as text_file:
        text_file.write("<?xml version=\"1.0\"?>\n")
        text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
        text_file.write("<PPoints>\n")
        text_file.write(
          "<PDataArray type=\"" + self.vtkprecision + "\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
        text_file.write("</PPoints>\n")
        text_file.write("<PCells>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"connectivity\" format=\"binary\"/>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"offsets\" format=\"binary\"/>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"types\" format=\"binary\"/>\n")
        text_file.write("</PCells>\n")
        text_file.write("<PPointData Scalars=\"h\">\n")
        text_file.write("<PDataArray type=\"" + self.vtkprecision + "\" Name=\"w\" format=\"binary\"/>\n")
        text_file.write("</PPointData>\n")

        for i in range(self.comm.size):
          name1 = "visu"
          bu1 = [10]
          bu1 = str(i)
          name1 += bu1
          name1 += "-" + str(miter)
          name1 += ".vtu"
          text_file.write("<Piece Source=\"" + str(name1) + "\"/>\n")
        text_file.write("</PUnstructuredGrid>\n")
        text_file.write("</VTKFile>")

  def save_on_node_multi(self, dt=0, time=0, niter=0, miter=0, variables=None, values=None, file_format="vtu"):

    if values is None:
      raise ValueError("value must be given")
    assert len(values[0]) == self.nbnodes, 'value size != number of nodes'

    elements = self._typeOfCells  # {"quad": self.cells._nodeid}
    points = self.nodes.vertex[:, :3]

    nvalues = len(values)
    data = {}
    for k in range(0, nvalues):
      data[variables[k]] = values[k]

    maxw = max(values[0])

    integral_maxw = np.zeros(1, dtype=types.np_float_type)

    self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)

    if self.comm.rank == 0:
      print(" **************************** Computing ****************************")
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      print("Iteration = ", niter, "time = ", time, "time step = ", dt)
      print("max" + variables[0] + " =", integral_maxw[0])

    meshio.write_points_cells(f"{self._vtkpath}/visu" + str(self.comm.rank) + "-" + str(miter) + "." + file_format,
                              points, elements, point_data=data, file_format=file_format)

    if self.comm.rank == 0:
      with open(self._vtkpath + "/visu" + str(miter) + ".pvtu", "w") as text_file:
        text_file.write("<?xml version=\"1.0\"?>\n")
        text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
        text_file.write("<PPoints>\n")
        text_file.write(
          "<PDataArray type=\"" + self.vtkprecision + "\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
        text_file.write("</PPoints>\n")
        text_file.write("<PCells>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"connectivity\" format=\"binary\"/>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"offsets\" format=\"binary\"/>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"types\" format=\"binary\"/>\n")
        text_file.write("</PCells>\n")
        text_file.write("<PPointData Scalars=\"h\">\n")
        for k in range(0, nvalues):
          text_file.write(
            "<PDataArray type=\"" + self.vtkprecision + "\" Name=\"" + variables[k] + "\" format=\"binary\"/>\n")
        text_file.write("</PPointData>\n")

        for i in range(self.comm.size):
          name1 = "visu"
          bu1 = [10]
          bu1 = str(i)
          name1 += bu1
          name1 += "-" + str(miter)
          name1 += ".vtu"
          text_file.write("<Piece Source=\"" + str(name1) + "\"/>\n")
        text_file.write("</PUnstructuredGrid>\n")
        text_file.write("</VTKFile>")

  def save_on_cell_multi(self, dt=0, time=0, niter=0, miter=0, variables=None, values=None, file_format="vtu"):

    if values is None:
      raise ValueError("value must be given")
    assert len(values[0]) == self.nbcells, 'value size != number of cells'

    elements = self._typeOfCells  # {"triangle": self.cells._nodeid}
    points = self.nodes.vertex[:, :3]

    nvalues = len(values)

    # data
    data = {variables[k]: [values[k]] for k in range(nvalues)}

    maxw = max(values[0])

    integral_maxw = np.zeros(1, dtype=types.np_float_type)

    self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)

    if self.comm.rank == 0:
      print(" **************************** Computing ****************************")
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      print("Iteration = ", niter, "time = ", time, "time step = ", dt)
      print("max" + variables[0] + " =", integral_maxw[0])

    meshio.write_points_cells(f"{self._vtkpath}/visu" + str(self.comm.rank) + "-" + str(miter) + "." + file_format,
                              points, elements, cell_data=data, file_format=file_format)

    if self.comm.rank == 0:
      with open(self._vtkpath + "/visu" + str(miter) + ".pvtu", "w") as text_file:
        text_file.write("<?xml version=\"1.0\"?>\n")
        text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
        text_file.write("<PPoints>\n")
        text_file.write(
          "<PDataArray type=\"" + self.vtkprecision + "\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
        text_file.write("</PPoints>\n")
        text_file.write("<PCells>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"connectivity\" format=\"binary\"/>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"offsets\" format=\"binary\"/>\n")
        text_file.write("<PDataArray type=\"int32\" Name=\"types\" format=\"binary\"/>\n")
        text_file.write("</PCells>\n")
        text_file.write("<PCellData Scalars=\"h\">\n")
        for k in range(0, nvalues):
          text_file.write(
            "<PDataArray type=\"" + self.vtkprecision + "\" Name=\"" + variables[k] + "\" format=\"binary\"/>\n")
        text_file.write("</PCellData>\n")

        for i in range(self.comm.size):
          name1 = "visu"
          bu1 = [10]
          bu1 = str(i)
          name1 += bu1
          name1 += "-" + str(miter)
          name1 += ".vtu"
          text_file.write("<Piece Source=\"" + str(name1) + "\"/>\n")
        text_file.write("</PUnstructuredGrid>\n")
        text_file.write("</VTKFile>")

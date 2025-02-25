#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:05:57 2022

@author: kissami
"""
__all__ = ['Domain']

import os
import meshio
from mpi4py import MPI
import numpy as np

import manapy.comms as manapy_comms
from manapy.comms          import update_haloghost_info_2d, update_haloghost_info_3d, prepare_comm

from manapy.ddm.ddm_utils2d  import (create_NeighborCellByFace, create_node_cellid, create_cellsOfFace, create_NormalFacesOfCell,
                                 create_node_ghostid, create_cell_faceid, dist_ortho_function,
                                 Compute_2dcentervolumeOfCell, create_2dfaces,  create_info_2dfaces, create_2d_halo_structure,
                                 face_info_2d, update_pediodic_info_2d, variables_2d, face_gradient_info_2d)


from manapy.ddm.ddm_utils3d   import (Compute_3dcentervolumeOfCell, create_3dfaces, create_info_3dfaces, create_3d_halo_structure,
                                      face_info_3d, oriente_3dfacenodeid, update_pediodic_info_3d, variables_3d, face_gradient_info_3d )

from manapy.ddm.pure_ddm   import read_mesh_file

from manapy.backends       import CPUBackend
from manapy.base.base import make_get_conf
from manapy.base.base import Struct
import warnings



from manapy.ddm.geometry   import Face, Cell, Node, Halo
class Domain():
    
    @classmethod
    def process_conf(cls, conf, kwargs):
        """                                                                                                                                                                                                
        Process configuration parameters.                                                                                                                                                                  
        """
        get = make_get_conf(conf, kwargs)
        
        if len(cls._parameters) and cls._parameters[0][0] != 'name':
            options = Domain._parameters + cls._parameters

        else:
            options = Domain._parameters
            
        
        opts = Struct()
        allow_extra = False
        for name, _, default, required, _ in options:
            if name == '*':
                allow_extra = True
                continue

            msg = ('missing "%s" in options!' % name) if required else None
            setattr(opts, name, get(name, default, msg))

        if allow_extra:
            all_keys = set(conf.to_dict().keys())
            other = all_keys.difference(list(opts.to_dict().keys()))
            for name in other:
                setattr(opts, name, get(name, None, None))
                
        return opts
    """ """
    _parameters = [
            ('backend', 'str', "numba", False,
             'Running pure python functions without numba.'),
            ('multithreading', 'str', 'single', None,
             'multithreading when running numba.'),
            ('signature', 'bool', True, False,
             'compile functions using types'),
            ('cache', 'bool', False, False,
             'save the compiled functions'),
            ('float_precision', 'str', "double", None,
             'precision of the float arrays'),
            ('int_precision', 'str', "signed", None,
             'precision of the integer arrays')
    ]

    def __init__(self, dim=None, comm=None, conf=None, **kwargs):
        
        if conf is None:
            conf = Struct()
        self.conf = self.process_conf(conf, kwargs)
        get = make_get_conf(self.conf, kwargs)
        
        if get("python"):
            warnings.warn("You are running pure python functions", Warning)
        
        self.backend = CPUBackend(multithread=get("multithreading"), backend=get("backend"), 
                                  cache=get("cache"), float_precision=get("float_precision"),
                                  int_precision=get("int_precision"))
        
        self.forcedbackend = "python"
        if self.backend.backend != "python":
            self.forcedbackend = "numba"
            
        self.signature = get('signature')
        
        if get('float_precision') == "single":
            self.float_precision = 'f4'
            self.vtkprecision = "Float32"
            self.mpi_precision = MPI.FLOAT
        else :
            self.float_precision = 'f8'
            self.vtkprecision = "Float64"
            self.mpi_precision = MPI.DOUBLE_PRECISION
        
        if get('int_precision') == "signed":
            self.int_precision = 'i4'
        else :
            self.int_precision = 'u4'
        
        if comm is None:
            comm = MPI.COMM_WORLD
            
        if dim is None:
            raise ValueError("dim must be given")
        
        self._comm = comm
        self._dim  = np.int32(dim)


        self.create_domain(self._comm.Get_size(), self._comm.Get_rank())
        

    def create_domain(self, size, rank):
        manapy_comms.SIZE = size
        manapy_comms.RANK = rank
        self._size = size
        self._rank = rank

        if self._rank == 0:
            print("Local domain contruction ...")

        self._nodes = Node()
        self._cells = Cell()
        self._faces = Face()
        self._halos = Halo()

        self._setup_vtkpath()
        self._remove_existing_vtk_files()

        self._read_partition()
        self._define_bounds()
        self._compute_cells_info()
        self._make_neighbors()
        self._define_eltypes()
        self._create_faces_cons()
        self._define_face_info()
        self._make_halo_info()
        self._update_boundaries()
        self._update_ghost_info()
        self._prepare_communication()
        self._update_periodic_boudaries()
        self._update_haloghost()
        self._compute_diamondCell_info()
        self._compute_orthocenter()  # Only in 2D
        
    def _setup_vtkpath(self):
        """
        Set up the path to save the VTK files.
        """
        self._mesh_dir = "meshes"+str(self._size)+"PROC"
        self._vtkpath  = "results"
        if self._rank == 0 :
            if not os.path.exists(self._vtkpath):
                os.mkdir(self._vtkpath)
        
        
    def _remove_existing_vtk_files(self):
        """
        Remove VTK files.
        """
        if self._rank == 0 :
            #removing existing vtk files
            if not os.path.exists(self._vtkpath):
                os.mkdir(self._vtkpath)
            for root, dirs, files in os.walk(self._vtkpath):
                for file in files:
                    os.remove(os.path.join(root, file))
        self._comm.Barrier()
        
    def _read_partition(self):
        """
        read the hdf5 partition file
        """

        #read nodes.vertex, cells.tc, cells.nodeid
        self._cells._tc, \
        self._cells._nodeid, \
        self._nodes._vertex, \
        self._halos._halosint, \
        self._halos._halosext, \
        self._halos._centvol, \
        self._cells._globtoloc, \
        self._cells._loctoglob, \
        self._nodes._loctoglob, \
        self._halos._neigh, \
        self._nodes._nparts = read_mesh_file(self._size, self._rank, self._mesh_dir, self.float_precision)
            
        self._nbcells = np.int32(len(self._cells._nodeid))
        self._nbnodes = np.int32(len(self._nodes._vertex))
        
        
    def _define_bounds(self):
        """
        define the boudaries of the geometry
        """
        if self._dim == 2:
            self._bounds  = np.array([[min(self._nodes.vertex[:,0]), max(self._nodes.vertex[:,0])], 
                              [min(self._nodes.vertex[:,1]), max(self._nodes.vertex[:,1])]], dtype=self.float_precision)
            
        if self._dim == 3:
            self._bounds  = np.array([[min(self._nodes.vertex[:,0]), max(self._nodes.vertex[:,0])], 
                              [min(self._nodes.vertex[:,1]), max(self._nodes.vertex[:,1])],
                              [min(self._nodes.vertex[:,2]), max(self._nodes.vertex[:,2])]], dtype=self.float_precision)
        
    def _compute_cells_info(self):
        """
        Compute the center and volume of cells
        """
        #Create center and volume for each cell
        
        self._cells._center = np.zeros((self._nbcells, 3), dtype=self.float_precision)
        self._cells._volume = np.zeros(self._nbcells, dtype=self.float_precision)
        if self._dim == 2:
            self.backend.compile(Compute_2dcentervolumeOfCell, signature=self.signature)(self._cells._nodeid, self._nodes._vertex, \
                                self._nbcells, self._cells._center, self._cells._volume)
        elif self._dim == 3:
            self.backend.compile(Compute_3dcentervolumeOfCell, signature=self.signature)(self._cells._nodeid, self._nodes._vertex, \
                                self._nbcells, self._cells._center, self._cells._volume)
        
        
    def _make_neighbors(self):
        """
        create the cells around each vertex & cells neighbor by vertex
        """
        
        self._nodes._cellid, \
        self._cells._cellnid =  self.backend.compile(create_node_cellid, signature=self.signature, \
                                                     forcedbackend=self.forcedbackend)(self._cells._nodeid, \
                                                     self._nodes._vertex, self._nbcells, self._nbnodes)        
        
    def _define_eltypes(self):
        """
        Define the type of cells
        """
        self._typeOfCells = {}
        self._maxcellnodeid = max(self._cells._nodeid[:,-1])
        
        if self._dim == 2:
            self._nbOfTriangles = len(self._cells._nodeid[self._cells._nodeid[:,-1]==3])
            self._nbOfQuad      = len(self._cells._nodeid[self._cells._nodeid[:,-1]==4])
            
            if self._nbOfQuad != 0:
                self._typeOfCells["quad"]     = self._cells._nodeid[self._cells._nodeid[:,-1]==4][:,:4]
            if self._nbOfTriangles != 0:
                self._typeOfCells["triangle"] = self._cells._nodeid[self._cells._nodeid[:,-1]==3][:,:3]
            
            self._maxfacenid = 2
            if self._maxcellnodeid == 3:
                self._maxcellfid = 3
            elif self._maxcellnodeid == 4:
                self._maxcellfid = 4
        
        elif self._dim == 3:
            
            self._nbOfTetra = len(self._cells._nodeid[self._cells._nodeid[:,-1]==4])
            self._nbOfpyra  = len(self._cells._nodeid[self._cells._nodeid[:,-1]==5])
            self._nbOfQuad  = len(self._cells._nodeid[self._cells._nodeid[:,-1]==8])
            
            if self._nbOfTetra != 0:
                self._typeOfCells["tetra"]      = self._cells._nodeid[self._cells._nodeid[:,-1]==4][:,:4]
            if self._nbOfQuad != 0:
                self._typeOfCells["hexahedron"] = self._cells._nodeid[self._cells._nodeid[:,-1]==8][:,:8]
            if self._nbOfpyra != 0:
                self._typeOfCells["pyramid"]    = self._cells._nodeid[self._cells._nodeid[:,-1]==5][:,:5]
            
            if self._maxcellnodeid == 4:
                self._maxcellfid = 4
                self._maxfacenid = 3
            elif self._maxcellnodeid == 5:
                self._maxcellfid = 4
                self._maxfacenid = 5
            elif self._maxcellnodeid == 8:
                self._maxcellfid = 6
                self._maxfacenid = 4
            
        self._maxcellfid = np.int32(self._maxcellfid)
        self._maxfacenid = np.int32(self._maxfacenid)
                
    def _create_faces_cons(self):
        """
        create all connectivity for faces
        """
        
        #creating faces
        if self._dim == 2:
            p_faces = -1*np.ones(((self._nbOfQuad*4 + self._nbOfTriangles*3), self._maxfacenid+1), dtype=np.int32)
        elif self._dim == 3:
            p_faces = -1*np.ones(((self._nbOfQuad*6 + self._nbOfpyra*5 + self._nbOfTetra*4 ), self._maxfacenid+1), dtype=np.int32)

        cellf = -1*np.ones((self._nbcells, self._maxcellfid), dtype=np.int32)
        
        if self._dim == 2:
            self.backend.compile(create_2dfaces, signature=self.signature)(self._cells._nodeid, self._nbcells, p_faces, cellf)
        elif self._dim == 3:
            self.backend.compile(create_3dfaces, signature=self.signature)(self._cells._nodeid, self._nbcells, p_faces, cellf)
            
        
        # create the nodes of each face
        p_faces[:, :-1] = np.sort(p_faces[:, :-1], axis=1, kind='quicksort')[:, ::-1]
        self._faces._nodeid, oldTonewIndex = np.unique(p_faces, axis=0, return_inverse=True)
        
        # set the numer of faces
        self._nbfaces = np.int32(len(self._faces._nodeid))
    
        # create the faces of each cell
        self._cells._faceid = -1*np.ones((self._nbcells, self._maxcellfid+1), dtype=np.int32)
        
        self.backend.compile(create_cell_faceid, signature=self.signature, \
                             forcedbackend=self.forcedbackend)(self._nbcells, \
                                                             oldTonewIndex, cellf, self._cells._faceid, self._maxcellfid)
        
        #creater cells left and right of each face
        self._faces._cellid = -1*np.ones((self._nbfaces, 2), dtype=np.int32)
        
        self.backend.compile(create_cellsOfFace, signature=self.signature)(self._cells._faceid, self._nbcells, self._nbfaces, self._faces._cellid, \
                            self._maxcellfid)
        ############################################################################
        self._cells._cellfid = self.backend.compile(create_NeighborCellByFace, signature=self.signature, forcedbackend=self.forcedbackend)(self._cells._faceid, self._faces._cellid, \
                                                   self._nbcells, self._maxcellfid)
        
            
    def _define_face_info(self):
        """
        define the bc/normal/area of each face
        """
        self._nodes._name = np.zeros(self._nbnodes, dtype=self.int_precision)
        for i in range(self._nbnodes):
            self._nodes._name[i] = int(self._nodes._vertex[i][3])
        
        #####################################################
        #create info of faces
        self._faces._name   = np.zeros(self._nbfaces, dtype=self.int_precision)
        self._faces._normal = np.zeros((self._nbfaces, 3), dtype=self.float_precision)
        self._faces._mesure = np.zeros(self._nbfaces, dtype=self.float_precision)
        self._faces._center = np.zeros((self._nbfaces, 3), dtype=self.float_precision)

        
        if self._dim == 2:
            self.backend.compile(create_info_2dfaces, signature=self.signature)(self._faces._cellid, self._faces._nodeid, self._nodes._name, \
                                self._nodes._vertex, self._cells._center, self._nbfaces, self._faces._normal, self._faces._mesure, \
                                self._faces._center, self._faces._name)
        elif self._dim == 3:
            self._faces._tangent = np.zeros((self._nbfaces, 3), dtype=self.float_precision)
            self._faces._binormal = np.zeros((self._nbfaces, 3), dtype=self.float_precision)        
            self.backend.compile(create_info_3dfaces, signature=self.signature)(self._faces._cellid, self._faces._nodeid, self._nodes._name, \
                                self._nodes._vertex, self._cells._center, self._nbfaces, self._faces._normal, self._faces._tangent, self._faces._binormal, self._faces._mesure, \
                                self._faces._center, self._faces._name)
        ############################################################################
        
        #Create outgoing normal vectors
        self._cells._nf = np.zeros((self._nbcells, self._maxcellfid , 3), dtype=self.float_precision)
        if self._dim == 2:
            self.backend.compile(create_NormalFacesOfCell, signature=self.signature)(self._cells._center, self._faces._center, self._cells._faceid, \
                                self._faces._normal, self._nbcells, self._cells._nf, self._maxcellfid)
        
        ###########################################################################
        
        if self._dim == 3:
            self.backend.compile(oriente_3dfacenodeid, signature=self.signature)(self._faces._nodeid, self._faces._normal, self._nodes._vertex)
        
        ###########################################################################
    def _make_halo_info(self):
        
        
        if self._dim == 2:
            self._nodes._halonid, self._faces._halofid, \
            self._nodes._name, self._faces._name, \
            self._faces._oldname = self.backend.compile(create_2d_halo_structure, signature=self.signature, forcedbackend=self.forcedbackend)(self._halos._halosext, \
                                                       self._faces._nodeid, self._faces._cellid, self._faces._name, \
                                                       self._nodes._name, self._nodes._loctoglob, self._size, self._nbcells, self._nbfaces, \
                                                       self._nbnodes)
        if self._dim == 3:
            self._nodes._halonid, self._faces._halofid, \
            self._nodes._name, self._faces._name, \
            self._faces._oldname = self.backend.compile(create_3d_halo_structure, signature=self.signature, forcedbackend=self.forcedbackend)(self._halos._halosext, \
                                                       self._faces._nodeid, self._faces._cellid, self._faces._name, \
                                                       self._nodes._name, self._nodes._loctoglob, self._size, self._nbcells, self._nbfaces, \
                                                       self._nbnodes)
        
    def _update_boundaries(self):
        
        ##################################Update faces and names type##########################################
        self._nodes._oldname = np.zeros(self._nbnodes, dtype=self.int_precision)
        self._nodes._oldname[:] = self._nodes._vertex[:,3]
        
        self._innerfaces  = np.where(self._faces._name==0)[0].astype(self.int_precision)
        self._infaces     = np.where(self._faces._name==1)[0].astype(self.int_precision)
        self._outfaces    = np.where(self._faces._name==2)[0].astype(self.int_precision)
        self._upperfaces  = np.where(self._faces._name==3)[0].astype(self.int_precision)
        self._bottomfaces = np.where(self._faces._name==4)[0].astype(self.int_precision)
        self._halofaces   = np.where(self._faces._name==10)[0].astype(self.int_precision)
        if self._size == 1:
            self._halofaces   = np.asarray([], dtype=self.int_precision)
        
        self._periodicinfaces     = np.where(self._faces._name==11)[0].astype(self.int_precision)
        self._periodicoutfaces    = np.where(self._faces._name==22)[0].astype(self.int_precision)
        self._periodicupperfaces  = np.where(self._faces._name==33)[0].astype(self.int_precision)
        self._periodicbottomfaces = np.where(self._faces._name==44)[0].astype(self.int_precision)
            
      
        self._boundaryfaces = np.concatenate([self._infaces, self._outfaces, self._bottomfaces, self._upperfaces] )
        self._periodicboundaryfaces = np.concatenate([self._periodicinfaces, self._periodicoutfaces, self._periodicbottomfaces, 
                                                      self._periodicupperfaces] )
        
        self._innernodes  = np.where(self._nodes._name==0)[0].astype(self.int_precision)
        self._innodes     = np.where(self._nodes._name==1)[0].astype(self.int_precision)
        self._outnodes    = np.where(self._nodes._name==2)[0].astype(self.int_precision)
        self._uppernodes  = np.where(self._nodes._name==3)[0].astype(self.int_precision)
        self._bottomnodes = np.where(self._nodes._name==4)[0].astype(self.int_precision)
        self._halonodes   = np.where(self._nodes._name==10)[0].astype(self.int_precision)
        if self._size == 1:
            self._halonodes   = np.asarray([], dtype=self.int_precision)
        
        self._periodicinnodes     = np.where(self._nodes._name==11)[0].astype(self.int_precision)
        self._periodicoutnodes    = np.where(self._nodes._name==22)[0].astype(self.int_precision)
        self._periodicuppernodes  = np.where(self._nodes._name==33)[0].astype(self.int_precision)
        self._periodicbottomnodes = np.where(self._nodes._name==44)[0].astype(self.int_precision)
        
        
        self._boundarynodes = np.concatenate([self._innodes, self._outnodes, self._bottomnodes, self._uppernodes] )
        self._periodicboundarynodes = np.concatenate([self._periodicinnodes, self._periodicoutnodes, self._periodicbottomnodes, 
                                                      self._periodicuppernodes] )
        
        if self._dim == 3:
            
            self._frontfaces         = np.where(self._faces._name==5)[0].astype(self.int_precision)
            self._backfaces          = np.where(self._faces._name==6)[0].astype(self.int_precision)
            self._periodicfrontfaces = np.where(self._faces._name==55)[0].astype(self.int_precision)
            self._periodicbackfaces  = np.where(self._faces._name==66)[0].astype(self.int_precision)

            self._frontnodes         = np.where(self._nodes._name==5)[0].astype(self.int_precision)
            self._backnodes          = np.where(self._nodes._name==6)[0].astype(self.int_precision)
            self._periodicfrontnodes = np.where(self._nodes._name==55)[0].astype(self.int_precision)
            self._periodicbacknodes  = np.where(self._nodes._name==66)[0].astype(self.int_precision)

            self._boundaryfaces = np.concatenate([self._boundaryfaces, self._backfaces, self._frontfaces] )
            self._periodicboundaryfaces = np.concatenate([self._periodicboundaryfaces, self._periodicbackfaces, self._periodicfrontfaces] )
            
            self._boundarynodes = np.concatenate([self._boundarynodes, self._backnodes, self._frontnodes] )
            self._periodicboundarynodes = np.concatenate([self._periodicboundarynodes, self._periodicbacknodes, self._periodicfrontnodes] )
            
        self._boundaryfaces = np.sort(self._boundaryfaces)
        self._periodicboundaryfaces = np.sort(self._periodicboundaryfaces)
        self._boundarynodes = np.sort(self._boundarynodes)
        self._periodicboundarynodes = np.sort(self._periodicboundarynodes)    
        
        
    def _update_ghost_info(self):
        
        if self._dim == 2:
            self._cells._halonid, self._faces._ghostcenter, \
            self._nodes._ghostcenter, self._nodes._ghostfaceinfo = face_info_2d(self._faces._cellid, self._cells._center, self._cells._nodeid, \
                                                                                self._faces._nodeid, self._boundaryfaces, self._faces._oldname, \
                                                                                self._faces._center, self._faces._normal, self._nodes.vertex, \
                                                                                self._nodes._halonid, self._nbcells, self._nbfaces, self._nbnodes, \
                                                                                self._size, self._nodes._name, self.float_precision )
            
        elif self._dim == 3:
            self._cells._halonid, self._faces._ghostcenter, \
            self._nodes._ghostcenter, self._nodes._ghostfaceinfo = face_info_3d(self._faces._cellid, self._cells._center, self._cells._nodeid, \
                                                                                self._faces._nodeid, self._boundaryfaces, self._faces._oldname, \
                                                                                self._faces._center, self._faces._normal, self._faces._mesure, 
                                                                                self._nodes.vertex,  self._nodes._halonid, self._nbcells, self._nbfaces, \
                                                                                self._nbnodes, self._size, self._nodes._name, self.float_precision )
            
    def _prepare_communication(self):
        #######################################################################################################
        #compute the arrays needed for the mpi communication
        self._halos._scount, self._halos._rcount, self._halos._indsend, self._nbhalos, self._halos._comm_ptr = prepare_comm(self._cells, self._halos)
        
        
    def _update_periodic_boudaries(self):
        #########################################################################################################
        self._nodes._periodicid  = np.zeros((self._nbnodes,2), dtype=np.int32)
        self._cells._periodicnid = np.zeros((self._nbcells,2), dtype=np.int32)
        self._cells._periodicfid = np.zeros(self._nbcells, dtype=np.int32)
        self._cells._shift       = np.zeros((self._nbcells, 3), dtype=self.float_precision)
        
        self._BCs = {"in":["neumann", 1], "out":["neumann", 2],  "upper":["neumann", 3], "bottom":["neumann", 4]}
   
        if len(self._periodicinfaces) != 0:
             self._BCs["in"]  =  ["periodic", 11]
             self._BCs["out"] =  ["periodic", 22]
                
        if len(self._periodicupperfaces) != 0:
            self._BCs["bottom"] = ["periodic", 44]
            self._BCs["upper"] = ["periodic", 33]
        
        if len(self._periodicboundaryfaces) > 0:
            #update periodic boundaries
            if self.dim == 2:
                update_pediodic_info_2d(self._faces.center, self._faces.cellid, self._cells.cellnid, self._cells._center, self._nodes._vertex,
                                        self._nodes.cellid, self._nbnodes, self._nbcells, 
                                        self._periodicinfaces, self._periodicoutfaces, self._periodicupperfaces, self._periodicbottomfaces,
                                        self._periodicinnodes, self._periodicoutnodes, self._periodicuppernodes, self._periodicbottomnodes)
            elif self.dim == 3:
                update_pediodic_info_3d(self._faces.center, self._faces.cellid, self._cells.cellnid, self._cells._center, self._nodes._vertex,
                                        self._nodes.cellid, self._nbnodes, self._nbcells, 
                                        self._periodicinfaces, self._periodicoutfaces, self._periodicupperfaces, self._periodicbottomfaces,
                                        self._periodicinnodes, self._periodicoutnodes, self._periodicuppernodes, self._periodicbottomnodes,
                                        self._periodicfrontfaces, self._periodicbackfaces, self._periodicfrontnodes, self._periodicbacknodes)
        
        if self.dim == 3:
            self._BCs["front"] = ["neumann", 5]
            self._BCs["back"]  = ["neumann", 6]
            
            if len(self._periodicfrontfaces) != 0:
                 self._BCs["front"] = ["periodic", 55]
                 self._BCs["back"] = ["periodic", 66]
        
                
    def _update_haloghost(self):
        
        if self._dim == 2:
            self._halos._sizehaloghost = update_haloghost_info_2d(self._nodes, self._cells, self._halos, self.nbnodes, self.halonodes, \
                                                                  self._halos._comm_ptr, self.float_precision, self.mpi_precision)
            
            
        elif self._dim == 3:
            self._halos._sizehaloghost = update_haloghost_info_3d(self._nodes, self._cells, self._halos, self.nbnodes, self.halonodes, \
                                                                  self._halos._comm_ptr, self.float_precision, self.mpi_precision)
            
            
        
        #TODO add ghostid and ghostnid
        self._nodes._ghostid, self._cells._ghostnid = self.backend.compile(create_node_ghostid, signature=self.signature, \
                                                                           forcedbackend=self.forcedbackend)(self._nodes.ghostcenter, \
                                                                          self._cells._nodeid)
        self._nodes._haloghostid = np.zeros((self._nbnodes, 2), dtype=np.int32)
        self._cells._haloghostnid = np.zeros((self._nbcells, 2), dtype=np.int32)
        
        if self._size > 1:
            #function already compiled
            self._nodes._haloghostid, self._cells._haloghostnid = create_node_ghostid(self._nodes.haloghostcenter, self._cells._nodeid)
        
        
    def _compute_diamondCell_info(self):
        
        self._nodes._R_x      = np.zeros(self._nbnodes, dtype=self.float_precision)
        self._nodes._R_y      = np.zeros(self._nbnodes, dtype=self.float_precision)
        self._nodes._R_z      = np.zeros(self._nbnodes, dtype=self.float_precision)
        self._nodes._lambda_x = np.zeros(self._nbnodes, dtype=self.float_precision)
        self._nodes._lambda_y = np.zeros(self._nbnodes, dtype=self.float_precision)
        self._nodes._lambda_z = np.zeros(self._nbnodes, dtype=self.float_precision)
        self._nodes._number   = np.zeros(self._nbnodes, dtype=self.int_precision)
        
        self._faces._airDiamond = np.zeros(self._nbfaces, dtype=self.float_precision)
        self._faces._param1     = np.zeros(self._nbfaces, dtype=self.float_precision)
        self._faces._param2     = np.zeros(self._nbfaces, dtype=self.float_precision)
        self._faces._param3     = np.zeros(self._nbfaces, dtype=self.float_precision)
        self._faces._param4     = np.zeros(self._nbfaces, dtype=self.float_precision)

        
        self._faces._f_1  = np.zeros((self._nbfaces, self._dim), dtype=self.float_precision)
        self._faces._f_2  = np.zeros((self._nbfaces, self._dim), dtype=self.float_precision)
        self._faces._f_3  = np.zeros((self._nbfaces, self._dim), dtype=self.float_precision)
        self._faces._f_4  = np.zeros((self._nbfaces, self._dim), dtype=self.float_precision)
            
        if self._dim == 2:        
            self.backend.compile(variables_2d, signature=self.signature)(self._cells._center, self._nodes._cellid, self._nodes._halonid, \
                                self._nodes._ghostid, self._nodes._haloghostid,\
                                self._nodes._periodicid, self._nodes._vertex, self._faces._ghostcenter, self._cells._haloghostcenter, \
                                self._halos._centvol, self._nodes._R_x, self._nodes._R_y, self._nodes._lambda_x, self._nodes._lambda_y, 
                                self._nodes._number, self._cells._shift)
            
            
            self.backend.compile(face_gradient_info_2d, signature=self.signature)(self._faces._cellid, self._faces._nodeid, self._faces._ghostcenter,\
                                self._faces._name, self._faces._normal, 
                                self._cells._center, self._halos._centvol, self._faces._halofid, self._nodes._vertex, self._faces._airDiamond, 
                                self._faces._param1, self._faces._param2, self._faces._param3, self._faces._param4, self._faces._f_1, 
                                self._faces._f_2, self._faces._f_3, self._faces._f_4, self._cells._shift, self._dim)
            
            
        elif self._dim == 3:
            
            self.backend.compile(variables_3d, signature=self.signature)(self._cells._center, self._nodes._cellid, self._nodes._halonid, \
                                self._nodes._ghostid, self._nodes._haloghostid,
                                self._nodes._periodicid, self._nodes._vertex, self._faces._ghostcenter, self._cells._haloghostcenter, self._halos._centvol, 
                                self._nodes._R_x, self._nodes._R_y, self._nodes._R_z, self._nodes._lambda_x, self._nodes._lambda_y, 
                                self._nodes._lambda_z, self._nodes._number, self._cells._shift)
            
            
            self.backend.compile(face_gradient_info_3d, signature=self.signature)(self._faces._cellid, self._faces._nodeid, self._faces._ghostcenter,\
                                self._faces._name,
                                self._faces._normal, self._cells._center, self._halos._centvol, 
                                self._faces._halofid, self._nodes._vertex, self._faces._airDiamond, self._faces._param1, 
                                self._faces._param2, self._faces._param3, self._faces._f_1, self._faces._f_2, self._cells._shift, self._dim)
            
            
    def _compute_orthocenter(self):
        
        #distance orthocenter
        self._faces._dist_ortho = np.zeros(self.nbfaces, dtype=self.float_precision)
        if self._dim == 2:
            self.backend.compile(dist_ortho_function, signature=self.signature, forcedbackend=self.forcedbackend)(self._innerfaces, self._boundaryfaces, self._infaces, \
                                self._faces._cellid, self._cells._center, 
                                self._faces._dist_ortho, self._faces._center, self._faces._normal, self._faces.mesure)
            
    def __str__(self):
        """Pretty printing"""
        txt = '\n'
        txt += '> dim   :: {dim}\n'.format(dim=self.dim)
        txt += '> total cells  :: {cells}\n'.format(cells=self.nbcells)
        txt += '> total nodes  :: {nodes}\n'.format(nodes=self.nbnodes)
        txt += '> total faces  :: {faces}\n'.format(faces=self.nbfaces)

        return txt
    
    def save_on_cell(self, dt=0, time=0, niter=0, miter=0, value=None):
        
        if value is None:
            raise ValueError("value must be given")
        assert len(value) == self.nbcells, 'value size != number of cells'
       
        elements = self._typeOfCells#{"quad": self.cells._nodeid}

        points = self._nodes._vertex[:, :3]
        points = np.array(points, dtype=self.float_precision)
                
        data = {"w" : value}
        data = {"w": data}
        
        maxw = max(value)
        
        integral_maxw = np.zeros(1, dtype=self.float_precision)
    
        self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)
      
        if self.comm.rank == 0:
            print(" **************************** Computing ****************************")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Iteration = ", niter, "time = ", time, "time step = ", dt)
            print("max w =", integral_maxw[0])
    
        meshio.write_points_cells("results/visu"+str(self.comm.rank)+"-"+str(miter)+".vtu",
                                  points, elements, cell_data=data, file_format="vtu")
    
        if(self.comm.rank == 0):
            with open(self._vtkpath+"/visu"+str(miter)+".pvtu", "w") as text_file:
                text_file.write("<?xml version=\"1.0\"?>\n")
                text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
                text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
                text_file.write("<PPoints>\n")
                text_file.write("<PDataArray type=\""+self.vtkprecision+"\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
                text_file.write("</PPoints>\n")
                text_file.write("<PCells>\n")
                text_file.write("<PDataArray type=\"uint32\" Name=\"connectivity\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"uint32\" Name=\"offsets\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"uint32\" Name=\"types\" format=\"binary\"/>\n")
                text_file.write("</PCells>\n")
                text_file.write("<PCellData Scalars=\"h\">\n")
                text_file.write("<PDataArray type=\""+self.vtkprecision+"\" Name=\"w\" format=\"binary\"/>\n")
                text_file.write("</PCellData>\n")

                for i in range(self.comm.size):
                    name1 = "visu"
                    bu1 = [10]
                    bu1 = str(i)
                    name1 += bu1
                    name1 += "-"+str(miter)
                    name1 += ".vtu"
                    text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
                text_file.write("</PUnstructuredGrid>\n")
                text_file.write("</VTKFile>")
                
    def save_on_node(self, dt=0, time=0, niter=0, miter=0, value=None):
        
        if value is None:
            raise ValueError("value must be given")
        assert len(value) == self.nbnodes, 'value size != number of nodes'
        
        elements = self._typeOfCells#{"quad": self.cells._nodeid}
        points = self._nodes._vertex[:, :3]
        
        data = {"w" : value}
        
        maxw = max(value)
        
        integral_maxw = np.zeros(1, dtype=self.float_precision)
    
        self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)
      
        if self.comm.rank == 0:
            print(" **************************** Computing ****************************")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Iteration = ", niter, "time = ", time, "time step = ", dt)
            print("max w =", integral_maxw[0])
    
        meshio.write_points_cells("results/visu"+str(self.comm.rank)+"-"+str(miter)+".vtu",
                                  points, elements, point_data=data, file_format="vtu")
    
        if(self.comm.rank == 0):
            with open(self._vtkpath+"/visu"+str(miter)+".pvtu", "w") as text_file:
                text_file.write("<?xml version=\"1.0\"?>\n")
                text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
                text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
                text_file.write("<PPoints>\n")
                text_file.write("<PDataArray type=\""+self.vtkprecision+"\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
                text_file.write("</PPoints>\n")
                text_file.write("<PCells>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"connectivity\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"offsets\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"types\" format=\"binary\"/>\n")
                text_file.write("</PCells>\n")
                text_file.write("<PPointData Scalars=\"h\">\n")
                text_file.write("<PDataArray type=\""+self.vtkprecision+"\" Name=\"w\" format=\"binary\"/>\n")
                text_file.write("</PPointData>\n")

                for i in range(self.comm.size):
                    name1 = "visu"
                    bu1 = [10]
                    bu1 = str(i)
                    name1 += bu1
                    name1 += "-"+str(miter)
                    name1 += ".vtu"
                    text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
                text_file.write("</PUnstructuredGrid>\n")
                text_file.write("</VTKFile>")
                
    def save_on_node_multi(self, dt=0, time=0, niter=0, miter=0, variables=None, values=None, file_format="vtu"):
        
        if values is None:
            raise ValueError("value must be given")
        assert len(values[0]) == self.nbnodes, 'value size != number of nodes'
        
        elements = self._typeOfCells#{"quad": self.cells._nodeid}
        points = self._nodes._vertex[:, :3]
        
        nvalues=len(values)
        data={}
        for k in range(0,nvalues):
            data[variables[k]]=values[k]
        
        maxw = max(values[0])
        
        integral_maxw = np.zeros(1, dtype=self.float_precision)
    
        self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)
      
        if self.comm.rank == 0:
            print(" **************************** Computing ****************************")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Iteration = ", niter, "time = ", time, "time step = ", dt)
            print("max"+variables[0]+" =", integral_maxw[0])
                                  
        meshio.write_points_cells("results/visu"+str(self.comm.rank)+"-"+str(miter)+"."+file_format,
                                  points, elements, point_data=data, file_format=file_format)
    
        if(self.comm.rank == 0):
            with open(self._vtkpath+"/visu"+str(miter)+".pvtu", "w") as text_file:
                text_file.write("<?xml version=\"1.0\"?>\n")
                text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
                text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
                text_file.write("<PPoints>\n")
                text_file.write("<PDataArray type=\""+self.vtkprecision+"\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
                text_file.write("</PPoints>\n")
                text_file.write("<PCells>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"connectivity\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"offsets\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"types\" format=\"binary\"/>\n")
                text_file.write("</PCells>\n")
                text_file.write("<PPointData Scalars=\"h\">\n")
                for k in range(0,nvalues):
                    text_file.write("<PDataArray type=\""+self.vtkprecision+"\" Name=\""+variables[k]+"\" format=\"binary\"/>\n")
                text_file.write("</PPointData>\n")

                for i in range(self.comm.size):
                    name1 = "visu"
                    bu1 = [10]
                    bu1 = str(i)
                    name1 += bu1
                    name1 += "-"+str(miter)
                    name1 += ".vtu"
                    text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
                text_file.write("</PUnstructuredGrid>\n")
                text_file.write("</VTKFile>")
                
    def save_on_cell_multi(self, dt=0, time=0, niter=0, miter=0, variables=None, values=None, file_format="vtu"):
        
        if values is None:
            raise ValueError("value must be given")
        assert len(values[0]) == self.nbcells, 'value size != number of cells'
        
        elements = self._typeOfCells#{"triangle": self.cells._nodeid}
        points = self._nodes._vertex[:, :3]
        
        nvalues=len(values)
        data_bis={}
        for k in range(0,nvalues):
            data_bis[variables[k]]=values[k]
        
        data = {}
        for k in range(0,nvalues):
            data[variables[k]]=data_bis
            
        maxw = max(values[0])
        
        integral_maxw = np.zeros(1, dtype=self.float_precision)
    
        self.comm.Reduce(maxw, integral_maxw, MPI.MAX, 0)
      
        if self.comm.rank == 0:
            print(" **************************** Computing ****************************")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Saving Results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Iteration = ", niter, "time = ", time, "time step = ", dt)
            print("max"+variables[0]+" =", integral_maxw[0])
    
        meshio.write_points_cells("results/visu"+str(self.comm.rank)+"-"+str(miter)+"."+file_format,
                                  points, elements, cell_data=data, file_format=file_format)
        
        if(self.comm.rank == 0):
            with open(self._vtkpath+"/visu"+str(miter)+".pvtu", "w") as text_file:
                text_file.write("<?xml version=\"1.0\"?>\n")
                text_file.write("<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
                text_file.write("<PUnstructuredGrid GhostLevel=\"0\">\n")
                text_file.write("<PPoints>\n")
                text_file.write("<PDataArray type=\""+self.vtkprecision+"\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\"/>\n")
                text_file.write("</PPoints>\n")
                text_file.write("<PCells>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"connectivity\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"offsets\" format=\"binary\"/>\n")
                text_file.write("<PDataArray type=\"int32\" Name=\"types\" format=\"binary\"/>\n")
                text_file.write("</PCells>\n")
                text_file.write("<PCellData Scalars=\"h\">\n")
                for k in range(0,nvalues):
                    text_file.write("<PDataArray type=\""+self.vtkprecision+"\" Name=\""+variables[k]+"\" format=\"binary\"/>\n")
                text_file.write("</PCellData>\n")

                for i in range(self.comm.size):
                    name1 = "visu"
                    bu1 = [10]
                    bu1 = str(i)
                    name1 += bu1
                    name1 += "-"+str(miter)
                    name1 += ".vtu"
                    text_file.write("<Piece Source=\""+str(name1)+"\"/>\n")
                text_file.write("</PUnstructuredGrid>\n")
                text_file.write("</VTKFile>")
    
    @property
    def cells(self):
        return self._cells

    @property
    def faces(self):
        return self._faces

    @property
    def nodes(self):
        return self._nodes

    @property
    def halos(self):
        return self._halos

    @property
    def dim(self):
        return self._dim
    
    @property
    def comm(self):
        return self._comm
    
    @property
    def nbnodes(self):
        return self._nbnodes
    
    @property
    def nbcells(self):
        return self._nbcells
    
    @property
    def nbfaces(self):
        return self._nbfaces
    
    @property
    def nbhalos(self):
        return self._nbhalos
    
    @property
    def innerfaces(self):
        return self._innerfaces
    
    @property
    def infaces(self):
        return self._infaces
    
    @property
    def outfaces(self):
        return self._outfaces
    
    @property
    def bottomfaces(self):
        return self._bottomfaces
    
    @property
    def upperfaces(self):
        return self._upperfaces
    
    @property
    def halofaces(self):
        return self._halofaces
    
    @property
    def innernodes(self):
        return self._innernodes
    
    @property
    def innodes(self):
        return self._innodes
    
    @property
    def outnodes(self):
        return self._outnodes
    
    @property
    def bottomnodes(self):
        return self._bottomnodes
    
    @property
    def uppernodes(self):
        return self._uppernodes
    
    @property
    def halonodes(self):
        return self._halonodes
    
    @property
    def boundaryfaces(self):
        return self._boundaryfaces
    
    @property
    def boundarynodes(self):
        return self._boundarynodes
    
    @property
    def periodicboundaryfaces(self):
        return self._periodicboundaryfaces
    
    @property
    def periodicboundarynodes(self):
        return self._periodicboundarynodes
    
    @property
    def typeOfCells(self):
        return self._typeOfCells
    
    @property
    def bounds(self):
        return self._bounds

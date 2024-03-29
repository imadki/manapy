#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kissami
"""
# coding: utf-8

__all__ = ['MeshPartition']

from manapy.partitions.partitions_utils      import create_npart_cpart, compute_halocell, define_ghost_node
from manapy.backends  import CPUBackend
from manapy.base.base import make_get_conf
from manapy.base.base import Struct
import warnings

import os
import numpy as np

def unique_func(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


class MeshPartition():
    
    @classmethod
    def process_conf(cls, conf, kwargs):
        """                                                                                                                                                                                                
        Process configuration parameters.                                                                                                                                                                  
        """
        get = make_get_conf(conf, kwargs)
        
        if len(cls._parameters) and cls._parameters[0][0] != 'name':
            options = MeshPartition._parameters + cls._parameters

        else:
            options = MeshPartition._parameters
            
        
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
    
    name = 'gmsh'
    extn = ['.msh']

    def __init__(self, filename = None, dim = None, comm = None, conf=None,  periodic=[0,0,0], **kwargs):
        
        if conf is None:
            conf = Struct()
        self.conf = self.process_conf(conf, kwargs)
        get = make_get_conf(self.conf, kwargs)
        
        if get("python"):
            warnings.warn("You are running pure python functions", Warning)
        
        self.backend = backend = CPUBackend(multithread=get("multithreading"), backend=get("backend"), 
                                            cache=get("cache"), int_precision=get("int_precision"))
        self.signature = get('signature')
        
        if get('float_precision') == "single":
            self.float_precision = 'f4'
        else :
            self.float_precision = 'f8'
            
        if get('int_precision') == "signed":
            self.int_precision = 'i4'
        else :
            self.int_precision = 'u4'
            
        # Node numbers associated with each element face
        self._etypes = {2:['triangle','quad'], 3:['tetra', 'pyramid', 'hexahedron']}
        
        self._filename = filename
        if self._filename is None:
            raise ValueError("filename must be given")
       
        self._dim = dim
        if self._dim is None:
            raise ValueError("dim must be given")     
        
        self._periodic = periodic
        
        self._comm = comm
        
        if self._comm is None:
            from mpi4py import MPI
            self._comm = MPI.COMM_WORLD
        
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()
        
        if self._dim == 2:
            from manapy.partitions.partitions_utils import convert_2d_cons_to_array
            self._convert_cons_to_array = backend.compile(convert_2d_cons_to_array, signature=self.signature)
        else:
            from manapy.partitions.partitions_utils import convert_3d_cons_to_array
            self._convert_cons_to_array = backend.compile(convert_3d_cons_to_array, signature=self.signature)
        
        from mpi4py import MPI
        #read mesh            
        if self._rank == 0:
            print("Reading gmsh file ...")
            self._readgmsh()

        #make partition
        if self._size > 1 and self._rank == 0 :
            print("Mesh partitionning ...")
            self._make_partition()

        #save mesh files
        if self._rank == 0:
            print("Saving partition files ...")
            self._savemesh()
        
            print("Number of Cells:", self._nbcells)
            print("Number of Vertices:", self._nbnodes)
            
        self._comm.Barrier()
        
        
    def __str__(self):
        """Pretty printing"""
        txt = '\n'
        txt += '> dim   :: {dim}\n'.format(dim=self.dim)
        txt += '> total cells  :: {cells}\n'.format(cells=self._nbcells)
        txt += '> total nodes  :: {nodes}\n'.format(nodes=self._nbnodes)

        return txt
        
    def _readgmsh(self):
        import meshio
        #load mesh
        self._mesh = meshio.read(self._filename)
        self._nbnodes = len(self._mesh.points)
        
        #coordinates x, y of each node
        self._vertices = np.zeros((self._nbnodes, 4), dtype=self.float_precision)
        self._vertices[:,:3] = self._mesh.points
        self._vertices[:,3]  = define_ghost_node(self._mesh, self._periodic, self._nbnodes, self._dim)

        from collections import defaultdict
        #nodes of each cell
        self._elements = {}
        self._elemetis = defaultdict(list)
        cmpt = 0
        for i in self._etypes[self._dim]:
            if i in self._mesh.cells.keys():
                self._elements["ele"+str(cmpt)] = self._mesh.cells[i].astype(self.int_precision)
                self._elemetis["ele"+str(cmpt)] = self._mesh.cells[i].astype(self.int_precision)
                cmpt += 1
            else:
                self._elements["ele"+str(cmpt)] = np.zeros((1,1), dtype=self.int_precision)
                cmpt += 1
                
        if self._dim == 2:   self._convert_args = [self._elements["ele0"], self._elements["ele1"]]
        elif self._dim == 3: self._convert_args = [self._elements["ele0"], self._elements["ele1"], self._elements["ele2"]]
            
        
    def _make_partition(self):
        from mgmetis import metis
        from mgmetis.enums import OPTION
        
        #Partitioning mesh
        opts = metis.get_default_options()
        opts[OPTION.MINCONN] = 1
        opts[OPTION.CONTIG] = 1
        opts[OPTION.NUMBERING] = 0
        opts[OPTION.OBJTYPE] = 1
        options = opts
        
        elem = [item for sublist in self._elemetis.values() for item in sublist]
        objval, self._epart, self._npart = metis.part_mesh_dual(self._size, elem, 
                                                                opts=options, nv=self._nbnodes)
        
        
        
        self._cell_nodeid = self._convert_cons_to_array(*self._convert_args).astype(self.int_precision)
        
        self._nbcells = len(self._cell_nodeid)

        self._npart, \
        self._cpart, \
        self._neighsub, \
        self._halo_cellid, \
        self._globcelltoloc, \
        self._locnodetoglob, self._tc = self.backend.compile(create_npart_cpart, signature=False)(self._cell_nodeid, self._npart, self._epart,
                                                                                                   self._nbnodes, self._nbcells, self._size)
        
        for i in range(self._size):
            self._neighsub[i]      = np.unique(self._neighsub[i])
            self._neighsub[i]      = self._neighsub[i][self._neighsub[i]!=i]
            self._halo_cellid[i]   = np.unique(self._halo_cellid[i])
            self._locnodetoglob[i] = unique_func(self._locnodetoglob[i])
            
        self._centvol, \
        self._haloextloc, \
        self._halointloc, \
        self._halointlen = compute_halocell(self._halo_cellid, self._cpart, self._cell_nodeid, 
                                            self._vertices, self._neighsub, self._size, self._dim, 
                                            self.float_precision)
        
    def _savemesh(self):
        
        import h5py
        self._mesh_dir = "meshes"+str(self._size)+"PROC"
        if not os.path.exists(self._mesh_dir):
            os.mkdir(self._mesh_dir)
        
        if self._size == 1:
            self._cell_nodeid = self._convert_cons_to_array(*self._convert_args).astype(self.int_precision)
            self._nbcells = len(self._cell_nodeid)
            
            filename = os.path.join(self._mesh_dir, "mesh0.hdf5")
            if os.path.exists(filename):
                os.remove(filename)
            
            filename = os.path.join(self._mesh_dir, "mesh0.hdf5")
            with h5py.File(filename, "w") as f:
                f.create_dataset("elements", data=self._cell_nodeid)
                f.create_dataset("nodes", data=self._vertices)
            f.close()
        else:
            for i in range(self._size):
                filename = os.path.join(self._mesh_dir, "mesh"+str(i)+".hdf5")
                with h5py.File(filename, "w") as f:
                    f.create_dataset("elements", data=self._cell_nodeid[self._globcelltoloc[i]])
                    f.create_dataset("nodes", data=self._vertices[self._locnodetoglob[i]])
                    f.create_dataset("nparts",data=self._npart[self._locnodetoglob[i]], dtype=np.int32)
                    if i == 0:
                        f.create_dataset("GtoL", data=self._tc, dtype=np.int32)
                    f.create_dataset("centvol", data=self._centvol[i])
                    f.create_dataset("localnodetoglobal", data=self._locnodetoglob[i], dtype=np.int32)
                    f.create_dataset("localcelltoglobal", data=self._globcelltoloc[i], dtype=np.int32)
                    f.create_dataset("neigh1", data=self._neighsub[i], dtype=np.int32)
                    f.create_dataset("neigh2", data=self._halointlen[i], dtype=np.int32)
                    f.create_dataset("halosext", data=self._haloextloc[i], dtype=np.int32)
                    f.create_dataset("halosint", data=self._halointloc[i], dtype=np.int32)
                    
                f.close()

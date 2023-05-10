#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:53:35 2022

@author: kissami
"""

import numpy as np

from manapy.comms import all_to_all, Iall_to_all, define_halosend

from manapy.ast.ast_utils  import (facetocell, celltoface)
from manapy.ast.functions2d import (centertovertex_2d, face_gradient_2d,
                                    cell_gradient_2d, barthlimiter_2d)

from manapy.ast.functions3d import (centertovertex_3d, face_gradient_3d,
                                    cell_gradient_3d, barthlimiter_3d)

 
from types import LambdaType
from manapy.boundary.bc  import Boundary

class Variable():
    is_called = False
    
    """ """
    def __init__(self, domain=None, terms=None, comm=None, name=None, BC=None, values=None, *args, **kwargs):
        if domain is None:
            raise ValueError("domain must be given")
        
        self._domain = domain
        self._BC = BC        
        self._values = values
        
        self.backend = self._domain.backend
        self.signature = self._domain.signature
        self.float_precision  = self._domain.float_precision
        self.mpi_precision = self._domain.mpi_precision
        self.forcedbackend = self._domain.forcedbackend
        
        self._dim = self.domain.dim
        self._comm   = self.domain.halos.comm_ptr
        
        self._nbfaces = self.domain.nbfaces
        self._nbcells = self.domain.nbcells
        self._nbnodes = self.domain.nbnodes
        self._nbhalos = self.domain.nbhalos
        self._nbghost = self.domain.nbfaces
        
        self.cell = np.zeros(self.nbcells, dtype=self.float_precision)
        self.node = np.zeros(self.nbnodes, dtype=self.float_precision)
        self.face = np.zeros(self.nbfaces, dtype=self.float_precision)
        self.ghost = np.zeros(self.nbfaces, dtype=self.float_precision)
        self.halo = np.zeros(self.nbhalos, dtype=self.float_precision)
        
        self.gradcellx = np.zeros(self._nbcells, dtype=self.float_precision)
        self.gradcelly = np.zeros(self._nbcells, dtype=self.float_precision)
        self.gradcellz = np.zeros(self._nbcells, dtype=self.float_precision)
        
        self.gradhalocellx = np.zeros(self._nbhalos, dtype=self.float_precision)
        self.gradhalocelly = np.zeros(self._nbhalos, dtype=self.float_precision)
        self.gradhalocellz = np.zeros(self._nbhalos, dtype=self.float_precision)
        
        self.gradfacex = np.zeros(self._nbfaces, dtype=self.float_precision)
        self.gradfacey = np.zeros(self._nbfaces, dtype=self.float_precision)
        self.gradfacez = np.zeros(self._nbfaces, dtype=self.float_precision)
        
        self.psi = np.zeros(self._nbcells, dtype=self.float_precision)
        self.psihalo = np.zeros(self._nbhalos, dtype=self.float_precision)
        
        self.halotosend  = np.zeros(len(self.domain.halos.halosint), dtype=self.float_precision)
        self.haloghost = np.zeros(self.domain.halos.sizehaloghost, dtype=self.float_precision)
        
        self._name    = name
        
        if terms is not None:
            for i in terms:
                self.__dict__[i] = np.zeros(self.nbcells)
       
        self._update_boundaries()
        Variable.compile_func(self._dim, self.backend, self.signature, self.forcedbackend)
        
    @classmethod
    def compile_func(cls, dim, backend, signature, forcedbackend):
        if not cls.is_called:
            # Perform the compilation using numba.jit or other numba decorators
            cls._facetocell    = backend.compile(facetocell, signature=signature)
            cls._celltoface    = backend.compile(celltoface, signature=signature)
            cls._define_halosend = backend.compile(define_halosend, signature=signature)
            if dim == 2:
                cls._func_interp   = backend.compile(centertovertex_2d, signature=signature)
                cls._face_gradient = backend.compile(face_gradient_2d, signature=signature)
                cls._cell_gradient = backend.compile(cell_gradient_2d, signature=signature)
                cls._barthlimiter  = backend.compile(barthlimiter_2d, signature=signature)
            elif dim == 3:
                cls._func_interp   = backend.compile(centertovertex_3d, signature=signature)
                cls._face_gradient = backend.compile(face_gradient_3d, signature=signature)
                cls._cell_gradient = backend.compile(cell_gradient_3d, signature=signature)
                cls._barthlimiter  = backend.compile(barthlimiter_3d, signature=signature)
            cls.is_called = True
            

    def _update_boundaries(self):
            
        if self._dim == 2:
            self._BCs = {"in":None, "out":None, "bottom":None, "upper":None}
        elif self._dim == 3:
            self._BCs = {"in":None, "out":None, "bottom":None, "upper":None, "front":None, "back":None}

        
        self.domain.Pbordnode = np.zeros(self.domain.nbnodes, dtype=self.float_precision)
        self.domain.Pbordface = np.zeros(self.domain.nbfaces, dtype=self.float_precision)
        
        self.dirichletfaces = []
        self.neumannfaces = []
        self.neumannNHfaces = []
        
        self.BCdirichlet = []
        self.BCneumann = []
        self.BCneumannNH = []
        
        
        valueface = np.zeros(self.domain._nbfaces, dtype=self.float_precision)
        valuenode = np.zeros(self.domain._nbnodes, dtype=self.float_precision)
        valuehalo = np.zeros(self.domain.halos.sizehaloghost, dtype=self.float_precision)
        
        if self._BC is None:
            for loc in self._BCs.keys():
                if self.domain._BCs[loc][0] == "periodic":
                    self.BCs[loc] = Boundary(BCtype = "periodic", BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCs[loc].BCvalueface = np.array([], dtype=self.float_precision)
                    self.BCs[loc].BCvaluenode = np.array([], dtype=self.float_precision)
                    self.BCs[loc].BCvaluehalo = np.array([], dtype=self.float_precision)
                   
                else:
                    self.BCs[loc] = Boundary(BCtype = "neumann", BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumann.append(self.BCs[loc]._BCtypeindex)
                    self.neumannfaces.extend(self.BCs[loc]._BCfaces)
                    
                    valueface = self.cell
                    valuenode = self.node
                    valuehalo = self.halo
                
                self.BCs[loc].BCvalueface = valueface
                self.BCs[loc].BCvaluenode = valuenode
                self.BCs[loc].BCvaluehalo = valuehalo
            
        else:
            
            for loc, bct in self._BC.items():
                if self.domain._BCs[loc][0] == "periodic":
                    if bct != "periodic":
                        raise ValueError("BC must be periodic for "+ str(loc))
                        
                elif self.domain._BCs[loc][0] != "periodic":
                    if bct == "periodic":
                        raise ValueError("BC must be not periodic for "+ str(loc))
                        
                
                
                if bct == "dirichlet":# or bct =="neumannNH":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCdirichlet.append(self.BCs[loc]._BCtypeindex)
                    self.dirichletfaces.extend(self.BCs[loc]._BCfaces)
                
                    if loc not in self._values.keys():
                        raise ValueError("Value of dirichlet BC for "+str(loc)+" faces must be given")
                    
                    #TODO check valuehalo (face center miss)
                    if isinstance(self._values[loc], LambdaType):
                        for i in self.BCs[loc]._BCfaces:
                            valueface[i] = self._values[loc](self.domain.faces.center[i][0], self.domain.faces.center[i][1], 
                                                        self.domain.faces.center[i][2])
                        for i in np.where(self.domain.nodes.oldname==self.BCs[loc]._BCtypeindex)[0]:
                            valuenode[i] = self._values[loc](self.domain.nodes.vertex[i][0], self.domain.nodes.vertex[i][1], 
                                                        self.domain.nodes.vertex[i][2])
                        
                            for j in range(len(self.domain.nodes.haloghostcenter[i])):
                                
                                cell = int(self.domain.nodes.haloghostcenter[i][j][-1])
                                if cell != -1:
                                    center = self.domain.nodes.haloghostfaceinfo[i][j][0:3]
                                    valuehalo[cell] = self._values[loc](center[0], center[1], center[2])
    
                    elif isinstance(self._values[loc], (int, float)):
                        for i in self.BCs[loc]._BCfaces:
                            valueface[i] = self._values[loc]
                            
                        for i in np.where(self.domain.nodes.oldname==self.BCs[loc]._BCtypeindex)[0]:
                            valuenode[i] = self._values[loc]
                        
                            for j in range(len(self.domain.nodes.haloghostcenter[i])):
                                    cell = int(self.domain.nodes.haloghostcenter[i][j][-1])
                                    if cell != -1:
                                        valuehalo[cell] = self._values[loc]
                                        
                    self.BCs[loc].BCvalueface = valueface
                    self.BCs[loc].BCvaluenode = valuenode
                    self.BCs[loc].BCvaluehalo = valuehalo
                    
                elif bct == "neumannNH":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumannNH.append(self.BCs[loc]._BCtypeindex)
                    self.neumannNHfaces.extend(self.BCs[loc]._BCfaces)
                    
                    if loc not in self._values.keys():
                        raise ValueError("Value of dirichlet BC for "+str(loc)+" faces must be given")
                    
                    #TODO check valuehalo (face center miss)
                    if isinstance(self._values[loc], LambdaType):
                        for i in self.BCs[loc]._BCfaces:
                            valueface[i] = self._values[loc](self.domain.faces.center[i][0], self.domain.faces.center[i][1], 
                                                        self.domain.faces.center[i][2])
                        for i in np.where(self.domain.nodes.oldname==self.BCs[loc]._BCtypeindex)[0]:
                            valuenode[i] = self._values[loc](self.domain.nodes.vertex[i][0], self.domain.nodes.vertex[i][1], 
                                                        self.domain.nodes.vertex[i][2])
                        
                            for j in range(len(self.domain.nodes.haloghostcenter[i])):
                                
                                cell = int(self.domain.nodes.haloghostcenter[i][j][-1])
                                if cell != -1:
                                    center = self.domain.nodes.haloghostfaceinfo[i][j][0:3]
                                    valuehalo[cell] = self._values[loc](center[0], center[1], center[2])
    
                    elif isinstance(self._values[loc], (int, float)):
                        for i in self.BCs[loc]._BCfaces:
                            valueface[i] = self._values[loc]
                            
                        for i in np.where(self.domain.nodes.oldname==self.BCs[loc]._BCtypeindex)[0]:
                            valuenode[i] = self._values[loc]
                        
                            for j in range(len(self.domain.nodes.haloghostcenter[i])):
                                    cell = int(self.domain.nodes.haloghostcenter[i][j][-1])
                                    if cell != -1:
                                        valuehalo[cell] = self._values[loc]
                    
                    self.BCs[loc].constNH = valueface
                    self.BCs[loc].constNHNode = valuenode
                    
                    valueface2 = self.cell
                    valuenode2 = self.node
                    valuehalo2 = self.halo
                    
                    self.BCs[loc].BCvalueface = valueface2
                    self.BCs[loc].BCvaluenode = valuenode2
                    self.BCs[loc].BCvaluehalo = valuehalo2
                        
               
                elif bct == "neumann":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumann.append(self.BCs[loc]._BCtypeindex)
                    self.neumannfaces.extend(self.BCs[loc]._BCfaces)
                    
                    valueface = self.cell
                    valuenode = self.node
                    valuehalo = self.halo
                    
                    self.BCs[loc].BCvalueface = valueface
                    self.BCs[loc].BCvaluenode = valuenode
                    self.BCs[loc].BCvaluehalo = valuehalo
                    
                elif bct == "periodic":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCs[loc].BCvalueface = np.array([], dtype=self.float_precision)
                    self.BCs[loc].BCvaluenode = np.array([], dtype=self.float_precision)
                    self.BCs[loc].BCvaluehalo = np.array([], dtype=self.float_precision)
                    
                    
                elif bct == "slip":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumann.append(self.BCs[loc]._BCtypeindex)
                    self.neumannfaces.extend(self.BCs[loc]._BCfaces)
                    
                    valueface = self.cell
                    valuenode = self.node
                    valuehalo = self.halo
                    
                    self.BCs[loc].BCvalueface = valueface
                    self.BCs[loc].BCvaluenode = valuenode
                    self.BCs[loc].BCvaluehalo = valuehalo
                    
                    
                elif bct == "nonslip":
                    self.BCs[loc] = Boundary(BCtype = bct, BCloc=loc, BCvalueface=self.cell,  BCvaluenode =self.cell, 
                                             BCvaluehalo=self.halo, BCtypeindex=self.domain._BCs[loc][1], domain=self.domain)
                    
                    self.BCneumann.append(self.BCs[loc]._BCtypeindex)
                    self.neumannfaces.extend(self.BCs[loc]._BCfaces)
                    
                    valueface = self.cell
                    valuenode = self.node
                    valuehalo = self.halo
                    
                    self.BCs[loc].BCvalueface = valueface
                    self.BCs[loc].BCvaluenode = valuenode
                    self.BCs[loc].BCvaluehalo = valuehalo
                
        self._BCin     = self.BCs["in"]
        self._BCout    = self.BCs["out"]
        self._BCbottom = self.BCs["bottom"]
        self._BCupper  = self.BCs["upper"]
        
        # print(self.BCs)
        if self.dim == 3:
            self._BCfront = self.BCs["front"]
            self._BCback  = self.BCs["back"]
        elif self.dim == 2:
            self._BCfront = 0
            self._BCback  = 0
        
        self.dirichletfaces.sort()
        self.neumannfaces.sort()
        
        self.dirichletfaces = np.asarray(self.dirichletfaces, dtype=np.uint32)
        self.neumannfaces = np.asarray(self.neumannfaces, dtype=np.uint32)
        self.neumannNHfaces = np.asarray(self.neumannNHfaces, dtype=np.uint32)
        self.BCdirichlet = np.asarray(self.BCdirichlet, dtype=np.uint32)
        self.BCneumannNH = np.asarray(self.BCdirichlet, dtype=np.uint32)
        self.BCneumann = np.asarray(self.BCneumann, dtype=np.uint32)
        
    
    def __str__(self):
        """Pretty printing"""
        txt = '\n'
        txt += '> dim   :: {dim}\n'.format(dim=self.dim)
        txt += '> total cells  :: {cells}\n'.format(cells=self.nbcells)
        txt += '> total nodes  :: {nodes}\n'.format(nodes=self.nbnodes)
        txt += '> total faces  :: {faces}\n'.format(faces=self.nbfaces)
        txt += '> Value on cells :: {faces}\n'.format(faces=self.cell)
        txt += '> Value on faces  :: {faces}\n'.format(faces=self.face)
        txt += '> Value on nodes  :: {faces}\n'.format(faces=self.node)
        txt += '> Value on ghosts  :: {faces}\n'.format(faces=self.ghost)

        return txt
    
    def __add__(self, other):
       
        res = Variable(self.domain)

        res.cell  = self.cell + other.cell
        # res.node  = self.node + other.node
        # res.face  = self.face + other.face
        # res.ghost = self.ghost + other.ghost
        # res.halo  = self.halo + other.halo

        return res
    
    def __sub__(self, other):
       
        res = Variable(self.domain)

        res.cell  = self.cell - other.cell
        # res.node  = self.node - other.node
        # res.face  = self.face - other.face
        # res.ghost = self.ghost - other.ghost
        # res.halo  = self.halo - other.halo

        return res
    
    def __mul__(self, other):
       
        res = Variable(self.domain)

        res.cell  = self.cell * other.cell
        # res.node  = self.node * other.node
        # res.face  = self.face * other.face
        # res.ghost = self.ghost * other.ghost
        # res.halo  = self.halo * other.halo

        return res
    
    def __truediv__(self, other):
       
        res = Variable(self.domain)

        if (other.cell != 0).all():
            res.cell  = self.cell / other.cell
        else:
            raise ValueError("Values of denominator must be different to 0")
        
        # if (other.node != 0).all():
        #     res.node  = self.node / other.node
        
        # if (other.node != 0).all():
        #     res.face  = self.face / other.face
        
        # if (other.node != 0).all():
        #     res.ghost = self.ghost / other.ghost
        # res.halo  = self.halo / other.halo

        return res
    
#    def update_values(self, value=None):
#        
#        self.update_halo_value()
#        self.update_ghost_value()
#        # self.interpolate_celltonode()
        
    def update_halo_value(self):
        #update the halo values
        Variable._define_halosend(self.cell, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.halo, 
                          self.comm, self.mpi_precision)
#        self.comm.Barrier()
    
    def interpolate_facetocell(self):
        Variable._facetocell(self.face, self.cell, self.domain.cells.faceid, self.dim)
    
    def interpolate_celltoface(self):
        Variable._celltoface(self.cell, self.face, self.ghost, self.halo, self.domain.faces.cellid,self.domain.faces.halofid,
                   self.domain.innerfaces, self.domain.boundaryfaces, self.domain.halofaces)
    
    def interpolate_celltonode(self):
        
        # self.update_halo_value()
        # self.update_ghost_value()
        Variable._func_interp(self.cell, self.ghost, self.halo, self.haloghost, self.domain.cells.center, self.domain.halos.centvol, 
                          self.domain.nodes.cellid, self.domain.nodes.ghostid, self.domain.nodes.haloghostid,
                          self.domain.nodes.periodicid, self.domain.nodes.halonid, self.domain.nodes.vertex, 
                          self.domain.faces.ghostcenter, self.domain.cells.haloghostcenter, 
                          self.domain.nodes.R_x, self.domain.nodes.R_y, self.domain.nodes.R_z, self.domain.nodes.lambda_x, 
                          self.domain.nodes.lambda_y, self.domain.nodes.lambda_z,
                          self.domain.nodes.number, self.domain.cells.shift, self.node)
        
        
    def compute_cell_gradient(self):
        
        Variable._cell_gradient(self.cell, self.ghost, self.halo, self.haloghost, self.domain.cells.center,
                            self.domain.cells.cellnid, self.domain.cells.ghostnid, self.domain.cells.haloghostnid,
                            self.domain.cells.halonid, self.domain.cells.nodeid, self.domain.cells.periodicnid, 
                            self.domain.nodes.periodicid, self.domain.faces.ghostcenter,  self.domain.cells.haloghostcenter, 
                            self.domain.nodes.vertex, self.domain.halos.centvol, self.domain.cells.shift, self.gradcellx, 
                            self.gradcelly, self.gradcellz)
        
        
       
        #The limiter depend on hc value
        Variable._barthlimiter(self.cell, self.ghost, self.halo, self.gradcellx, self.gradcelly, self.gradcellz,
                           self.psi, self.domain.faces.cellid, self.domain.cells.faceid, self.domain.faces.name, 
                           self.domain.faces.halofid, self.domain.cells.center, self.domain.faces.center)
        
        
        self.comm.Barrier()
        #update the halo values
        Variable._define_halosend(self.gradcellx, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.gradhalocellx, 
                   self.comm, self.mpi_precision)
        
        #update the halo values
        Variable._define_halosend(self.gradcelly, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.gradhalocelly,
                   self.comm, self.mpi_precision)
        
        #update the halo values
        Variable._define_halosend(self.gradcellz, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.gradhalocellz,
                   self.comm, self.mpi_precision)
        
        #update the halo values
        Variable._define_halosend(self.psi, self.halotosend, self.domain.halos.indsend)
        all_to_all(self.halotosend, self.nbhalos, self.domain.halos.scount, self.domain.halos.rcount, self.psihalo,
                   self.comm, self.mpi_precision)
        
    def compute_face_gradient(self):
        
        Variable._face_gradient(self.cell, self.ghost, self.halo, self.node, self.domain.faces.cellid, 
                            self.domain.faces.nodeid, self.domain.faces.ghostcenter, 
                            self.domain.faces.halofid, self.domain.cells.center,
                            self.domain.halos.centvol, self.domain.nodes.vertex, self.domain.faces.airDiamond,
                            self.domain.faces.normal, self.domain.faces.f_1, self.domain.faces.f_2, 
                            self.domain.faces.f_3, self.domain.faces.f_4, self.domain.cells.shift, 
                            self.gradfacex, self.gradfacey, self.gradfacez, self.domain._innerfaces,
                            self.domain.halofaces, self.dirichletfaces, self.neumannfaces, 
                            self.domain.periodicboundaryfaces)
        
    def update_ghost_value(self):
        for BC in self._BCs.values():
            BC._func_ghost(BC.BCvalueface, self.ghost, self.domain.faces.cellid, np.asarray(BC.BCfaces, dtype=np.uint32),
                           BC.constNH, self.domain.faces.dist_ortho)
            BC._func_haloghost(BC.BCvaluehalo, self.haloghost, self.domain.nodes.haloghostcenter, 
                               BC.BCtypeindex,  self.domain.halonodes, BC.constNHNode) 
            
            
    def norml2(self, exact, order=None):
        
        if order is None:
            order = 1
        assert self.nbcells == len(exact), 'exact solution must have length of cells'
        
        Error = np.zeros(self.nbcells, dtype=self.float_precision)
        Ex = np.zeros(self.nbcells, dtype=self.float_precision)
       
        for i in range(len(exact)):
            Error[i] = np.fabs(self.cell[i] - exact[i]) * self.domain.cells.volume[i]
            Ex[i] = np.fabs(exact[i]) * self.domain.cells.volume[i]
    
        ErrorL2 = np.linalg.norm(Error,ord=order)/np.linalg.norm(Ex,ord=order)
        
        return ErrorL2
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def comm(self):
        return self._comm
    
    @property
    def nbfaces(self):
        return self._nbfaces
    
    @property
    def nbcells(self):
        return self._nbcells
    
    @property
    def nbnodes(self):
        return self._nbnodes
    
    @property
    def nbhalos(self):
        return self._nbhalos
    
    @property
    def name(self):
        return self._name
    
    @property
    def BCs(self):
        return self._BCs
    
    @property
    def BCin(self):
        return self._BCin
    
    @property
    def BCout(self):
        return self._BCout
    
    @property
    def BCupper(self):
        return self._BCupper
    
    @property
    def BCbottom(self):
        return self._BCbottom
    
    @property
    def BCback(self):
        return self._BCback
    
    @property
    def BCfront(self):
        return self._BCfront
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:54:55 2022

@author: kissami
"""
from manapy.ast.ast_utils import (convert_solution)
from manapy.boundary.bc_utils import (rhs_value_dirichlet_node, rhs_value_dirichlet_face)

from manapy.ast.functions2d import (get_triplet_2d, compute_2dmatrix_size, compute_P_gradient_2d_diamond, 
                                    get_rhs_glob_2d, get_rhs_loc_2d, compute_P_gradient_2d_FV4)

from manapy.ast.functions3d import ( get_triplet_3d, compute_3dmatrix_size, compute_P_gradient_3d_diamond,
                                    get_rhs_glob_3d, get_rhs_loc_3d)

from manapy.base.base import get_default, make_get_conf, try_imports

from manapy.ast import Variable
import numpy as np
from mpi4py import MPI
from manapy.base.base import Struct
import six

def standard_call(call):
    """                                                                                                                                                                                                    
    Decorator handling argument preparation for linear solvers.                                                                                                                                 
    """
    def _standard_call(self, rhs=None, conf=None, **kwargs):

        conf = get_default(conf, self.conf)
        result = call(self, rhs, conf, **kwargs)
    	
        return result

    return _standard_call

class LinearSolver():
    
    _parameters = [('scheme', 'str', 'diamond', 'fv4',
                    'scheme diamond or fv4.'),
                    ('reordering', 'bool', False, False,
                    'reordering the matrix only in serial case.')
    ]
             
    
    @classmethod
    def process_conf(cls, conf, kwargs):
        """                                                                                                                                                                                                
        Process configuration parameters.                                                                                                                                                                  
        """
        get = make_get_conf(conf, kwargs)
        
        if len(cls._parameters) and cls._parameters[0][0] != 'name':
            options = LinearSolver._parameters + cls._parameters

        else:
            options = LinearSolver._parameters
            
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
    def __init__(self, domain, var, conf, comm=None, **kwargs):
        
        if comm is None:
            comm = MPI.COMM_WORLD
        
        if comm.Get_rank() == 0:
            print("SetUp the Linear system ...")
            
        if conf is None:
            conf = Struct()
            
        new_conf = self.process_conf(conf, kwargs)
        
        self.conf = new_conf
        get = make_get_conf(self.conf, kwargs)
        
        scheme = get("scheme")
        self.verbose = get("verbose")
        
        self.var    = var
        self.domain = domain
        self.dim    = self.domain.dim
        
        #Backend
        self.backend = self.domain.backend
        self.signature = self.domain.signature
        self.float_precision = self.domain.float_precision
        self.mpi_precision = self.domain.mpi_precision
        
        self.localsize  = self.domain.nbcells
        self.globalsize = self.comm.allreduce(self.localsize, op=MPI.SUM)
        self.domain.globalsize = self.globalsize
        
        
        self.sendcounts1 = np.array(self.comm.gather(self.localsize, root=0), dtype=self.float_precision)
        self.x1converted = np.zeros(self.globalsize, dtype=self.float_precision)
        
        self.domain.Pbordnode = np.zeros(self.domain.nbnodes, dtype=self.float_precision)
        self.domain.Pbordface = np.zeros(self.domain.nbfaces, dtype=self.float_precision)
        
        self.domain.Ibordnode = np.zeros(self.domain.nbnodes, dtype=self.float_precision)
        self.domain.Ibordface = np.zeros(self.domain.nbfaces, dtype=self.float_precision)
        
        matrixinnerfaces = np.concatenate([self.domain._innerfaces, self.domain._periodicinfaces, self.domain._periodicupperfaces])
        if self.dim == 3:
            matrixinnerfaces = np.concatenate([matrixinnerfaces, self.domain._periodicfrontfaces])
        self.matrixinnerfaces = np.sort(matrixinnerfaces)

        if scheme == "fv4":
            self._compute_P_gradient = compute_P_gradient_2d_FV4
            sizeM = 4*len(matrixinnerfaces)+len(self.var.dirichletfaces) + 2*len(self.domain.halofaces)
            self._row  = np.zeros(sizeM, dtype=np.int32)
            self._col  = np.zeros(sizeM, dtype=np.int32)
            self._data = np.zeros(sizeM, dtype=self.float_precision)
       
        elif scheme == "diamond":     
            if self.dim == 2:
                self._compute_P_gradient = compute_P_gradient_2d_diamond
                self._get_triplet  = get_triplet_2d
                self.dataSize =  self.backend.compile(compute_2dmatrix_size, signature=self.signature)(self.domain.faces.nodeid, 
                                                      self.domain.faces.halofid,  self.domain.nodes.cellid, 
                                                      self.domain.nodes.halonid, self.domain.nodes.periodicid,
                                                      self.domain.nodes.ghostcenter, self.domain.nodes.haloghostcenter, self.domain.nodes.oldname,
                                                      self.var.BCdirichlet, self.matrixinnerfaces, self.domain.halofaces, 
                                                      self.var.dirichletfaces)
            elif self.dim == 3:
                self._compute_P_gradient = compute_P_gradient_3d_diamond
                self._get_triplet  = get_triplet_3d
                self.dataSize =  self.backend.compile(compute_3dmatrix_size, signature=self.signature)(self.domain.faces.nodeid, 
                                                      self.domain.faces.halofid, self.domain.nodes.cellid, 
                                                      self.domain.nodes.halonid, self.domain.nodes.periodicid,
                                                      self.domain.nodes.ghostcenter, self.domain.nodes.haloghostcenter, self.domain.nodes.oldname,
                                                      self.var.BCdirichlet, self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
            
            
            self._row = np.zeros(self.dataSize, dtype=np.int32)
            self._col = np.zeros(self.dataSize, dtype=np.int32)
            self._data = np.zeros(self.dataSize, dtype=self.float_precision)
            
            self.convert_solution = self.backend.compile(convert_solution, signature=self.signature)
    
    def assembly(self):
        self._get_triplet(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.vertex, self.domain.faces.halofid, 
                          self.domain.halos.halosext, self.domain.nodes.oldname, self.domain.cells.volume, 
                          self.domain.nodes.cellid, self.domain.cells.center, self.domain.halos.centvol, self.domain.nodes.halonid, self.domain.nodes.periodicid,
                          self.domain.nodes.ghostcenter, self.domain.nodes.haloghostcenter, self.domain.faces.airDiamond, 
                          self.domain.nodes.lambda_x, self.domain.nodes.lambda_y, self.domain.nodes.lambda_z, self.domain.nodes.number, self.domain.nodes.R_x, 
                          self.domain.nodes.R_y,  self.domain.nodes.R_z, self.domain.faces.param1, self.domain.faces.param2, self.domain.faces.param3, 
                          self.domain.faces.param4, self.domain.cells.shift, self.localsize, self.domain.cells.loctoglob, self.var.BCdirichlet, self._data, 
                          self._row, self._col, self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
    
        self._get_rhs(self.domain.faces.cellid, self.domain.faces.nodeid, self.domain.nodes.oldname,
                      self.domain.cells.volume, self.domain.nodes.ghostcenter, self.domain.cells.loctoglob, 
                      self.domain.faces.param1, self.domain.faces.param2, self.domain.faces.param3, 
                      self.domain.faces.param4, self.domain.Pbordnode, self.domain.Pbordface, 
                      self.rhs0, self.var.BCdirichlet, self.domain.faces.ghostcenter, 
                      self.matrixinnerfaces, self.domain.halofaces, self.var.dirichletfaces)
        
    def update_ghost_values(self):
        
         for BC in self.var.BCs.values():
            if BC._BCtype == "dirichlet":
                rhs_value_dirichlet_face(self.domain.Pbordface, np.asarray(BC.BCfaces, dtype=np.int32), BC.BCvalueface)
                rhs_value_dirichlet_node(self.domain.Pbordnode, np.where(self.domain.nodes.oldname==BC.BCtypeindex)[0].astype(np.int32),
                                         BC.BCvaluenode)
                
            elif BC._BCtype == "neumann":
                for i in np.where(self.domain.nodes.oldname==BC.BCtypeindex)[0]:
                    self.domain.Pbordnode[i] = 1.
            
    def compute_Sol_gradient(self):
        self._compute_P_gradient(self.var.cell, self.var.ghost, self.var.halo, self.var.node, self.domain.faces.cellid, 
                                 self.domain.faces.nodeid, self.domain.faces.ghostcenter,
                                 self.domain.faces.halofid, self.domain.cells.center,
                                 self.domain.halos.centvol, self.domain.nodes.oldname, self.domain.faces.airDiamond,
                                 self.domain.faces.f_1, self.domain.faces.f_2,  self.domain.faces.f_3, self.domain.faces.f_4, 
                                 self.domain.faces.normal, self.domain.cells.shift, self.domain.Pbordnode, self.domain.Pbordface, 
                                 self.var.gradfacex, self.var.gradfacey, self.var.gradfacez, self.var.BCdirichlet, 
                                 self.domain.innerfaces, self.domain.halofaces, self.var.neumannfaces, 
                                 self.var.dirichletfaces, self.domain.periodicboundaryfaces)

    def reordering_matrix(self):
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        from scipy.sparse import csr_matrix
        matrix = csr_matrix((self._data, (self._row, self._col)))
        # Compute the reverse Cuthill-Mckee ordering
        self.perm = reverse_cuthill_mckee(matrix, symmetric_mode=False)
        matrix = matrix[:, self.perm][self.perm, :]
        ## Convert the reordered matrix back to AIJ format
        self._row, self._col = matrix.nonzero()
        self._data = matrix.data
        self.rhs0 = self.rhs0[self.perm]
        
    def build_solver_kwargs(self, conf):
        """                                                                                                                                                                                                
        Build the `kwargs` dict for the underlying solver function using the                                                                                                                               
        extra options (marked by '*' in ``_parameters``) in `conf`. The                                                                                                                                    
        declared parameters are omitted.                                                                                                                                                                   
        """
        if len(self._parameters) and self._parameters[0][0] != 'name':
            options = LinearSolver._parameters + self._parameters

        else:
            options = self._parameters

        std = set([ii[0] for ii in options if ii[0] != '*'])
        
        kwargs = {}
        for key, val in conf.to_dict().items():
            if key not in std:
                kwargs[key] = val
        
        return kwargs
        
            
class MUMPSSolver(LinearSolver):
    
    _parameters = [
        ('reuse_mtx', 'bool', True, False,
         'If True, pre-factorize the matrix.'),
        ('with_mtx', 'bool', False, True,
         'If True, the matrix should be given.'),
        ('memory_relaxation', 'int', 20, False,
         'The percentage increase in the estimated working space.'),
         ('verbose', 'bool', False, False,
         """If True, the solver can print more information about the                                                                                                                                       
            solution.""")
    ]
    
    def __init__(self, domain=None, var=None, comm=None, conf=None, **kwargs):
        
        import manapy.solvers.ls.ls_mumps as mumps
        
        if not mumps.use_mpi:
            raise AttributeError('No mpi4py found! Required by MUMPS solver.')
        mumps.load_mumps_libraries()
        
        self.mumps = mumps
        self.mumps_ls = None
        
        if comm is None:
            self.comm = MPI.COMM_WORLD
        if domain is None:
            raise ValueError("domain must be given")
        if var is None:
            raise ValueError("variable must be given")
        assert(isinstance(var, Variable)), "Variable must be a Variable type"
        
        LinearSolver.__init__(self, domain=domain, var=var, conf=conf, **kwargs)
        
        self._domain = domain
        self._dim = self._domain.dim
        self.var = var
        self._domain.solver = "mumps"

        self.rhs0 = np.zeros(self.globalsize, dtype=self.float_precision)
        
        if self._dim == 2:
            self._get_rhs_glob = get_rhs_glob_2d
        elif self._dim == 3:
            self._get_rhs_glob = get_rhs_glob_3d

        #compile functions
        self._get_rhs = self.backend.compile(self._get_rhs_glob, signature=self.signature)
        self._get_triplet = self.backend.compile(self._get_triplet, signature=self.signature)
        self._compute_P_gradient = self.backend.compile(self._compute_P_gradient, signature=self.signature)
    
    @standard_call
    def __call__(self, rhs=None, conf=None, **kwargs):
        
        if not conf.reuse_mtx:
            self.clear()
            
        import timeit
        ts = timeit.default_timer()
            
        self.presolve(reuse_mtx=self.conf.reuse_mtx, with_mtx=self.conf.with_mtx)
        
        if rhs is not None:
            rhs += self.rhs00
        else:
            rhs= self.rhs00
        
        #Allocation size of rhs
        if self.comm.Get_rank() == 0:
            self.sol = rhs.copy().astype(np.double)
            self.mumps_ls.set_rhs(self.sol)
        
        

        #Solution Phase
        self.mumps_ls._mumps_call(job=3)
        
        print("solving time", timeit.default_timer() - ts)
        
        if self.conf.reordering and self.comm.Get_size() == 1:
            self.sol = self.sol[np.argsort(self.perm)]
        
        if self.comm.Get_rank() == 0:
            #Convert solution for scattering
            self.convert_solution(self.sol.astype(self.float_precision), self.x1converted, self.domain.cells.tc, self.globalsize)
        self.comm.Scatterv([self.x1converted, self.sendcounts1, self.mpi_precision], self.var.cell, root = 0)
        
    def presolve(self,reuse_mtx=False, with_mtx=False):
        if not reuse_mtx or (self.mumps_ls is None):
            self.update_ghost_values()
            #assembly row, col , data, rhs(bc)
            if not with_mtx:
                self.assembly()
            if self.conf.reordering  and self.comm.Get_size() == 1:
                self.reordering_matrix()
            
#            print( self._data.shape[0], self._row.shape[0], self._col.shape[0], with_mtx)
            self.rhs00 = self.comm.reduce(self.rhs0, op=MPI.SUM, root=0)

            if self.mumps_ls is None:
                system = 'real'
                mem_relax = self.conf.memory_relaxation
                self.mumps_ls = self.mumps.MumpsSolver(system=system, #is_sym=1,
                                                       mem_relax=mem_relax,)
            if self.conf.verbose:
                self.mumps_ls.set_verbose()
            
#            print(self._row.shape[0], self._col.shape[0], self._data.shape[0] )
            self.mumps_ls.set_rcd_distributed(self._row+1, self._col+1, self._data.astype(np.float64))
            self.mumps_ls.set_icntl(18,3)
            #self.ctx.set_icntl(28,2)
            #self.ctx.set_icntl(16,2)
            
            if self.comm.Get_rank() == 0:
                self.mumps_ls.struct.n = self.globalsize
                self.sol = self.rhs00.copy()
                
            #Analyse 
            self.mumps_ls._mumps_call(job=1)
            #Factorization Phase
            self.mumps_ls._mumps_call(job=2)
            
            #Allocation size of rhs
            if self.comm.Get_rank() == 0:
                self.mumps_ls.set_rhs(self.sol)
            else :
                self.sol = np.zeros(self.globalsize, dtype=self.float_precision)                                                                                                                                                    
    
    def clear(self):
        if self.mumps_ls is not None:
            del(self.mumps_ls)
        self.mumps_ls = None
        
    def __del__(self):
        self.clear()
        
        
class PETScKrylovSolver(LinearSolver):
    
    _parameters = [
        ('reuse_mtx', 'bool', True, False,
         'If True, pre-factorize the matrix.'),
        ('with_mtx', 'bool', False, True,
         'If True, the matrix should be given.'),
        ('method', 'str', 'fgmres', False,
         'The actual ksp solver to use.'),
        ('precond', 'str', 'gamg', False,
         'The preconditioner.'),
        ('sub_precond', 'str', 'none', False,
         'The preconditioner for matrix blocks (in parallel runs).'),
        ('factor_solver', 'str', 'none', False,
         'Use Factor solver type such as mumps, superlu_dist'),
        ('i_max', 'int', 1000, False,
         'The maximum number of iterations.'),
        ('eps_a', 'float', 1e-6, False,
         'The absolute tolerance for the residual.'),
        ('eps_r', 'float', 1e-12, False,
         'The relative tolerance for the residual.'),
        ('eps_d', 'float', 1e5, False,
         'The divergence tolerance for the residual.'),
        ('*', '*', None, False,
         """Additional parameters supported by the method. Can be used to pass                                                                                                                             
            all PETSc options supported by :func:`petsc.Options()`."""),

    ]
    
    def __init__(self, domain=None, var=None, comm=None, conf=None, **kwargs):
        
        try_imports(['import petsc4py',],
                    'cannot import petsc4py solver!')
        
        from petsc4py import PETSc as petsc
#        import petsc4py
#        import sys
#        petsc4py.init(sys.argv)
        
        self.petsc = petsc
        self.ksp   = None
        
        self.converged_reasons = {}                                                                                                                                                                             
        for key, val in six.iteritems(petsc.KSP.ConvergedReason.__dict__):                                                                                                                                 
            if isinstance(val, int):                                                                                                                                                                       
                self.converged_reasons[val] = key               
        
        if comm is None:
            self.comm = MPI.COMM_WORLD
        if domain is None:
            raise ValueError("domain must be given")
        if var is None:
            raise ValueError("variable must be given")
        assert(isinstance(var, Variable)), "Variable must be a Variable type"
        
        LinearSolver.__init__(self, domain=domain, var=var, conf=conf, **kwargs)
        
        self._domain = domain
        self._dim = self._domain.dim
        self.var = var
        self._domain.solver = "petsc"
        
        self.rhs0 = np.zeros(self.globalsize, dtype=self.float_precision)
        
        if self._dim == 2:
            self._get_rhs_loc = get_rhs_loc_2d
        elif self._dim == 3:
            self._get_rhs_loc = get_rhs_loc_3d

        #compile functions
        self._get_rhs = self.backend.compile(self._get_rhs_loc, signature=self.signature)
        self._get_triplet = self.backend.compile(self._get_triplet, signature=self.signature)
        self._compute_P_gradient = self.backend.compile(self._compute_P_gradient, signature=self.signature)
        
    @standard_call
    def __call__(self, rhs=None, conf=None, **kwargs):
        
        def custom_monitor(ksp, its, r_norm):
            print(f"Iteration {its}: Residual Norm = {r_norm}")
        
        if not conf.reuse_mtx:
            self.clear()
        
        self.create_petsc_matrix(reuse_mtx=self.conf.reuse_mtx, with_mtx=self.conf.with_mtx)
        
        if rhs is not None:
            self.update_rhs(rhs=rhs)

        self.ksp.solve(self.rhs, self.sol)
        
        if self.conf.verbose:
            print(self.ksp.getType(), self.ksp.getPC().getType(), self.conf.sub_precond,
                  self.ksp.reason, self.converged_reasons[self.ksp.reason],
                  self.ksp.getIterationNumber())
        
            for i in range(self.ksp.getIterationNumber()):
                r_norm = self.ksp.getResidualNorm()
                custom_monitor(self.ksp, i+1, r_norm)
        
        if self.conf.reordering and self.comm.Get_size() == 1:
            self.sol.array = self.sol.array[np.argsort(self.perm)]
        
        self.comm.Gatherv(sendbuf=self.sol.array.astype(self.float_precision), recvbuf=(self.recvbuf, self.sendcounts2), root=0)
        
        if self.comm.Get_rank() == 0:
            #Convert solution for scattering
            convert_solution(self.recvbuf, self.x1converted, self.domain.cells.tc, self.globalsize)
            
        self.comm.Scatterv([self.x1converted, self.sendcounts1, self.mpi_precision], self.var.cell, root = 0)
        
    def create_ksp(self, options=None, comm=None):
        optDB = self.petsc.Options()
        
        optDB['sub_pc_type'] = self.conf.sub_precond
        if options is not None:
            for key, val in six.iteritems(options):
                optDB[key] = val
        
        self.ksp = self.petsc.KSP()
        self.ksp.create(comm)

        self.ksp.setType(self.conf.method)
        pc = self.ksp.getPC()
        
        if self.conf.precond == "lu" and self.conf.factor_solver == 'none':
            self.conf.factor_solver = "mumps"
            optDB['pc_factor_mat_ordering_type'] = "rcm"
        
        pc.setType(self.conf.precond)
        if self.conf.factor_solver is not None:
            pc.setFactorSolverType(self.conf.factor_solver)
        
        self.ksp.setFromOptions()
    
    def update_rhs(self, rhs=None):
        self.rhs = self.mat.getVecLeft()
        for i in range(self.domain.nbcells):
                self.rhs.setValues(self.domain.cells.loctoglob[i], self.rhs0[i]+rhs[i])
        
        self.rhs.assemblyBegin()
        self.rhs.assemblyEnd()
        
    def create_rhs(self):
        
        self.rhs = self.mat.getVecLeft()
        for i in range(self.domain.nbcells):
            self.rhs.setValues(self.domain.cells.loctoglob[i], self.rhs0[i])
                
        self.rhs.assemblyBegin()
        self.rhs.assemblyEnd()
    
    def Initiate_sol(self):
        self.ksp.setInitialGuessNonzero(True)
    
    def create_petsc_matrix(self, reuse_mtx=False, with_mtx=False):
        if not reuse_mtx or (self.ksp is None):
            ###################################################################
            self.update_ghost_values()
            #assembly row, col , data, rhs(bc)
            if not with_mtx:
                self.assembly()
           
            ###################################################################
            #reordering matrix 
            if self.conf.reordering  and self.comm.Get_size() == 1:
                self.reordering_matrix()
            
            ###################################################################
            #non zero values for each rows
            NNZ_loc = np.zeros(self.globalsize, dtype=np.int32)
            unique, counts = np.unique(np.asarray(self._row, dtype=np.int32), return_counts=True)
            
            for i in range(self.domain.nbcells):
                NNZ_loc[self.domain.cells.loctoglob[i]] = counts[i]
            
            self.NNZ = np.zeros(self.globalsize, dtype=np.int32)
            self.comm.Allreduce(NNZ_loc, self.NNZ, op=MPI.SUM)#, root=0)
            
            #Create the petsc matrix 
            ###################################################################
            self.mat = self.petsc.Mat().create()
            self.mat.setSizes(self.globalsize)
            self.mat.setType("mpiaij")
            self.mat.setFromOptions()
            self.mat.setPreallocationNNZ(self.NNZ)
            self.mat.setOption(option=19, flag=0)
            
            
            for i in range(len(self._row)):
                self.mat.setValues(self._row[i], self._col[i], self._data[i], addv=True)
            
            self.mat.assemblyBegin(self.mat.AssemblyType.FINAL)
            self. mat.assemblyEnd(self.mat.AssemblyType.FINAL)
            
            ###################################################################
            self.sol = self.mat.getVecRight()
            self.sendcounts2 = np.array(self.comm.gather(len(self.sol.array),  root=0))
            
            if self.comm.Get_rank() == 0:
                self.recvbuf = np.empty(sum(self.sendcounts2), dtype=self.float_precision)
            else:
                self.recvbuf = None
            ###################################################################
            # Create the ksp solver
            ###################################################################
            solver_kwargs = self.build_solver_kwargs(self.conf)
            
            
            self.create_ksp(options=solver_kwargs, comm=self.comm)
            self.ksp.setOperators(self.mat)
            self.ksp.setTolerances(atol=self.conf.eps_a, rtol=self.conf.eps_r, divtol=self.conf.eps_d,
                                   max_it=self.conf.i_max)
            self.ksp.setFromOptions()
            ###################################################################
            
#            #Create the solution
#            self.Initiate_sol()
           
            #Create the Rhs
            self.create_rhs()
            
    def view(self):
        self.mat.view()
        self.rhs.view()
        self.sol.view()
    
    def clear(self):
        if self.ksp is not None:
            del(self.ksp)
        self.ksp = None
        
    def __del__(self):
        self.clear()

class ScipySolver(LinearSolver):
    
    _parameters = [
        ('reuse_mtx', 'bool', True, False,
         'If True, pre-factorize the matrix.'),
        ('memory_relaxation', 'int', 20, False,
         'The percentage increase in the estimated working space.'),
         ('verbose', 'bool', False, False,
         """If True, the solver can print more information about the                                                                                                                                       
            solution.""")
    ]
    
    def __init__(self, domain=None, var=None, method=None, comm=None, conf=None, **kwargs):
        
        aux = try_imports(['import scipy.sparse.linalg as sls',
                           'import scipy.sparse.linalg.dsolve as sls'],
                           'cannot import scipy sparse direct solvers!')
        
        if comm is None:
            self.comm = MPI.COMM_WORLD
        
        if self.comm.Get_size() > 1:
            raise ValueError('ScipySolver is not parallel! use MUMPSSolver or PETScKrylovSolver')
        self.solve = None  
        
        if 'sls' in aux:
            self.sls = aux['sls']
        else:
            raise ValueError('SuperLU not available!')
        
        if method is None:
            method = "superlu"
        
        if method in ['auto', 'umfpack']:
            aux = try_imports(['import scikits.umfpack as um'])

            is_umfpack = True if 'um' in aux\
            and hasattr(aux['um'], 'UMFPACK_OK') else False
            if method == 'umfpack' and not is_umfpack:
                raise ValueError('UMFPACK not available!')
        elif method == 'superlu':
            is_umfpack = False
        else:
            raise ValueError('uknown solution method! (%s)' % method)

        if is_umfpack:
            self.sls.use_solver(useUmfpack=True,
                                assumeSortedIndices=True)
        else:
            self.sls.use_solver(useUmfpack=False)

        self.clear()
        
        if domain is None:
            raise ValueError("domain must be given")
        if var is None:
            raise ValueError("variable must be given")
        assert(isinstance(var, Variable)), "Variable must be a Variable type"
        
        LinearSolver.__init__(self, domain=domain, var=var, conf=conf, **kwargs)
        
        self._domain = domain
        self._dim = self._domain.dim
        self.var = var
        self._domain.solver = "scipy"

        self.rhs0 = np.zeros(self.globalsize, dtype=self.float_precision)
        
        if self._dim == 2:
            self._get_rhs_glob = get_rhs_glob_2d
        elif self._dim == 3:
            self._get_rhs_glob = get_rhs_glob_3d

        #compile functions
        self._get_rhs = self.backend.compile(self._get_rhs_glob, signature=self.signature)
        self._get_triplet = self.backend.compile(self._get_triplet, signature=self.signature)
        self._compute_P_gradient = self.backend.compile(self._compute_P_gradient, signature=self.signature)
    
    @standard_call
    def __call__(self, rhs=None, conf=None, **kwargs):
        
        if not conf.reuse_mtx:
            self.clear()
            
        self.presolve(reuse_mtx=self.conf.reuse_mtx)
        
        if rhs is not None:
            rhs += self.rhs0
        else:
            rhs = self.rhs0
        
        if self.conf.reuse_mtx:
            self.var.cell = self.solve(rhs)
        else:
            self.var.cell = self.sls.spsolve(self.mat, rhs)
            
        if self.conf.reordering and self.comm.Get_size() == 1:
            self.var.cell = self.var.cell[np.argsort(self.perm)]
            
    def presolve(self,reuse_mtx=False):
        
        if not reuse_mtx or (self.solve is None) :
            from scipy import sparse
            self.update_ghost_values()
            
            #assembly row, col , data, rhs(bc)
            self.assembly()
            if self.conf.reordering  and self.comm.Get_size() == 1:
                self.reordering_matrix()
            
            self.mat = sparse.csc_matrix((self._data, (self._row, self._col)))
            self.solve = self.sls.factorized(self.mat)        
            
            
    def clear(self):                                                                                                                                                                                       
        if self.solve is not None:                                                                                                                                                                         
            del self.solve    
            self.destroy()                                                                                                                                                                             
                                                                                                                                                                                                           
        self.solve = None  
        

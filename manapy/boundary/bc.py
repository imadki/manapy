from manapy.boundary.bc_utils import (ghost_value_neumann, ghost_value_dirichlet,
#                                   ghost_value_slip, haloghost_value_slip
                                   ghost_value_nonslip, haloghost_value_nonslip,
                                   ghost_value_neumannNH, haloghost_value_neumannNH,
                                   haloghost_value_dirichlet, haloghost_value_neumann)
from numpy import zeros

class Boundary():
    
    is_called = False

    """ """
    def __init__(self, BCtype=None, BCvalueface = None, BCvaluenode = None, BCvaluehalo = None, 
                 BCloc = None, BCtypeindex = None, domain=None):
        
        if domain is None:
            raise ValueError("domain must be given")
            
        self._BCtype   = BCtype
        self.BCvalueface  = BCvalueface
        self.BCvaluenode  = BCvaluenode
        self.BCvaluehalo  = BCvaluehalo
        self._domain   = domain
        
        self.backend = self.domain.backend
        self.signature = self.domain.signature
        
        self._func_ghost_args = []
        self._func_haloghost_args = []
        
        self.constNH = zeros(1)
        self.constNHNode = zeros(1)
        
        if BCloc == "in":
            self._BCfaces  = self.domain._infaces
            self._BCnodes  = self.domain._innodes
            self._BCtypeindex  = BCtypeindex
        elif BCloc == "out":
            self._BCfaces  = self.domain._outfaces
            self._BCnodes  = self.domain._outnodes
            self._BCtypeindex  = BCtypeindex
        elif BCloc == "bottom":
            self._BCfaces  = self.domain._bottomfaces
            self._BCnodes  = self.domain._bottomnodes
            self._BCtypeindex  = BCtypeindex
        elif BCloc == "upper":
            self._BCfaces  = self.domain._upperfaces
            self._BCnodes  = self.domain._uppernodes
            self._BCtypeindex  = BCtypeindex
        elif BCloc == "front":
            self._BCfaces  = self.domain._frontfaces
            self._BCnodes  = self.domain._frontnodes
            self._BCtypeindex  = BCtypeindex
        elif BCloc == "back":
            self._BCfaces  = self.domain._backfaces
            self._BCnodes  = self.domain._backnodes
            self._BCtypeindex  = BCtypeindex
        
        if self._BCtype == "neumann" or self._BCtype == "periodic":
            self._func_ghost = ghost_value_neumann
            self._func_haloghost = haloghost_value_neumann
        elif self._BCtype == "dirichlet":
            self._func_ghost = ghost_value_dirichlet
            self._func_haloghost = haloghost_value_dirichlet
        elif self._BCtype == "neumannNH":
           self._func_ghost = ghost_value_neumannNH
           self._func_haloghost = haloghost_value_neumannNH
#        elif self._BCtype == "slip":
#            self._func_ghost = ghost_value_slip
#            self._func_haloghost = haloghost_value_slip
#            self._func_ghost_args.extend([self._BCvaluefacetmp, self.domain.faces.normal, self.domain.faces.mesure])
#            self._func_haloghost_args.extend([self._BCvaluehalotmp, self.domain.nodes.ghostfaceinfo])
        
        elif self._BCtype == "nonslip":
            self._func_ghost = ghost_value_nonslip
            self._func_haloghost = haloghost_value_nonslip
        
        Boundary.compile_func(self._func_ghost, self._func_haloghost, self.backend, self.signature )
#        self._func_ghost  = self.backend.compile(self._func_ghost)
#        self._func_haloghost = self.backend.compile(self._func_haloghost)
    
    @classmethod
    def compile_func(cls, _func_ghost, _func_haloghost, backend, signature):
        if not cls.is_called:
            Boundary._func_ghost  = backend.compile(_func_ghost, signature=signature)
            Boundary._func_haloghost = backend.compile(_func_haloghost, signature=signature)
            cls.is_called = True
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def BCfaces(self):
        return self._BCfaces
    
    @property
    def BCnodes(self):
        return self._BCnodes
    
    @property
    def BCtypeindex(self):
        return self._BCtypeindex
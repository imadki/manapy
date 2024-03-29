#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:05:46 2023

@author: kissami
"""
from numpy import zeros
from mpi4py import MPI
import numpy as np

from manapy.solvers.advec import (explicitscheme_convective_2d, 
                         	   explicitscheme_convective_3d,
                                  time_step, update_new_value)
from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.base.base import make_get_conf


class AdvectionSolver():
    
    _parameters = [('dt', float, 0., 0.,
                    'time step'),
                    ('order', int, 1, 1,
                     'order of the convective scheme'),
                    ('cfl', float, .4, 0,
                     'cfl of the explicit scheme')
    ]
    
    @classmethod
    def process_conf(cls, conf, kwargs):
        """                                                                                                                                                                                                
        Process configuration parameters.                                                                                                                                                                  
        """
        get = make_get_conf(conf, kwargs)
        
        if len(cls._parameters) and cls._parameters[0][0] != 'name':
            options = AdvectionSolver._parameters + cls._parameters

        else:
            options = AdvectionSolver._parameters
            
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
    
    def __init__(self, var=None, vel=(None, None, None), conf=None, **kwargs):
        
        if conf is None:
            conf = Struct()
            
        new_conf = self.process_conf(conf, kwargs)
        self.conf = new_conf
        get = make_get_conf(self.conf, kwargs)
        
        if not isinstance(var, Variable):
            raise ValueError("primal var must be a Variable type")
        
        if not isinstance(vel[0], Variable):
            raise ValueError("u must be a Variable type")
        
        if not isinstance(vel[1], Variable):
            raise ValueError("v must be a Variable type")
        
        self.var = var
        self.comm = self.var.comm
        self.domain = self.var.domain
        self.dim = self.var.dim
        self.float_precision = self.domain.float_precision
        
        self.u   = vel[0]
        self.v   = vel[1]
        
        if len(vel) == 3:
            if not isinstance(vel[2], Variable):
                raise ValueError("w must be a Variable type")
            self.w = vel[2]
        else:
            self.w = Variable(domain=self.domain)
        
        self.dt    = np.float64(get("dt"))
        self.order = np.int32(get("order"))
        self.cfl   = np.float64(get("cfl"))
        
        self.backend = self.domain.backend
        self.signature = self.domain.signature
        
        self.var.__dict__["convective"] = zeros(self.domain.nbcells, dtype=self.float_precision)
        self.var.__dict__["dissipative"] = zeros(self.domain.nbcells, dtype=self.float_precision)
        self.var.__dict__["source"] = zeros(self.domain.nbcells, dtype=self.float_precision)
        
        if self.dim == 2:
            self._explicitscheme_convective = explicitscheme_convective_2d
        elif self.dim == 3:
            self._explicitscheme_convective = explicitscheme_convective_3d
        
        self._explicitscheme_convective  = self.backend.compile(self._explicitscheme_convective, signature=self.signature)
        self._time_step  = self.backend.compile(time_step, signature=self.signature)
        self._update_new_value = self.backend.compile(update_new_value, signature=self.signature)
            
    
    def explicit_convective(self):
        if self.order == 2:
            self.var.compute_cell_gradient()
        self._explicitscheme_convective(self.var.convective, self.var.cell, self.var.ghost, self.var.halo, self.u.face, self.v.face, self.w.face,
                                        self.var.gradcellx, self.var.gradcelly, self.var.gradcellz, self.var.gradhalocellx, 
                                        self.var.gradhalocelly, self.var.gradhalocellz, self.var.psi, self.var.psihalo, 
                                        self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol, 
                                        self.domain.faces.ghostcenter, self.domain.faces.cellid, self.domain.faces.normal, 
                                        self.domain.faces.halofid, self.domain.faces.name, 
                                        self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, 
                                        self.domain.periodicboundaryfaces, self.domain.cells.shift, order=self.order)
    
    
    def stepper(self):
        d_t = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                             self.domain.cells.volume, self.domain.cells.faceid,  self.dim)
        self.dt = self.comm.allreduce(d_t, op=MPI.MIN)
        return  self.dt
    
    
    def compute_fluxes(self):
        
        #interpolate cell to node
        self.var.update_halo_value()
        self.var.update_ghost_value() 
        
        #convective flux
        self.explicit_convective()
        
        
    def compute_new_val(self):
        self._update_new_value(self.var.cell, self.var.convective, self.var.dissipative, self.var.source, self.dt, self.domain.cells.volume)         
        
        






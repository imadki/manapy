#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:05:46 2023

@author: kissami
"""
from numpy import zeros, zeros_like
from mpi4py import MPI

from manapy.solvers.streamer import (update_ST, explicitscheme_source_ST, 
                                     time_step_ST, explicitscheme_dissipative_ST,
                                     compute_el_field, compute_velocity,
                                     update_rhs_glob, update_rhs_loc)

from manapy.solvers.advecdiff import (explicitscheme_convective_2d,
                                      explicitscheme_convective_3d)

from manapy.comms import Iall_to_all
from manapy.comms.comm import define_halosend

from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.base.base import make_get_conf

class StreamerSolver():
    
    _parameters = [('De', float, 0., 0.,
                    'Diffusion in x direction'),
                    ('dt', float, 0., 0.,
                    'time step'),
                    ('order', int, 1, 1,
                     'order of the convective scheme'),
                    ('cfl', float, .4, 0,
                     'cfl of the explicit scheme'),
                    ('Mann', float, 0., 0.,
                     'Manning number for the friction'),
                     ('fc', float, 0., 0.,
                     'Coriolis force'),
                     ('grav', float, 9.81, 0.,
                      'gravity constant')
    ]
    
    @classmethod
    def process_conf(cls, conf, kwargs):
        """                                                                                                                                                                                                
        Process configuration parameters.                                                                                                                                                                  
        """
        get = make_get_conf(conf, kwargs)
        
        if len(cls._parameters) and cls._parameters[0][0] != 'name':
            options = StreamerSolver._parameters + cls._parameters

        else:
            options = StreamerSolver._parameters
            
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
    
    def __init__(self, ne=None, ni=None, vel=(None, None, None),  E=(None, None, None), P=None, conf=None, **kwargs):
        
        
        if conf is None:
            conf = Struct()
            
        new_conf = self.process_conf(conf, kwargs)
        self.conf = new_conf
        get = make_get_conf(self.conf, kwargs)
        
        if not isinstance(ne, Variable):
            raise ValueError("ne must be a Variable type")
        
        if not isinstance(ni, Variable):
            raise ValueError("ni must be a Variable type")
        
        if not isinstance(vel[0], Variable):
            raise ValueError("u must be a Variable type")
        
        if not isinstance(vel[1], Variable):
            raise ValueError("v must be a Variable type")
            
        if not isinstance(E[0], Variable):
            raise ValueError("Ex must be a Variable type")
        if not isinstance(E[1], Variable):
            raise ValueError("Ey must be a Variable type")
        
        if not isinstance(P, Variable):
            raise ValueError("P must be a Variable type")
    
        
        self.ne = ne
        self.ni = ni
        
        self.comm = self.ne.comm
        self.domain = self.ne.domain
        self.dim = self.ne.dim
        
        self.u   = vel[0]
        self.v   = vel[1]
        self.Ex  = E[0]
        self.Ey  = E[1]
        self.P   = P
        
        self.varbs = {}
        self.varbs['ne'] = self.ne
        self.varbs['ni'] = self.ni
        
        
        if len(vel) == 3:
            if not isinstance(vel[2], Variable):
                raise ValueError("w must be a Variable type")
            self.w = vel[2]
        else:
            self.w = Variable(domain=self.domain)
            
        if len(E) == 3:
            if not isinstance(vel[2], Variable):
                raise ValueError("Ez must be a Variable type")
            self.Ez = vel[2]
        else:
            self.Ez = Variable(domain=self.domain)
        
        terms = ['source', 'dissipation', "convective"]
        for var in self.varbs.values():
            for term in terms:
                var.__dict__[term] = zeros(self.domain.nbcells)
        
        # Constants
        self.De   = get("De")
        self.dt    = get("dt")
        self.order = get("order")
        self.cfl   = get("cfl")

        self.backend = self.domain.backend
        self.signature = self.domain.signature
        
        if self.dim == 2:
            self._explicitscheme_convective = explicitscheme_convective_2d
        elif self.dim == 3:
            self._explicitscheme_convective = explicitscheme_convective_3d
        
        self._explicitscheme_convective  = self.backend.compile(self._explicitscheme_convective, signature=self.signature)
        self._explicitscheme_dissipative  = self.backend.compile(explicitscheme_dissipative_ST, signature=self.signature)
        self._explicitscheme_source = self.backend.compile(explicitscheme_source_ST, signature=self.signature)
        
        self._compute_el_field = self.backend.compile(compute_el_field, signature=self.signature)
        self._compute_velocity = self.backend.compile(compute_velocity, signature=self.signature)
        
        self._time_step = self.backend.compile(time_step_ST, signature=self.signature)
        self._update_new_value = self.backend.compile(update_ST, signature=self.signature)
        
        self.update_rhs_glob = self.backend.compile(update_rhs_glob, signature=self.signature)
        self.update_rhs_loc = self.backend.compile(update_rhs_loc, signature=self.signature)
        
#        if self.domain.solver == "petsc":
#            self.rhs_updated = zeros(self.domain.nbcells, dtype=self.ne.cell.dtype)
#            self._update_rhs = self.backend.compile(self.update_rhs_glob, signature=self.signature)
#        else:
#            self.rhs_updated = zeros(self.domain.globalsize)
#            self._update_rhs = self.backend.compile(self.update_rhs_loc, signature=self.signature)
        #update rhs
#        update_rhs_glob(ne.cell, ni.cell, cells.loctoglob, rhs_updated) 
#        rhs = COMM.reduce(rhs_updated, op=MPI.SUM, root=0)
    
    
    def update_rhs(self):
        if self.domain.solver == "petsc":
            self.rhs_updated = zeros(self.domain.nbcells, dtype=self.ne.cell.dtype)
            self.update_rhs_loc(self.ne.cell, self.ni.cell, self.domain.cells.loctoglob, self.rhs_updated)
        else:
            self.rhs_updated = zeros(self.domain.globalsize, dtype=self.ne.cell.dtype)
            self.update_rhs_glob(self.ne.cell, self.ni.cell, self.domain.cells.loctoglob, self.rhs_updated)
            
        return self.rhs_updated
    
    def explicit_convective(self):
        if self.order == 2:
            self.ne.compute_cell_gradient()
        self._explicitscheme_convective(self.ne.convective, self.ne.cell, self.ne.ghost, self.ne.halo, self.u.face, self.v.face, self.w.face,
                                        self.ne.gradcellx, self.ne.gradcelly, self.ne.gradcellz, self.ne.gradhalocellx, 
                                        self.ne.gradhalocelly, self.ne.gradhalocellz, self.ne.psi, self.ne.psihalo, 
                                        self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol, 
                                        self.domain.faces.ghostcenter, self.domain.faces.cellid, self.domain.faces.normal, 
                                        self.domain.faces.halofid, self.domain.faces.name, 
                                        self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, 
                                        self.domain.periodicboundaryfaces, self.domain.cells.shift, order=self.order)
    
    def explicit_dissipative(self):
        self.ne.compute_face_gradient()
        self._explicitscheme_dissipative(self.u.face, self.v.face, self.w.face, self.Ex.face,  self.Ey.face, self.Ez.face, self.ne.gradfacex,
                                         self.ne.gradfacey, self.ne.gradfacez, self.domain.faces.cellid, self.domain.faces.normal,
                                         self.domain.faces.name, self.ne.dissipation)
        
    def stepper(self):
        ######calculation of the time step
        dt_c = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.Ex.cell, self.Ey.cell, 
                               self.Ez.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                               self.domain.cells.volume, self.domain.cells.faceid, self.dim)
        
        self.dt = self.comm.allreduce(dt_c, MPI.MIN)
        return  self.dt
    
    
    def update_halo_values(self):
        requests = []
        for var in self.varbs.values():
            define_halosend(var.cell, var.halotosend, var.domain.halos.indsend)
            req = Iall_to_all(var.halotosend, var.nbhalos, var.domain.halos.scount, var.domain.halos.rcount, var.halo, 
                              var.comm)
            requests.append(req)
        MPI.Request.Waitall( requests )

            
    def update_ghost_values(self):
        for var in self.varbs.values():
            var.update_ghost_value()
     
    def update_term_source(self, branching=0):
        
        self._explicitscheme_source(self.ne.cell, self.u.cell, self.v.cell, self.w.cell, self.Ex.cell, self.Ey.cell, 
                                       self.Ez.cell, self.ne.source, self.ni.source, self.domain.cells.center, branching)
    def compute_new_val(self):
        
        self._update_new_value(self.ne.cell, self.ni.cell, self.ne.convective, self.ni.convective, self.ne.dissipation, self.ni.dissipation,
                               self.ne.source, self.ni.source, self.dt, self.domain.cells.volume)
    
    def compute_Electric_Field(self):
        self._compute_el_field(self.P.gradfacex, self.P.gradfacey, self.P.gradfacez, self.Ex.face, self.Ey.face, self.Ez.face)
        
    def compute_Velocity(self):
        self._compute_velocity(self.Ex.face, self.Ey.face, self.Ez.face, self.u.face, self.v.face, self.w.face, self.Ex.cell, self.Ey.cell,
                               self.Ez.cell, self.u.cell, self.v.cell, self.w.cell, self.domain.cells.faceid, self.dim)
        
        
    def compute_fluxes(self):
        
        #update halos
        self.update_halo_values()
        
        #update boundary conditions
        self.update_ghost_values()
        
        # update term source
        self.update_term_source()
        
        #convective flux
        self.explicit_convective()
        
        #dissipative flux
        self.ne.interpolate_celltonode()
        self.explicit_dissipative()
            
        
            
        

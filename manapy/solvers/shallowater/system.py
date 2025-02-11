#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 03:05:46 2023

@author: kissami
"""
from numpy import zeros
from mpi4py import MPI

from manapy.solvers.shallowater import (update_SW, time_step_SW, explicitscheme_convective_SW, 
                                        term_source_srnh_SW, term_friction_SW, term_coriolis_SW,
                                        term_wind_SW)

from manapy.solvers.advecdiff import (explicitscheme_dissipative)
from manapy.comms import Iall_to_all, define_halosend, HaloUpdater
#from manapy.comms import define_halosend

from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.base.base import make_get_conf

class ShallowWaterSolver():
    
    _parameters = [('Dxx', float, 0., 0.,
                    'Diffusion in x direction'),
                    ('Dyy', float, 0., 0.,
                    'Diffusion in y direction'),
                    ('dt', float, 0., 0.,
                    'time step'),
                    ('order', int, 1, 1,
                     'order of the convective scheme'),
                    ('cfl', float, .8, 0,
                     'cfl of the explicit scheme'),
                    ('Mann', float, 0., 0.,
                     'Manning number for the friction'),
                     ('fc', float, 0., 0.,
                     'Coriolis force'),
                     ('grav', float, 9.81, 0.,
                      'gravity constant'),
                     ('wind', bool, False, True,
                      'wind')
    ]
    
    @classmethod
    def process_conf(cls, conf, kwargs):
        """                                                                                                                                                                                                
        Process configuration parameters.                                                                                                                                                                  
        """
        get = make_get_conf(conf, kwargs)
        
        if len(cls._parameters) and cls._parameters[0][0] != 'name':
            options = ShallowWaterSolver._parameters + cls._parameters

        else:
            options = ShallowWaterSolver._parameters
            
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
    
    def __init__(self, h=None, hvel=(None, None), hc=None, Z=None, conf=None, **kwargs):
        
        
        if conf is None:
            conf = Struct()
            
        new_conf = self.process_conf(conf, kwargs)
        self.conf = new_conf
        get = make_get_conf(self.conf, kwargs)
        
        if not isinstance(h, Variable):
            raise ValueError("h must be a Variable type")
        
        if not isinstance(hvel[0], Variable):
            raise ValueError("hu must be a Variable type")
        
        if not isinstance(hvel[1], Variable):
            raise ValueError("hv must be a Variable type")
        
        self.h = h
        self.comm = self.h.comm
        self.domain = self.h.domain
        self.dim = self.h.dim
        self.float_precision = self.domain.float_precision
        
        self.hu   = hvel[0]
        self.hv   = hvel[1]
        
        self.varbs = {}
        self.varbs['h'] = self.h
        self.varbs['hu'] = self.hu
        self.varbs['hv'] = self.hv
        
        if Z is not None:
            if not isinstance(Z, Variable):
                raise ValueError("Z must be a Variable type") 
        else:
            Z = Variable(domain=self.domain)
            
        
        if hc is not None:
            if not isinstance(hc, Variable):
                raise ValueError("hC must be a Variable type") 
        else:
            hc = Variable(domain=self.domain)
            
        self.hc = hc
        self.Z  = Z
        self.varbs['hc'] = self.hc
        self.varbs['Z'] = self.Z
        
        
        terms = ['source', 'dissipation', 'coriolis', 'friction', "convective"]
        for var in self.varbs.values():
            for term in terms:
                var.__dict__[term] = zeros(self.domain.nbcells, dtype=self.float_precision)
        
        # Constants
        self.Dxx   = get("Dxx")
        self.Dyy   = get("Dyy")
        self.Dzz   = 0.
        self.dt    = get("dt")
        self.order = get("order")
        self.cfl   = get("cfl")
        self.Mann  = get("Mann")
        self.fc    = get("fc")
        self.grav  = get("grav")
        self.wind  = get("wind")
        
        
        if self.Dxx == self.Dyy == 0:
            self.diffusion = False
        
        self.backend = self.domain.backend
        self.signature = self.domain.signature
        
        self._explicitscheme_convective  = self.backend.compile(explicitscheme_convective_SW, signature=self.signature)
        self._explicitscheme_dissipative  = self.backend.compile(explicitscheme_dissipative, signature=self.signature)
        self._time_step_SW = self.backend.compile(time_step_SW, signature=self.signature)
        self._update_new_value = self.backend.compile(update_SW, signature=self.signature)
        self._term_coriolis_SW = self.backend.compile(term_coriolis_SW, signature=self.signature)
        self._term_friction_SW = self.backend.compile(term_friction_SW, signature=self.signature)
        self._term_wind_SW = self.backend.compile(term_wind_SW, signature=self.signature)
        self._term_source_srnh = self.backend.compile(term_source_srnh_SW, signature=self.signature)
        
    
    def explicit_convective(self):
        
        if self.order == 2:
            self.h.compute_cell_gradient()
            self.hc.compute_cell_gradient()
            
        explicitscheme_convective_SW(self.h.convective, self.hu.convective, self.hv.convective, self.hc.convective, self.Z.convective, 
                                     self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell,
                                     self.h.ghost, self.hu.ghost, self.hv.ghost, self.hc.ghost, self.Z.ghost, self.h.halo, 
                                     self.hu.halo, self.hv.halo, self.hc.halo, self.Z.halo,
                                     self.h.gradcellx, self.h.gradcelly, self.h.gradhalocellx, self.h.gradhalocelly,
                                     self.hc.gradcellx, self.hc.gradcelly, self.hc.gradhalocellx, 
                                     self.hc.gradhalocelly, self.hc.psi, self.hc.psihalo, 
                                     self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol, 
                                     self.domain.faces.ghostcenter, self.domain.faces.cellid, self.domain.faces.mesure, 
                                     self.domain.faces.normal, self.domain.faces.halofid,  
                                     self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, self.grav, self.order)
    
    def explicit_dissipative(self):
    
        self.hc.compute_face_gradient()
        self._explicitscheme_dissipative(self.hc.gradfacex, self.hc.gradfacey, self.hc.gradfacez, self.domain.faces.cellid, 
                                         self.domain.faces.normal, self.domain.faces.name, self.hc.dissipation, self.Dxx, self.Dyy, self.Dzz)
    
    def stepper(self):
        ######calculation of the time step
        dt_c = self._time_step_SW(self.h.cell, self.hu.cell, self.hv.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                                  self.domain.cells.volume, self.domain.cells.faceid, self.grav, self.Dxx, self.Dyy)
        
        self.dt = self.comm.allreduce(dt_c, MPI.MIN)
#        return  self.dt
    
    
    def update_halo_values(self):
        
#        updater = HaloUpdater(self.domain, self.varbs, self.comm)
#        updater.update_halo_values()
        requests = []
        for var in self.varbs.values():
            define_halosend(var.cell, var.halotosend, var.domain.halos.indsend)
            req = Iall_to_all(var.halotosend, var.nbhalos, var.domain.halos.scount, var.domain.halos.rcount, var.halo, 
                              var.comm)
            requests.append(req)
        #MPI.Request.Waitall( requests )
        
        return requests
            
    def update_ghost_values(self):
        for var in self.varbs.values():
            var.update_ghost_value()
            
    def interpolate_cell2node(self):
        for var in self.varbs.values():
            var.interpolate_celltonode()
        
    def update_term_source(self):
        self._term_source_srnh(self.h.source, self.hu.source, self.hv.source,self.hc.source, self.Z.source,
                               self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell,
                               self.h.ghost, self.hu.ghost, self.hv.ghost, self.hc.ghost, self.Z.ghost,
                               self.h.halo, self.hu.halo, self.hv.halo, self.hc.halo, self.Z.halo,
                               self.h.gradcellx, self.h.gradcelly, self.hc.psi, self.h.gradhalocellx, self.h.gradhalocelly, 
                               self.hc.psihalo, 
                               self.domain.cells.nodeid, self.domain.cells.faceid, self.domain.cells.cellfid, self.domain.faces.cellid,
                               self.domain.cells.center, self.domain.cells.nf, 
                               self.domain.faces.name, self.domain.faces.center, self.domain.halos.centvol,
                               self.domain.nodes.vertex, self.domain.faces.halofid, self.grav, self.order)
    
    
    def update_term_friction(self):
        self._term_friction_SW(self.h.cell, self.hu.cell, self.hv.cell, self.grav, self.Mann, self.dt) 
    
    def update_term_coriolis(self):
        self._term_coriolis_SW(self.hu.cell, self.hv.cell, self.hu.coriolis, self.hv.coriolis, self.fc)
        
    
    def compute_new_val(self):
        self._update_new_value(self.h.cell, self.hu.cell, self.hv.cell, self.hc.cell, self.Z.cell ,
                               self.h.convective, self.hu.convective, self.hv.convective, self.hc.convective, self.Z.convective,
                               self. h.source, self.hu.source, self.hv.source, self.hc.source, self.Z.source,
                               self.hc.dissipation, self.hu.coriolis, self.hv.coriolis,
                               0., 0., self.dt, self.domain.cells.volume)
        
        
    def update_term_wind(self):
        self._term_wind_SW(self.domain.cells.center, self.Tx_wind, self.Ty_wind, self.wind, self.iteration)
        
    def compute_fluxes(self):
        
        
        #update halos
        requests = self.update_halo_values()
        
        #update friction term
        if self.Mann != 0:
            self.update_term_friction() 
        
        self.stepper()
        
        MPI.Request.Waitall( requests )        
        
        #update boundary conditions
        self.update_ghost_values()
        
        #convective flux
        self.explicit_convective()
        
        #dissipative flux
        if self.diffusion: 
            self.var.interpolate_celltonode()
            self.explicit_dissipative()
        
        # update term source
        self.update_term_source()
        
        if self.fc != 0:
            #update coriolis forces
            self.update_term_coriolis()
            
        if self.wind:
            self.update_term_wind()
        

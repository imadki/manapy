import re


class CreateSystem():

  def __init__(self, path : str, config):
    self.path = path
    self.process_token = self.__get_process_token()
    self.conf = config
  
  def generate(self):
    with open(self.conf.template) as file:
      data = file.read()
      pattern = r'\$\$\[(.*?)\]'
      result = re.sub(pattern, self.__replace_with, data)
      with open(self.path, "w") as out_file:
        out_file.write(result)
  
  ##########################################
  ## Replacer
  ##########################################
  """
  Replacer
  """
  def __get_process_token(self):
    return {
      "__import_solver_utils" : self.__import_solver_utils,
      "__advection_explicit_scheme" : self.__advection_explicit_scheme,
      "__diffusion_explicit_scheme" : self.__diffusion_explicit_scheme,
      "__diffusion_params" : self.__diffusion_params,
      "__init_variables" : self.__init_variables,
      "__self_explicit_scheme_convective" : self.__self_explicit_scheme_convective,
      "__self_explicit_scheme_dissipative" : self.__self_explicit_scheme_dissipative,
      "__self_var_dict_convective" : self.__self_var_dict_convective,
      "__self_var_dict_dissipative" : self.__self_var_dict_dissipative,
      "__self_var_dict_source" : self.__self_var_dict_source,
      "__def_explicit_convective" : self.__def_explicit_convective,
      "__def_explicit_dissipative" : self.__def_explicit_dissipative,
      "__def_stepper" : self.__def_stepper,
      "__def_compute_fluxes" : self.__def_compute_fluxes,
    }
  
  def __import_solver_utils(self):
    res = "import "
    if self.conf.advection == True and self.conf.diffusion == True:
      res += "manapy.solvers.advecdiff "
    elif self.conf.advection == True:
      res += "manapy.solvers.advec "
    elif self.conf.diffusion == True:
      res += "manapy.solvers.diffusion "
    res += "as solver_utils"
    return res

  def __advection_explicit_scheme(self):
    if self.conf.advection == True:
      if self.conf.dim == 3:
        return 'self.explicitscheme_convective = solver_utils.explicitscheme_convective_2d'
      else :
        return 'self.explicitscheme_convective = solver_utils.explicitscheme_convective_3d'


  def __diffusion_explicit_scheme(self):
    if self.conf.diffusion == True:
      return "self.explicitscheme_dissipative = solver_utils.explicitscheme_dissipative"


  def __diffusion_params(self):
    res = ", Dxx, Dyy"
    if self.conf.diffusion == True:
      if self.conf.dim == 3:
        res += ", Dzz"
      return res


  def __init_variables(self):
    backend = self.conf.backend
    mesh_path = self.conf.mesh_path
    dim = self.conf.dim

    a = ""
    if self.conf.dim == 3:
      a = "self.w  = Variable(domain=domain)"

    b = ""
    if self.conf.diffusion == True:
      b = """
    self.Dxx   = np.float64(Dxx)
    self.Dyy   = np.float64(Dyy)
    """
      if self.conf.dim == 3:
        b += "self.Dzz   = np.float64(Dzz)"
    
    res = f"""
    


    running_conf = Struct(backend="{backend}", signature=True, cache=True, precision="double")
    MeshPartition("{mesh_path}", dim={dim}, conf=running_conf, periodic=[0,0,0])

    running_conf = Struct(backend="{backend}", signature=True, cache =True, precision="double")
    domain = Domain(dim={dim}, conf=running_conf)

    ne = Variable(domain=domain)
    self.u  = Variable(domain=domain)
    self.v  = Variable(domain=domain)
    {a}

    self.var = ne
    self.comm = self.var.comm
    self.domain = self.var.domain
    self.dim = self.var.dim
    self.float_precision = self.domain.float_precision
    

    self.cfl   = np.float64(cfl)
    {b}

    self.backend = self.domain.backend
    self.signature = self.domain.signature
    """
    return res


  def __self_explicit_scheme_convective(self):
    if self.conf.advection == True:
      return "self._explicitscheme_convective  = self.backend.compile(solvers.explicitscheme_convective, signature=self.signature)"


  def __self_explicit_scheme_dissipative(self):
    if self.conf.diffusion == True:
      return "self._explicitscheme_dissipative  = self.backend.compile(solvers.explicitscheme_dissipative, signature=self.signature)"
  

  def __self_var_dict_convective(self):
    if self.conf.advection == True:
      return 'self.var.__dict__["convective"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)'

  def __self_var_dict_dissipative(self):
    if self.conf.diffusion == True:
      return 'self.var.__dict__["dissipative"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)'

  def __self_var_dict_source(self):
    return 'self.var.__dict__["source"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)'


  def __def_explicit_convective(self):
    if self.conf.advection == True:
      a = ""
      if self.conf.order == 2:
        a = "self.var.compute_cell_gradient()"
      res = f"""
  def __explicit_convective(self):
    {a}
    self._explicitscheme_convective(self.var.convective, self.var.cell, self.var.ghost, self.var.halo, self.u.face, self.v.face, self.w.face,
                                    self.var.gradcellx, self.var.gradcelly, self.var.gradcellz, self.var.gradhalocellx, 
                                    self.var.gradhalocelly, self.var.gradhalocellz, self.var.psi, self.var.psihalo, 
                                    self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol, 
                                    self.domain.faces.ghostcenter, self.domain.faces.cellid, self.domain.faces.normal, 
                                    self.domain.faces.halofid, self.domain.faces.name, 
                                    self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, 
                                    self.domain.periodicboundaryfaces, self.domain.cells.shift, order=self.order)
      
      """
      return res


  def __def_explicit_dissipative(self):
    Dzz = ""
    if self.conf.dim == 3:
      Dzz = ", self.Dzz"

    if self.conf.diffusion == True:
      res = f"""
  def __explicit_dissipative(self):
    self.var.compute_face_gradient()
    self._explicitscheme_dissipative(self.var.gradfacex, self.var.gradfacey, self.var.gradfacez, self.domain.faces.cellid, 
                                      self.domain.faces.normal, self.domain.faces.name, self.var.dissipative, self.Dxx, self.Dyy {Dzz})   
      """
      return res


  def __def_stepper(self):
    res = "def __stepper(self):"
    if self.conf.advection and not self.conf.diffusion:
      a = """
    d_t = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                          self.domain.cells.volume, self.domain.cells.faceid, self.dim)
      """
    else:
      Dzz = ""
      if self.conf.dim == 3:
       Dzz = ", self.Dzz"
      a = f"""
    d_t = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                          self.domain.cells.volume, self.domain.cells.faceid, self.dim, self.Dxx, self.Dyy {Dzz})
      """
    res = f"""
  def __stepper(self):
    {a}
    self.dt = self.comm.allreduce(d_t, op=MPI.MIN)
    return  self.dt
    """
    return res


  def __def_compute_fluxes(self):
    a = ""
    b = ""
    if self.conf.advection == True:
      a = """
    self.__explicit_convective()
      """
    if self.conf.diffusion == True:
      b = """
    self.var.interpolate_celltonode()
    self.__explicit_dissipative()
      """
    res = f"""
  def compute_fluxes(self):

    self.var.update_halo_value()
    self.var.update_ghost_value()
    
    {a}
    {b}
    """
    return res

  def __replace_with(self, match):
    matched_value = match.group(1)
    if matched_value in self.process_token:
      res = self.process_token[matched_value]()
      if res != None:
        return res
    return ""





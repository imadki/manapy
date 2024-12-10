from manapy.cuda.utils import (
  VarClass,
  GPU_Backend
)

import numpy as np
from mpi4py import MPI
from manapy.ast import Variable
from manapy.base.base import Struct
from manapy.ddm import Domain
from manapy.partitions import MeshPartition
from manapy.solvers.advec.tools_utils import initialisation_gaussian_2d
import os

from timeit import default_timer as timer

def test_time(iter, fun):
  #fun()
  start_time = timer()
  for _ in range(iter):
    fun()
  end_time = timer()
  elapsed_time = (end_time - start_time) / iter
  print(f"{elapsed_time * 1000:.5f} ms")
  return elapsed_time * 1000
  #print(f"{elapsed_time * 1000000:.5f} micros")

def init(dim, mesh_file_name, float_precision):
  GPU_Backend.float_precision = 'float64'
  if float_precision == 'single':
    GPU_Backend.float_precision = 'float32'
  running_conf = Struct(backend="numba", signature=True, cache=True, float_precision=float_precision)

  #running_conf = Struct(backend="numba", signature=True, cache=True, precision="double")
  #?create the mesh that will be used by domain
  MeshPartition(mesh_file_name, dim=dim, conf=running_conf, periodic=[0,0,0])
  domain = Domain(dim=dim, conf=running_conf)
  ne = Variable(domain=domain)
  u  = Variable(domain=domain)
  v  = Variable(domain=domain)
  w  = Variable(domain=domain)
  
  P = Variable(domain=domain)
  Pinit = 2.0
  cells = domain.cells
  initialisation_gaussian_2d(ne.cell, u.cell, v.cell, P.cell, cells.center, Pinit)

  u.face[:] = 2.
  v.face[:] = 0.
  w.face[:] = 0.
  
  u.interpolate_facetocell()
  v.interpolate_facetocell()
  w.interpolate_facetocell()
  return (domain, ne, u, v, w)


class Solver():
  
  def __init__(self, var, vel, order, cft, diffusion = False, advection = False, Dxx = 0.0, Dyy = 0.0, Dzz = 0.0, dim = 2):

    self.var = var
    self.comm = self.var.comm
    self.domain = self.var.domain
    self.dim = self.var.dim
    self.float_precision = self.domain.float_precision
    
    self.u  = vel[0]
    self.v  = vel[1]
    self.w  = vel[2]

  

    self.advection = advection
    self.diffusion = diffusion
    self.order = np.int32(order)
    self.cfl   = np.float64(cft)
    self.Dxx   = np.float64(Dxx)
    self.Dyy   = np.float64(Dyy)
    self.Dzz   = np.float64(Dzz)

    self.backend = self.domain.backend
    self.signature = self.domain.signature


  
    #self.dt    = self.__stepper()
    


    self.var.__dict__["convective"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)
    self.var.__dict__["dissipative"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)
    self.var.__dict__["source"] = np.zeros(self.domain.nbcells, dtype=self.float_precision)
    

  


  #!only if advec or advecDiff
  def __explicit_convective(self):
      if self.order == 2:
          self.var.compute_cell_gradient()
      self._explicitscheme_convective(self.var.convective, self.var.cell, self.var.ghost, self.var.halo, self.u.face, self.v.face, self.w.face,
                                      self.var.gradcellx, self.var.gradcelly, self.var.gradcellz, self.var.gradhalocellx, 
                                      self.var.gradhalocelly, self.var.gradhalocellz, self.var.psi, self.var.psihalo, 
                                      self.domain.cells.center, self.domain.faces.center, self.domain.halos.centvol, 
                                      self.domain.faces.ghostcenter, self.domain.faces.cellid, self.domain.faces.normal, 
                                      self.domain.faces.halofid, self.domain.faces.name, 
                                      self.domain.innerfaces, self.domain.halofaces, self.domain.boundaryfaces, 
                                      self.domain.periodicboundaryfaces, self.domain.cells.shift, self.order)
  

  
  def stepper(self):
    d_t = self._time_step(self.u.cell, self.v.cell, self.w.cell, self.cfl, self.domain.faces.normal, self.domain.faces.mesure, 
                          self.domain.cells.volume, self.domain.cells.faceid, self.dim)
    self.dt = self.comm.allreduce(d_t, op=MPI.MIN)
    return  self.dt

  def compute_fluxes(self):
      #interpolate cell to node
      self.var.update_halo_value()
      self.var.update_ghost_value()
      self.__explicit_convective()
      

  def compute_new_val(self):
    self._update_new_value(self.var.cell, self.var.convective, self.var.dissipative, self.var.source, self.dt, self.domain.cells.volume)
  



dim=2
#mesh_file_name="/home/aben-ham/Desktop/work/stage/my_manapy/manapy/mesh/2D/carre.msh"
#mesh_file_name="/home/aben-ham/Desktop/work/stage/manapy/mesh/2D/square_larger.msh"
mesh_file_name="/home/aben-ham/Desktop/work/stage/my_manapy/gpu_accelerator/functions/square.msh"
diffusion=False
advection=True
order=2

domain, ne, u, v, w = init(dim=dim, mesh_file_name=mesh_file_name, float_precision='single')
S = Solver(ne, vel=(u, v, w), order=order, cft=0.8,
          diffusion=diffusion, advection=advection, Dxx=0.0, Dyy=0.0, Dzz=0.0, dim = dim)



VarClass.convert_to_var_class([
  ne,
  u,
  v,
  w,
  domain,
  domain.nodes,
  domain.faces,
  domain.cells,
  domain.halos,
  S,
])



GPU_TIME_COUNTER = 0.0
CPU_TIME_COUNTER = 0.0

def INIT_FOR(exec_type):

  print("===================================")
  print(f"Testing on {exec_type}")
  print("===================================")
 
  if exec_type == 'cpu':
    exec_type = 0
  elif exec_type == 'cuda':
    exec_type = 1
  elif exec_type == 'test':
    exec_type = 2
  elif exec_type == 'test_2':
    exec_type = 3
  else:
    raise RuntimeError("wrong exec_type value in INIT_FOR")

  from manapy.cuda.manapy.ast.cuda_ast_utils import (
    get_kernel_facetocell,
    get_kernel_celltoface,

  )

  from manapy.cuda.manapy.comms.cuda_communication import (
    get_kernel_define_halosend
  )

  from manapy.cuda.manapy.ast.cuda_functions2d import (
    get_kernel_centertovertex_2d,
    get_kernel_face_gradient_2d,
    get_kernel_cell_gradient_2d,
    get_kernel_barthlimiter_2d,
  )

  from manapy.cuda.manapy.solvers.advec.cuda_fvm_utils import (
    get_kernel_explicitscheme_convective_2d,
    get_kernel_time_step,
    get_kernel_update_new_value,
  )

  from manapy.cuda.manapy.util_kernels import get_kernel_assign

  #==========================
  #==========================

  from manapy.ast.ast_utils  import (
        facetocell,
        celltoface
      )

  from manapy.comms import define_halosend

  from manapy.ast.functions2d import (
    centertovertex_2d,
    face_gradient_2d,
    cell_gradient_2d,
    barthlimiter_2d
  )

  from manapy.solvers.advec import (
    explicitscheme_convective_2d,
    time_step,
    update_new_value
  )

  #==========================
  #==========================

  def arr_assign(arr, val):
    arr[:] = val

  def get_test_function(cpu_fun, gpu_fun):
    def result(*args):
      print(f'testing cpu... {cpu_fun}')
      cpu_val = cpu_fun(*args)
      print(f'testing gpu...{gpu_fun}')
      gpu_val = gpu_fun(*args)
      
      if cpu_val != None:
        np.testing.assert_almost_equal(cpu_val, gpu_val, decimal=1)
      for arg in args:
        try:
          if isinstance(arg, VarClass) == True:
            np.testing.assert_almost_equal(arg, arg.to_host(), decimal=1)
        except:
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      return cpu_val

    return result
  
  def test_speed(cpu_fun, gpu_fun):
    def result(*args):
      #global CPU_TIME_COUNTER
      global GPU_TIME_COUNTER
      #cpu_val = cpu_fun(*args)
      gpu_val = gpu_fun(*args)
      #CPU_TIME_COUNTER += test_time(20, lambda : cpu_fun(*args))
      GPU_TIME_COUNTER += test_time(20, lambda : gpu_fun(*args))
      #if cpu_val != None:
      #  np.testing.assert_almost_equal(cpu_val, gpu_val, decimal=1)
      return gpu_val
    return result



  funcs = [
    [domain.backend.compile(facetocell, signature=True), get_kernel_facetocell(), None],
    [domain.backend.compile(celltoface, signature=True), get_kernel_celltoface(), None],
    [domain.backend.compile(define_halosend, signature=True), get_kernel_define_halosend(), None],
    [domain.backend.compile(centertovertex_2d, signature=True), get_kernel_centertovertex_2d(), None],
    [domain.backend.compile(face_gradient_2d, signature=True), get_kernel_face_gradient_2d(), None],
    [domain.backend.compile(cell_gradient_2d, signature=True), get_kernel_cell_gradient_2d(), None],
    [domain.backend.compile(barthlimiter_2d, signature=True), get_kernel_barthlimiter_2d(), None],
    [domain.backend.compile(explicitscheme_convective_2d, signature=True), get_kernel_explicitscheme_convective_2d(), None],
    [domain.backend.compile(time_step, signature=True), get_kernel_time_step(), None],
    [domain.backend.compile(update_new_value, signature=True), get_kernel_update_new_value(), None],

    [arr_assign, get_kernel_assign(), None],
  ]

  if exec_type == 2:
    for fun in funcs:
      fun[2] = get_test_function(fun[0], fun[1])
  elif exec_type == 3:
    for fun in funcs:
      fun[2] = test_speed(fun[0], fun[1])
    exec_type = 2

  i = 0
  Variable._facetocell = funcs[i][exec_type]; i += 1
  Variable._celltoface = funcs[i][exec_type]; i += 1
  Variable._define_halosend = funcs[i][exec_type]; i += 1
  #2d
  Variable._func_interp   = funcs[i][exec_type]; i += 1
  Variable._face_gradient = funcs[i][exec_type]; i += 1
  Variable._cell_gradient = funcs[i][exec_type]; i += 1
  Variable._barthlimiter  = funcs[i][exec_type]; i += 1

  S._explicitscheme_convective = funcs[i][exec_type]; i += 1
  S._time_step = funcs[i][exec_type]; i += 1
  S._update_new_value = funcs[i][exec_type]; i += 1

  arr_assign = funcs[i][exec_type]; i += 1
  
  S.dt    = S.stepper()
  
  return arr_assign

#=======================================================================
#=======================================================================
#=======================================================================
# Testing
#=======================================================================
#=======================================================================
#=======================================================================

arr_assign = INIT_FOR('cpu')

def work(domain, S, vars, tfinal):
  print("Start working ...")
  ne, u, v, w = vars

  

  time = 0
  #tfinal = .25
  niter = 1
  miter = 0
  dt = S.dt

  ts = MPI.Wtime()


  while time < tfinal :
    arr_assign(u.face, 2.0)
    arr_assign(v.face, 2.0)
    arr_assign(w.face, 2.0)
    #u.face[:] = 2.
    #v.face[:] = 0.
    #w.face[:] = 0.
    
    u.interpolate_facetocell()
    v.interpolate_facetocell()
    w.interpolate_facetocell()

    
    tot = int(tfinal / (dt * 50)) + 1

    time = time + dt
  
    S.compute_fluxes()
    S.compute_new_val()

    print(niter, f"=> time {time} / {tfinal}")
    # if niter == 1 or niter % tot == 0:
    #   values = [a.to_host() for a in [ne.cell, u.cell,v.cell]]
    #   domain.save_on_cell_multi(dt, time, niter, miter, variables=["ne", "u","v","P"], values=values)
    #   miter += 1
    
    niter += 1

  te = MPI.Wtime()

  COMM = MPI.COMM_WORLD
  RANK = COMM.Get_rank()

  tt = COMM.reduce(te - ts, op=MPI.MAX, root=0)
  if RANK == 0:
      print("Time to do calculation", tt)
  return tt


def test_1():
  print('==================================')
  print('==================================')
  u.interpolate_facetocell()
  print('==================================')
  print('==================================')
  S.compute_fluxes()
  print('==================================')
  print('==================================')
  S.compute_new_val()


def test_2():
  # 34.29
  # 167.05

  def block():
    u.interpolate_facetocell()
    u.interpolate_facetocell()
    u.interpolate_facetocell()
    S.compute_fluxes()
    S.compute_new_val()

  test_time(10, block) 


def test_3():
  # Time to do calculation 221.77s CPU 
  # Time to do calculation 65.645s GPU x3.37
  work(domain, S, (ne, u, v, w), 0.25)


def test_4(root_path):
  arr_nb_threads = [32, 64, 128, 256, 512]
  nb_blocks = 32

  def count_files_in_folder(folder_path):
      try:
          # List all files and directories in the specified folder
          entries = os.listdir(folder_path)
          
          # Count files
          file_count = sum(os.path.isfile(os.path.join(folder_path, entry)) for entry in entries)
          
          return file_count
      
      except FileNotFoundError:
          print("The specified folder does not exist.")
          return 0
      except Exception as e:
          print(f"An error occurred: {e}")
          return 0

  def save_test(out_file, nb_blocks, nb_threads, is_free):
    GPU_Backend.free = is_free
    GPU_Backend.nb_blocks = nb_blocks
    GPU_Backend.nb_threads = nb_threads
    taken_time = 0
    nb_iter = 10
    for _ in range(nb_iter):
      taken_time += work(domain, S, (ne, u, v, w), 0.01)
    taken_time /= nb_iter
    if nb_blocks == -1:
      nb_blocks = 'free'
    msg = f'{nb_blocks}-{nb_threads} {taken_time}\n'
    out_file.write(msg)
    out_file.flush()
    #print(msg)


  #root_path = "/home/aben-ham/Desktop/work/stage/manapy/manapy/cuda/example/benchmark_result"
  next_name = count_files_in_folder(root_path) + 1
  with open(f'{root_path}/benchmark_result_{next_name}.csv', 'w') as out_file:
    out_file.write(f"advection 2d {mesh_file_name} nb_cells: {domain.nbcells}\n")
    
    GPU_Backend.free = True
    for nb_threads in arr_nb_threads:
      save_test(out_file, -1, nb_threads, True)
    
    GPU_Backend.free = False
    while nb_blocks <= 65536:
      for nb_threads in arr_nb_threads:
        save_test(out_file, nb_blocks, nb_threads, False)
      nb_blocks *= 2


root_path = "/home/aben-ham/Desktop/work/stage/manapy/manapy/cuda/examples/benchmark_result"
#create benchmark_result folder
#test_4(root_path)
ne.compute_cell_gradient()







from numba import cuda
import inspect
import numpy as np
from . import GPU_Backend

class VarClass(np.ndarray):
  def __new__(cls, input_array):
    if isinstance(input_array, (int, float, np.int32, np.float32, np.float64, VarClass)):
      return input_array
      
    obj = np.asarray(input_array).view(cls)
    # obj = np.ndarray.__new__(cls, shape=input_array.shape,
    #                              dtype=input_array.dtype,
    #                              buffer=input_array,
    #                              strides=input_array.strides,
    #                              order=None)
    obj.type_shape = f'VarClass<{input_array.dtype}, {input_array.shape}>'
    # obj.__on_device__ = False
    # obj.__cuda__ = None
    return obj
  
  # def __repr__(self):
  #   return self.type_shape

  # def __array_finalize__(self, obj):
  #     if obj is None: return
  #     #print(obj.type_shape)
  #     # Copy custom attributes from the source object if available
  #     self.type_shape = getattr(obj, 'type_shape', None)
  #     self.__on_device__ = getattr(obj, '__on_device__', None)
  #     self.__cuda__ = getattr(obj, '__cuda__', None)

  def to_host(self):
    if hasattr(self, "__on_device__") == False:
      return self
    v = getattr(self, '__cuda__')
    return v.copy_to_host(stream=v.stream)
  

  @staticmethod
  def to_device(numpy_arr):
    if isinstance(numpy_arr, (int, float, np.int32, np.float32, np.float64)):
      return numpy_arr
    if hasattr(numpy_arr, "__on_device__") == False:
      setattr(numpy_arr, '__on_device__', True)
      stream = GPU_Backend.stream
      d_numpy_arr = cuda.to_device(numpy_arr, stream=stream)
      setattr(numpy_arr, '__cuda__', d_numpy_arr)
      return d_numpy_arr
    return getattr(numpy_arr, '__cuda__')
  
  # @staticmethod
  # def debug(fun, args):
  #   pass

  @staticmethod
  def debug(fun, args):
    if hasattr(fun, "__is_called__") == False:
      setattr(fun, '__is_called__', False)
    if getattr(fun, '__is_called__') == False:
      print(f"{fun} is called")
      for i, arg in enumerate(args):
        if isinstance(arg, (int, float, np.int32, np.float32, np.float64)) == False:
          if isinstance(arg, VarClass) == False:
            raise TypeError(f"{i + 1} => {type(arg)}!!")
        if isinstance(arg, (int, float, np.int32, np.float32, np.float64)) == False:
          print(f'{i + 1} => {arg.type_shape}')
        else:
          print(type(arg))
      setattr(fun, '__is_called__', True)

  @staticmethod
  def convert_all_tables(obj):
    #print("---------------------------------")
    #print("---------------------------------")
    attributes = dir(obj)
    arr = []
    

    for attr_name in attributes:
      try:
        attr_value = getattr(obj, attr_name)
        if isinstance(attr_value, np.ndarray):
          arr.append(attr_name)
      except Exception as e:
        pass
        # print(f"can't get attr {attr_name} => {e}")
      
    
    for attr_name in arr:
      try:
        attr_value = getattr(obj, attr_name)
        setattr(obj, attr_name, VarClass(attr_value))
        #print("set att:", attr_name)
      except Exception as e:
        pass
        #print(f"can't set attr for {attr_name} => {e}")
  
  @staticmethod
  def convert_to_var_class(list_obj):
    for item in list_obj:
      VarClass.convert_all_tables(item)
  

  def sync_with(self, input_device):
    if hasattr(self, "__on_device__") == True:
      gpu_array = getattr(self, '__cuda__')
      if input_device == 'cpu':
        cuda.to_device(self, to=gpu_array)
      elif input_device == 'cuda':
        gpu_array.copy_to_host(self, stream=gpu_array.stream)
      return
    elif input_device == 'cpu':
      VarClass.to_device(self)
      return
    elif input_device == 'cuda':
      raise RuntimeError("array isn't on device yet")
    raise RuntimeError("expect input_device to be cpu or cuda")
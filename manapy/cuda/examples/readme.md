## Objective: Run the advection model on GPU.


### Create kernels to all concerned functions
- kernels
  - All kernels support striding.
  - Basically, a kernel can be wrapped inside a function that is call the kernel with the appropriate environment.
  - Appropriate environment like:
    - Return value
    - Extra parameters
    - Calling the debug function aka `VarClass.debug`
    - Get the number of blocks and threads aka `GPU_Backend.get_gpu_prams`
    - Compile kernels inside the wrapper function aka `GPU_backend.compile`:
      - Device kernel
      - Main kernel
      - Auxiliary kernels
    - Create a new device array as an extra parameter
    - Get the device arrays from the argument list aka `VarClass.to_device`
  - Concerned Functions
    - manapy/gpu_fun/ast/cuda_ast_utils.py
      - get_kernel_convert_solution
      - get_kernel_facetocell
      - get_kernel_celltoface
    - manapy/gpu_fun/ast/cuda_functions2d.py
      - cell_gradient_2d
      - face_gradient_2d
      - centertovertex_2d
      - barthlimiter_2d
      - get_triplet_2d 
      - compute_2dmatrix_size
      - get_rhs_loc_2d
      - get_rhs_glob_2d
      - compute_P_gradient_2d_diamond
      - get_triplet_2d_with_contrib
    - manapy/gpu_fun/comms/cuda_communication.py
      - define_halosend `(it turn out that this function it related to MPI)`
    - manapy/gpu_fun/solvers/advec/cuda_fvm_utils.py
      - compute_upwind_flux
      - explicitscheme_convective_2d
      - explicitscheme_convective_3d
      - time_step
      - update_new_value

### Utils
- manapy/gpu_fun/utils/utils_kernels.py
  - VarClass
    - This class is needed to add attributes to NumPy arrays, like an attribute to store a GPU array.
    - NumPy arrays are interfaces, not Python objects. They can't be manipulated to add new attributes.
    - transform NumPy arrays into a Python object (`VarClass`) in order to add attributes to the object.
    - In the case of literals datatype, it will return the same type -> VarClass(int) is an int, not a VarClass type.
    - attributes
      - `__cuda__`
      - `__on_device__`
    - Methods
      - to_host
        - A method to get the host array for the `__cuda__` object.
      - to_device
        - A static method to create a `__cuda__` object if it does not exist.
        - For a variable of type VarClass, it will create an attribute `__cuda__` that represents the variable on the CUDA device.
        - It returns the `__cuda__` attribute.
        - The `__cuda__` attribute is only created when the kernel is called.
      - debug
        - A static method for debugging purposes.
        - For each kernel, this function outputs once.
        - It checks if all arguments are of type VarClass.
        - It prints the shape of the array.
        - It prints the called function.
      - convert_all_tables
        - A static method that loops over all the attributes of an object. If it encounters an `np.ndarray`, it transforms it to `VarClass`.
      - convert_to_var_class
        - A static method that takes an array argument and calls `convert_all_tables` for each item.
      - sync_with
        - A method to synchronize NumPy arrays on CPU with GPU and vice versa.

  - GPU_Backend
    - Set configurations (cache, float_precision, ...).
    - The objective of this class is to compile kernels based on the predefined configuration.
    - Methods
      - set_config
        - A static method to set configurations (cache, float_precision, ...).
      - compile_kernel
        - A static method that calls `GPU_backend.compile()`.
      - get_gpu_prams
        - A static method to get the number of threads and the number of blocks.


### The flow of manapy/cuda/example/main.py
  - Create all essential variables such as domain, ne, system model, etc. => `init, Solver ...`
  - Convert all underlying attributes of type `np.ndarray` to `VarClass` using. => `VarClass.convert_to_var_class`
  - Assign the target functions based on the desired test, whether for CPU or GPU => `INIT_FOR('cuda')`
  - Lunch the test => `test_1, test_2, ...`


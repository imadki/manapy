## How to Add a Compute Unit

Follow these steps to add a new compute unit or integrate an existing one:

1. Implement the required functions and test them with Python or Numba.
2. Decide whether the module will support CPU only or both CPU and GPU.
3. Choose a clear module name. The module will be accessible through:

   ```python
   manapy_compute_ff_ii.module_name.module_function
   ```

   Here, `ff` and `ii` represent the configured floating-point and integer
   precisions.
4. Implement the module in `c_api` by following its README instructions and
   using the existing modules as examples.
5. Build and install the `c_api` library.
6. Expose the new functions in `_compute.py` and make sure you use ManapyArray read/read-write/write access, Also use Partial.
7. Create a dedicated class for the module in the `compute` directory.
8. If necessary, add configurable parameters to the class constructor
   (`__init__`). For example, a `dim` parameter can select the appropriate
   compute functions for a particular dimension.
9. Test the complete Python, CPU, and GPU integration, as applicable.
10. update `README.md#Contents` in `c_api`

### Publishing a New Version

Once everything works correctly:

1. Commit and push the changes.
2. Create a new version tag.
3. Build and publish the new release to PyPI.
4. Check the `.toml` installation files to ensure they include the new version tag.
5. If you make any changes to the `VariableCompute`, `DomainCompute`, or `PartitioningCompute` functions, don't forget to run the dedicated pytest suite.

Publishing is optional and should only be done when the changes are ready for
release.

## Debugging

The following environment variables and utilities can help diagnose problems.

### `MANAPY_CUDA_SYNC`

Enable synchronous CUDA execution to report kernel-launch failures immediately.
This helps identify the kernel responsible for an error.

```bash
export MANAPY_CUDA_SYNC=1
```

### `MANAPY_DEBUG`

Enable verbose debug output and timing information.

```bash
export MANAPY_DEBUG=1
```

### Debugging C Code

Use `print_debug` to print diagnostic information from C code. Remove or disable
unnecessary debug output before publishing a release.

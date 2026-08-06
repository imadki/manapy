import numpy as np
import os
from manapy.backends.ManapyArray import Device

_DTYPES = {
  "float32": np.float32,
  "float64": np.float64,
  "int32": np.int32,
  "int64": np.int64,
}

class ManapyConfig:
  def __init__(self, float_precision: str, int_precision: str, device: str, verbose: bool = False):
    float_precision = float_precision.lower()
    int_precision = int_precision.lower()
    device = device.lower()

    if float_precision not in ("float32", "float64"):
      raise ValueError(f"Unknown float precision: {float_precision!r}")
    if int_precision not in ("int32", "int64"):
      raise ValueError(f"Unknown int precision: {int_precision!r}")
    if device not in ("cpu", "cuda"):
      raise ValueError(f"Unknown device: {device!r}")

    float_dtype = _DTYPES[float_precision]
    int_dtype = _DTYPES[int_precision]

    self.float_precision = float_precision
    self.int_precision = int_precision
    self.device = Device.CPU if device == "cpu" else Device.CUDA
    self.float_dtype = float_dtype
    self.int_dtype = int_dtype
    self.verbose = verbose
    self.manapy_save_threads = 1
    self.verbose_save_local_domains = True
    self.gpu_aware_mpi = False

  @staticmethod
  def getDefaultConfig(**overrides) -> "ManapyConfig":
    """The ManapyConfig every test builds its domain with.

    Defaults to the float64/int64 CPU pair the reference tables were generated
    with, and stays overridable from the environment so the same suite can be run
    against another precision pair or on the GPU without editing a test:
        MANAPY_DEVICE=cuda pytest tests/
    """
    kwargs = {
      "float_precision": os.environ.get("MANAPY_FLOAT", "float32"),
      "int_precision": os.environ.get("MANAPY_INT", "int64"),
      "device": os.environ.get("MANAPY_DEVICE", "cpu"),
    }
    kwargs.update(overrides)
    return ManapyConfig(**kwargs)
        

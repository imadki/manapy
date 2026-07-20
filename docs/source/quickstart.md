# Quickstart

Advect a Gaussian blob across a 128×128 unit square and write VTK output for
ParaView:

```python
import numpy as np
from manapy.api import Mesh, AdvectionModel

mesh = Mesh.rectangle(bounds=((0, 1), (0, 1)), n=(128, 128), cell_type="quad")
phi = mesh.field("phi",
                 init=lambda x, y, z: np.exp(-((x - 0.25)**2 + (y - 0.5)**2) / 0.01),
                 limiter="vanalbada")
model = AdvectionModel(phi, mesh, velocity=(2.0, 0.0), cfl=0.8, order=2, scheme="upwind")
model.run(T=0.25, output_every=10)
```

Save this as `quickstart.py` and run it:

```bash
python3 quickstart.py             # serial
mpirun -n 4 python3 quickstart.py # parallel — no code change
```

The run writes VTK files that can be opened in ParaView. The same script runs on
the GPU backend when manapy is configured with CUDA.

See {doc}`models` for the other physics models and {doc}`examples` for the full
set of runnable 2D/3D cases.

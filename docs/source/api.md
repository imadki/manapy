# API reference

The public API is exposed from `manapy.api`.

```{note}
The reference below is generated from the source docstrings with Sphinx
`autodoc`. Building it requires the compiled/runtime dependencies to be either
installed or mocked (see `autodoc_mock_imports` in `docs/source/conf.py`).
```

## Mesh

```{eval-rst}
.. autoclass:: manapy.api.Mesh
   :members:
   :undoc-members:
   :show-inheritance:
```

## Models

```{eval-rst}
.. autoclass:: manapy.api.AdvectionModel
   :members:
   :show-inheritance:

.. autoclass:: manapy.api.DiffusionModel
   :members:
   :show-inheritance:

.. autoclass:: manapy.api.PoissonModel
   :members:
   :show-inheritance:

.. autoclass:: manapy.api.DarcyModel
   :members:
   :show-inheritance:
```

## Mesh generation

```{eval-rst}
.. automodule:: manapy.api.meshgen
   :members:
   :undoc-members:
```

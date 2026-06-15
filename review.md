# Code Review

## Findings

1. **Python version metadata is wrong.**
   `pyproject.toml` annonce `requires-python = ">=3.8"`, mais le code utilise `tuple[...]`, `dict[...]` et surtout `A | B`, par exemple dans `manapy/solvers/advec/system.py` et `manapy/solvers/streamer/system.py`. Cette syntaxe casse au moins Python 3.8, et `|` nécessite Python 3.10. Il faut soit monter `requires-python` à `>=3.10`, soit revenir à `typing.Tuple` / `typing.Union`.

2. **Bug runtime dans `ShallowWaterSolver` avec diffusion active.**
   `manapy/solvers/shallowater/system.py` définit `self.diffusion = False` seulement si `Dxx == Dyy == 0`; sinon l'attribut n'existe pas. Plus bas, `if self.diffusion:` peut lever `AttributeError`. Initialiser explicitement `self.diffusion = not (self.Dxx == self.Dyy == 0)`.

3. **La suite de tests contient un test casse.**
   `pytest -q tests/variable/test_gradient.py` donne `1 failed, 3 passed`. Le test appelle `_check_cell_gradient(domain, a=0, b=1, p=0)`, mais la fonction declaree attend `(domain, fun, atol=...)`. Le test ne teste donc pas le gradient lineaire comme prevu.

4. **Fuite de references dans l'extension C.**
   Dans `manapy/c_api/src/py_manapy_part.cpp`, les fonctions de partitionnement construisent `part_array`, puis retournent `Py_BuildValue("O", part_array)`. `"O"` incremente la refcount et le code ne decremente pas `part_array` au chemin succes. Retourner directement `part_array`, ou utiliser `"N"`, serait plus correct. En plus, le chemin d'erreur de `py_make_n_part_mesh_nodal` appelle `free(part_array)`, ce qui est incorrect pour un `PyObject *`, et ne libere pas `part_vert` si la creation NumPy echoue.

5. **Les caches de domaines sont relatifs au cwd et collisionnent entre maillages.**
   `DomainClass.py` et `LocalDomainInterface.py` utilisent `local_domain_{size}` sans inclure le chemin du maillage, la methode de partitionnement, la dimension ni la precision. Deux simulations differentes avec le meme nombre de rangs peuvent reutiliser ou supprimer les fichiers l'une de l'autre si `recreate=False`.

6. **Les BC partielles peuvent casser a l'execution.**
   Si l'utilisateur passe un dict `BC` incomplet, seules les cles fournies sont remplies. Les autres restent `None`, mais `update_ghost_value()` et `LinearSolver.update_ghost_values()` iterent ensuite sans garde `None`.

7. **Code bibliotheque qui fait `sys.exit`.**
   `ScipySolver.py` et `PETScKrylovSolver.py` quittent le process si une dependance optionnelle manque. Pour une bibliotheque, il vaut mieux lever `ImportError` ou `RuntimeError` avec message clair, sinon un import ou constructeur peut tuer une application hote.

8. **Packaging incomplet pour les maillages/helpers.**
   `manapy/helpers/mesh_files.py` pointe vers `../../meshes` et `../../tests/data/meshes`, mais `pyproject.toml` ne package que `manapy*`. Dans une wheel installee, `get_mesh()` risque de pointer vers des fichiers absents.

## Verification

- `python3 -m py_compile manapy/domain/DomainClass.py manapy/domain/MeshClass.py manapy/domain/PartitioningClass.py manapy/core/Variable.py manapy/boundary/Boundary.py manapy/comms/NeighborCommunication.py manapy/backends/gpu/gpu_backend.py` passed.
- `pytest -q` started green, then produced many errors around 70%; it was stopped after prolonged blocking.
- `pytest -q tests/variable/test_gradient.py` reproduced a concrete failure: `1 failed, 3 passed`.

No source files were modified for this review.

# Rapport comparatif — `main` (API stable) vs `manapy-1.0` (refonte)

> Mis à jour le 2026-06-12, **après `git pull`** (HEAD `34db2e1`).
> `main` = 5 commits depuis l'ancêtre commun ; `manapy-1.0` = **125 commits** (était 93
> avant le pull). L'écart entre les deux versions s'est nettement réduit.

## 0. Ce qui a changé depuis la version précédente du rapport

| Point | Avant le pull | Après le pull (`34db2e1`) |
|---|---|---|
| Migration solveurs/exemples vers le nouveau cœur | non faite (imports `manapy.ast/base/ddm`) | ✅ **terminée** dans les sources suivies |
| Bug `BCneumannNH` (`core/Variable.py`) | présent | ✅ **corrigé** |
| Packaging | `setup.py`, `numpy==2.4.3` figé, build `c_api` non intégré | ✅ `pyproject.toml`, `numpy>=2.0,<3`, partitionneur = paquet externe `manapy-part` |
| Kernels solveurs | `fvm_utils.py` / `system.py` monolithiques | 🆕 split `*_compute.py` + classes |
| Bug `create_domain` série | présent | ⚠️ **partiellement présent** |
| Bug `NeighborCommunication` série | présent | ⚠️ **inchangé** |

## 1. Résumé exécutif

`manapy-1.0` reste une **réécriture du cœur** par rapport à `main`, mais la migration est
désormais **largement aboutie** :

- **Solveurs & exemples portés** : aucune source suivie n'utilise plus l'ancienne API
  (`manapy.ast`, `manapy.base`, `manapy.ddm`, `manapy.partitions`). Les imports legacy ne
  subsistent que dans du scratch / venv non versionné (`#system.py#`, `myvenv/`, `darcy2d_bis.py`…).
- **Partitionnement** : METIS/GKlib n'est plus vendoré dans le dépôt principal mais **externalisé
  en paquet publié** `manapy-part` (dépendance `pyproject.toml`).
- **Configuration** : constantes de compilation globales (`backends/types.py`) au lieu du système
  runtime `Struct`/`process_conf`/`CPUBackend`.

## 2. Correspondance des modules

| `main` (ancien) | `manapy-1.0` (nouveau) | Nature du changement |
|---|---|---|
| `ddm/` (domain.py 961, utils2d/3d ~1700) | `domain/` (DomainClass, LocalDomain, Mesh, Partitioning, domain_compute 1619…) | **Réécrit + éclaté** |
| `partitions/` (mgmetis) | paquet externe **`manapy-part`** + `domain/PartitioningClass` | **Externalisé** (natif C++) |
| `ast/` (core.py, functions2d/3d) | `core/` (Variable, variable_compute_2d/3d, utils) | **Réécrit**, renommé |
| `base/` (`Struct`, `make_get_conf`) | — (supprimé) | **Abandonné** |
| `backends/` + `backends/cpu/` (`CPUBackend`) | `backends/` (compile_fun, types) | **Simplifié** |
| `comms/communication.py` | `comms/NeighborCommunication.py` | **Refactoré** (même technique MPI) |
| `solvers/*/system.py` + `fvm_utils.py` | `solvers/*/system.py` + **`*_compute.py`** ; `ls/` éclaté en classes | **Migré** vers le nouveau cœur |
| (quasi rien) | `testing/` + `tests/domain/` + `tests/variable/` | **Ajout majeur** |
| — | `helpers/` (`mesh_files`, `get_mesh`) | **Nouveau** |
| — | `LocalDomainInterface` (save/load hdf5) | **Nouveau** |
| `setup.py` seul | `setup.py` + **`pyproject.toml`** | **Packaging modernisé** |

## 3. Différences architecturales par axe

### Partitionnement
- **main** : `MeshPartition` → `mgmetis.metis.part_mesh_dual` à l'exécution (Python pur, `mgmetis`
  commenté dans `setup.py` → installation fragile).
- **1.0** : partitionneur natif **publié séparément** (`manapy-part`), piloté par
  `PartitioningClass`. Build géré par pip, dépôt principal allégé.

### Domaine
- **main** : `ddm/domain.py` monolithe (961 lignes), tout en mémoire à chaque run.
- **1.0** : `Mesh` / `Partitioning` / `LocalDomain` / `Domain` / `LocalDomainInterface`
  (**persistance hdf5** → partitionner une fois, rejouer N fois). Modulaire et testable.

### Backend & précision
- **main** : `CPUBackend` configurable au runtime (`float_precision`/`int_precision`, signatures
  numba patchées dynamiquement). Abstraction multi-hardware.
- **1.0** : `FLOAT_TYPE`/`INT_TYPE` figés dans `backends/types.py` (float32/int32 par défaut),
  `compile_fun.compile` simple avec **cache par hash de source**. Plus lisible, mais
  **configurabilité runtime perdue**.

### Variable
- **main** : `Variable(domain, terms, comm, name, BC, values, *args)`, lit backend/précision
  depuis le domaine.
- **1.0** : `Variable(domain, BC, values_dict, name)` épurée, branchée sur `core/`. Régression
  mineure : `__add__`/etc. recréent une `Variable` complète (réallocations + dépendance aux BC).

### Solveurs
- **main** : chaque solveur a un `system.py` + `fvm_utils.py`, dépendant de `ast`/`base`/`CPUBackend`.
- **1.0** : même découpage mais **kernels isolés dans `*_compute.py`** et compilés via
  `backends.compile_fun` ; `ls/` désormais éclaté en classes (`LinearSolver`, `MUMPSSolver`,
  `PETScKrylovSolver`, `ScipySolver`).

### Communication MPI
- Les **deux** utilisent `Create_dist_graph_adjacent` + `Neighbor_alltoallv`. **1.0** l'encapsule
  proprement dans `NeighborCommunication` avec cache des `counts/displs`.

### Tests
- **main** : quasi aucun test.
- **1.0** : `tests/domain/` (validation contre tables de référence : triangles, rectangles,
  tétraèdres, cuboïdes, hybrides 2D/3D) + `tests/variable/` (gradient, laplacien).

## 4. Bilan : gains et dette restante

### Gains de `manapy-1.0` (consolidés après pull)
- Architecture domaine modulaire et testée.
- Partitionneur natif externalisé (`manapy-part`), build pip propre.
- Packaging modernisé (`pyproject.toml`, dépendances bornées au lieu de figées).
- Solveurs/exemples migrés et cohérents avec le nouveau cœur.
- Persistance des sous-domaines (partition once / run many).

### Dette restante
- **Précision/backend non configurables au runtime** (vs `CPUBackend`).
- **Bug #2** : branche série de `create_domain` (`domain/DomainClass.py:223-225`) `print` et
  retourne `None` au lieu de `raise`/`Abort` (incohérent avec la branche parallèle `comm.Abort(1)`).
- **Bug #3** : `NeighborCommunication.exchange()` ignore `recv_buffer` en `size==1` ;
  `immediate_exchange()` ne renvoie pas une *MPI request* en série.
- Email auteur `kissami.imad@gmail.ma` (typo `.ma`) dupliqué dans `setup.py` et `pyproject.toml`.
- `setup.py` et `pyproject.toml` coexistent (sources de vérité à unifier).
- Arbre de travail encombré de scratch non versionné (`myvenv/`, `#system.py#`, `sanstitre1.py`…).

## 5. Recommandations restantes

1. **Uniformiser la gestion d'erreur** de `create_domain` (Abort/raise des deux côtés).
2. **Harmoniser les contrats série/parallèle** de `NeighborCommunication`.
3. **Choisir une seule source de packaging** (idéalement tout dans `pyproject.toml`) et corriger
   l'email.
4. **Documenter** le choix « constantes globales » vs configurabilité runtime, ou réintroduire un
   réglage de précision.
5. **Vérifier la non-régression numérique** des kernels migrés (`ast/functions2d.py` de `main`
   vs `core/variable_compute_2d.py` + `solvers/*/\*_compute.py` de `1.0`).
6. **Nettoyer l'arbre** et renforcer `.gitignore` (venv, autosaves, fichiers scratch).

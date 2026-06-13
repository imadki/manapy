# Handoff — session manapy (branche `manapy-1.0`)

> Dernière mise à jour : 2026-06-13. Branche : `manapy-1.0`. Aucun commit effectué
> (tout est en **modifications locales**, par choix de l'utilisateur).

## 1. État de l'environnement (important)

| Élément | État | Notes |
|---|---|---|
| numpy | **1.26.4** | ⚠️ **rétrogradé** depuis 2.4.6 par `pip install petsc petsc4py` (petsc4py exigeait `numpy<2`) |
| manapy-part | **1.0.1**, recompilé | rebuild local float64 OK ; `~/.local/lib/python3.12/site-packages/manapy_part*_*.so` |
| petsc4py | 3.22.2 (système) | fonctionne sous numpy 1.26.4 |
| FLOAT_TYPE | **float64** (`manapy/backends/types.py`) | défaut voulu par l'utilisateur |

- Toute la pile tourne sous numpy 1.26.4 (advection + laplacien/PETSc testés OK).
- `manapy_part` est compilé contre numpy 2.4.6 mais tourne sur 1.26.4 (testé, OK, mais
  techniquement ABI-non-supporté → si besoin de propreté : recompiler contre 1.26.4 via
  `pip install --user --break-system-packages --force-reinstall ./manapy/c_api`).
- Installation : PEP668 actif → utiliser `--break-system-packages` (ou un venv).

## 2. Modifications locales NON commitées

### Corrections de bugs (cœur) — à conserver
- **`manapy/comms/NeighborCommunication.py`** (Bug #3) : `exchange` honore désormais
  `recv_buffer` quand pas de voisin et ne renvoie plus une copie des données locales ;
  `immediate_exchange` renvoie `MPI.REQUEST_NULL` au lieu d'un tableau.
- **`manapy/domain/DomainClass.py`** (Bug #2) : branche série de `create_domain` fait
  désormais `raise` après le print (au lieu de retourner `None`).
- **`manapy/domain/vtk_writer.py`** : Points VTK écrits en `types.np_float_type` (au lieu de
  `np.float64` codé en dur) → corrige le `.pvtu` qui affichait un maillage noir dans ParaView
  (incohérence de type Points entre `.vtu` et `.pvtu`).

### Réglages d'exemples (tests de l'utilisateur — ne pas committer tels quels)
- `manapy/examples/2D/advection2d.py` : `order=2`.
- `manapy/examples/2D/laplacien2d.py` : PETScKrylovSolver activé (MUMPS commenté).
- `manapy/examples/2D/shallow_water2d.py` : `order=1`.

### Fichiers non versionnés générés
- `meshes/big/carre.msh` : régénéré via `gmsh meshes/geo/carre.geo -2 -format msh2 -o meshes/big/carre.msh`
  (le dossier `meshes/big/` n'est pas dans le dépôt).
- `RAPPORT_COMPARAISON.md` : rapport comparatif `main` vs `manapy-1.0`.
- `handoff.md` : ce fichier.

## 3. Problème OUVERT — Shallow Water diverge (terme source SRNH / orientation)

**Symptôme** : `examples/2D/shallow_water2d.py` (dam-break, fond plat `Z=0`) diverge :
`h` passe de 5 à >40 puis NaN, `hu` explose. Aux deux ordres (ordre 2 plus rapide vers NaN).

**Diagnostic confirmé** :
- Source **commenté** (`src_hu=src_hv=0`) → **stable** (`h` reste ≤ 5). Donc le coupable est
  `_term_source_srnh_SW` dans `manapy/solvers/shallowater/fvm_utils_compute.py`.
- Tous les triangles du nouveau domaine sont **orientés horaire** (aire signée < 0), alors que
  le terme source SRNH est écrit pour une orientation **anti-horaire (CCW)** → perte de la
  C-propriété → quantité de mouvement parasite au bord.
- Le terme source est **identique** à celui de `main` (qui marchait), car l'ancien domaine
  fournissait les mailles dans l'ordre CCW attendu.

**Option B (corriger uniquement le terme source) = NON VIABLE par ré-indexation** :
le terme fait une reconstruction par règle de Cramer (`delta`, `deltax`, `deltay`, `deltaz`)
avec des **motifs de lignes codés en dur** (ex. `c_3 = 3*h_1p` en row 0). Permuter les indices
de face réordonne les entrées mais pas ces motifs → `delta` change de signe sans que `deltax`
suive → reconstruction corrompue. Deux tentatives de relabeling local (`swap 0↔1` puis
`jj=2-j`, `b=node2`) ont échoué (divergence identique).

**Recommandation : Option A** — ré-orienter les triangles 2D en **CCW à la construction du
domaine**, juste après le chargement de `self.cells` et **avant `_create_info`** dans
`manapy/domain/LocalDomainClass.py` (ainsi `cell_faceid`/`cell_nf` suivent). Pour chaque triangle
d'aire signée < 0, échanger deux nœuds. Ça réaligne le domaine sur la convention que **tout le
code porté depuis `main` attend déjà** (donc faible risque pour les autres solveurs).
Valider avec le test SW : `h` doit rester ≤ ~5, pas de NaN.

Alternative (lourde) : réécrire la reconstruction du terme source pour qu'elle soit
géométriquement intrinsèque (indépendante de l'orientation).

Note : `_term_source_srnh_SW` est **triangle-only** (boucle `range(3)`).

## 4. Bug A (à part, réel mais pas la cause de la divergence SW)

Dans `manapy/solvers/shallowater/system.py` (`explicit_convective`), `h` est reconstruit avec
ses propres gradients **mais le limiteur de `hc`** (`self.hc.psi`/`psihalo`) est passé au schéma ;
`h.psi` est calculé (via `h.compute_cell_gradient()`) puis jeté. Devrait probablement utiliser
`h.psi` pour `h`. À corriger après le problème d'orientation.

## 5. Vérifs déjà faites cette session
- Reproductibilité parallèle ordre 2 (advection) : écart série/20-procs ~1e-15 en float64
  (était ~1e-6 en float32) → **non-associativité flottante normale**, pas un bug.
- petsc laplacien2d : tourne et résout (warnings PETSc d'options inutilisées, sans gravité).

## 6. Prochaines étapes suggérées
1. Implémenter **Option A** (ré-orientation CCW des triangles au domaine) + valider SW.
2. Corriger **Bug A** (passer `h.psi` pour `h`).
3. Décider du sort de numpy : rester en 1.26.4 (tout marche) ou viser numpy 2 + un petsc4py
   compatible numpy 2 (≥ 3.23 réellement buildé pour numpy 2).
4. Décider quoi committer parmi les 3 corrections de bugs du §2.

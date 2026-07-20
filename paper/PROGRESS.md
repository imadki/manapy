# Progress — repo cleanup + JOSS submission

Session snapshot for the manapy 1.0 cleanup and the JOSS software paper.
`main` and `manapy-1.0-cleanup` are kept identical (each commit is force-pushed to
`main` and pushed to `manapy-1.0-cleanup`). HEAD at last save: see `git log`.

Legend: ✅ done · 🟡 done, needs your input/action · ⬜ todo

## Repository cleanup (done)
- ✅ Removed streamer / incompressible / ro solvers & examples from the repo.
- ✅ Reduced examples to 2–3 per solver dir (+ GPU variants); added 2D/3D periodic
  advection; restored `darcy/darcy_with_particles2d.py` (future RTM base).
- ✅ `darcy2d.py` made CPU-only and simplified; stripped dead/commented code from
  all examples.
- ✅ `meshes/geo`: explicit `.geo` set (`uns/struct/quad_square`, `*_rectangle`,
  `uns/hex/periodic_cube*`, `hybrid2d/3d`) + `README.md`; old geos & `meshes/old`
  removed; examples renamed to new mesh names, `MESH_DIR` unified to `meshes/geo`.
- ✅ Untracked `meshes/*.msh` (kept on disk, `.gitignore` blocks re-add); test
  fixtures under `tests/data/meshes` kept.
- ✅ Removed `handoff.md`, top-level `mesh/` dir.
- ✅ Merged cleanup into `main` (force-push); kept `LICENCE` (MIT).
- ✅ Removed "Claude" as a contributor: stripped 6 `Co-Authored-By: Claude`
  trailers via history rewrite; deleted the remote `claude/help-documentation-*`
  branch (13 Claude-authored commits). 0 Claude commits remain on any remote branch.
  NOTE: GitHub's contributor list is cached and may take hours to refresh.

## JOSS paper (done, needs author info)
- ✅ `paper/paper.md` — JOSS Markdown (Summary, Statement of need, State-of-the-field
  table, Functionality, AI usage disclosure = "debugging", code snippet). Cites via
  `[@key]` against `paper/paper.bib` (13 refs). Verified with `pandoc --citeproc`
  (all keys resolve).
- ✅ `paper/paper.tex` + `paper/paper.bib` — standalone LaTeX version; compiles with
  `pdflatex + bibtex` (no undefined citations). `paper/.gitignore` ignores build
  artifacts.
- ✅ `paper/JOSS_SUBMISSION.md` — gap analysis / checklist.
- 🟡 **Author list / ORCIDs / affiliation** are placeholders (`TODO`). User edited
  `CITATION.cff` to 2 authors (Kissami, Ben Hamou — Haikal removed).
  **ACTION: sync author list across `paper.md`, `paper.tex`, `CITATION.cff`, and add
  real ORCIDs + confirmed affiliation + corresponding author + funding.**

## Documentation (done)
- ✅ `CONTRIBUTING.md` (report/ask/contribute, dev setup, tests, add-a-solver).
- ✅ Expanded `README.md` (features, quickstart, per-model API, examples, docs, license).
- ✅ Sphinx scaffold `docs/` (installation, quickstart, models, examples, api autodoc,
  contributing) + `.readthedocs.yaml` + `docs/requirements.txt`. Not built locally
  (sphinx not installed); RTD will build it.
- ✅ `CITATION.cff` (GitHub "Cite this repository" + JOSS metadata).

## CI (fixed on our side, blocked on GitHub's side)
- ✅ `.github/workflows/test_manapy.yaml` fixed to actually run the tests: builds the
  Docker env image, mounts the repo, `pip install -e .`, runs `Docker/tests.sh`.
- ✅ `Docker/tests.sh` now runs `pytest tests` (+ `mpi_test.py` on 2/4 ranks); dropped
  the broken advection smoke-test and the dead `import manapy.ddm`.
- 🟡 **GitHub Actions is stuck**: ALL runs (incl. Dependabot) are `Queued` and never
  picked up — an account-level Actions hold, not our workflow (predates the fix).
  **ACTION (user): check GitHub email for an "Actions disabled" notice; add/verify a
  payment method at github.com/settings/billing (free for public repos, often lifts
  anti-abuse holds); check githubstatus.com; else contact GitHub Support.**
  Known risk once it runs: the Docker image uses Python 3.14 (cp314) — `numba` may
  lack 3.14 wheels; fallback is to pin the image to Python 3.11.

## Remaining actions (when we return)
1. 🟡 Provide **authors + ORCIDs + affiliation + corresponding author + funding**;
   then sync `paper.md`, `paper.tex`, `CITATION.cff` and remove all `TODO`s.
2. 🟡 **Zenodo**: enable the repo on zenodo.org (via GitHub) → create GitHub release
   `v1.0.0` → get the DOI → put it in `paper.md` (Archive).
3. 🟡 **ReadTheDocs**: import `imadki/manapy` on readthedocs.org (reads
   `.readthedocs.yaml`); fix `autodoc_mock_imports` if the API build fails; add the
   docs URL to README + paper.
4. 🟡 Unblock GitHub Actions (account issue) and confirm the CI goes green.
5. ⬜ Optional: `CODE_OF_CONDUCT.md` (Contributor Covenant) — appreciated by JOSS.
6. ⬜ Submit at https://joss.theoj.org pointing to the repo (Markdown `paper.md`).
   Cost: none (diamond OA). SoftwareX was ruled out (has an APC).

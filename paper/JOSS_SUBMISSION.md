# JOSS submission — status & checklist

Draft paper: `paper/paper.md` (+ `paper/paper.bib`). JOSS reviews the **software**,
not just the paper, so most of the work is making the repository review-ready.

## Hard requirements

| Requirement | Status | Action |
|---|---|---|
| OSI-approved open-source license | ✅ MIT (`LICENCE`) | — |
| Public version-controlled repo | ✅ GitHub | — |
| `paper.md` (≤ ~1000 words) + `paper.bib` | ✅ drafted | fill author list, ORCIDs, affiliation, date |
| Substantial scholarly effort | ✅ (multi-solver FV framework, CPU/GPU, MPI) | — |
| Statement of need | ✅ in paper | review wording |
| Installation instructions | ✅ README `## Install` | — |
| Example usage | ✅ `manapy/examples/` | link them from README |
| API / functionality documentation | ⚠️ partial | add a usage/API doc (extended README or docs site) |
| Automated tests | ⚠️ `tests/` exist | document how to run (`pytest`), report coverage |
| Continuous integration | ❌ none found | add GitHub Actions running the tests |
| Community guidelines (`CONTRIBUTING`) | ❌ missing | add `CONTRIBUTING.md` (contribute / report / seek support) |
| Archived release with DOI | ❌ | tag `v1.0.0`, archive on Zenodo, put DOI in metadata |

## Before submitting

1. **Confirm authorship** — everyone with a significant contribution, in order, each
   with an ORCID. (git history shows: Imad Kissami, Ayoub Ben Hamou, Mouad Haikal.)
2. **Add `CONTRIBUTING.md`** — how to contribute, report issues, and get support.
3. **Add CI** — a GitHub Actions workflow that installs manapy and runs `pytest`.
4. **Flesh out docs** — a short "Usage" section linking each example to the model it
   demonstrates; document the public API (`Mesh`, `*Model`, `meshgen`).
5. **Tag + archive** — create a GitHub release `v1.0.0`, connect the repo to Zenodo,
   and record the resulting DOI (JOSS needs it at acceptance).

## Submission

- Submit at <https://joss.theoj.org> by pointing to the GitHub repo (paper on the
  default branch). Review happens openly in a GitHub issue on the JOSS side.
- Cost: **none** (diamond open access).

## Notes / TODO in the draft

- `paper.md` YAML has `TODO` placeholders: ORCIDs, affiliation confirmation, date.
- The paper describes only solvers shipped in the repo (advec, advecdiff, burgers,
  diffusion, euler, ls, shallowater, swmhd). If incompressible/multilayer are to be
  part of the JOSS release, re-add and document them first.
- Double-check every `paper.bib` entry (DOIs/authors) before submission.

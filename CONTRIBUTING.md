# Contributing to manapy

Thanks for your interest in `manapy`! Contributions of all kinds are
welcome — bug reports, documentation, examples, new solvers, and performance
improvements. This document explains how to get help, report problems, and
submit changes.

## Getting help / asking questions

- **Questions and usage help:** open a [GitHub
  Discussion](https://github.com/imadki/manapy/discussions) or an issue with the
  `question` label.
- **Contact:** for anything that does not fit an issue, email the maintainers
  (see the corresponding author in the repository / paper).

## Reporting bugs

Open an issue at <https://github.com/imadki/manapy/issues> and include:

- what you expected to happen and what happened instead;
- a **minimal, runnable example** (mesh + a few lines of Python) that reproduces
  the problem;
- your environment: OS, Python version, `manapy` version (`manapy.__version__`),
  and the number of MPI ranks / whether a GPU backend was used;
- the full error message / traceback.

Please search existing issues first to avoid duplicates.

## Requesting features

Open an issue describing the use case and, if possible, a sketch of the API you
have in mind. For a new conservation law / solver, see *Adding a solver* below.

## Development setup

```bash
git clone https://github.com/imadki/manapy.git
cd manapy
python3 -m pip install -e ".[dev]"     # editable install + test dependencies
```

`manapy` requires a working MPI runtime for `mpi4py`; the GPU backend
additionally requires a CUDA toolchain. See the `README.md` for details and the
optional PETSc / MUMPS setup guides under `tools/`.

## Running the tests

```bash
python3 -m pytest tests                 # serial test suite
mpirun -n 4 python3 -m pytest tests     # exercise the parallel paths
```

Please make sure the test suite passes before opening a pull request, and add
tests for any new behavior or bug fix.

## Submitting changes (pull requests)

1. Fork the repository and create a topic branch from `main`
   (`git checkout -b my-feature`).
2. Make your change, keeping edits focused and consistent with the surrounding
   code style.
3. Add or update tests and documentation as needed.
4. Run the test suite locally.
5. Push your branch and open a pull request against `main`, describing the
   motivation and summarizing the change.

Small, self-contained pull requests are easier to review and merge. For larger
changes, please open an issue first to discuss the approach.

## Adding a solver

`manapy` is designed so that a new solver reuses the existing finite-volume
machinery (fluxes, gradients, boundary conditions, linear solvers) rather than
re-implementing it. See
[`manapy/solvers/ADDING_A_SOLVER.md`](manapy/solvers/ADDING_A_SOLVER.md) for a
step-by-step guide, and use the existing solvers (`advec`, `diffusion`, `euler`,
`shallowater`, …) and the `manapy/examples` cases as templates.

## Coding conventions

- Follow the style of the surrounding code (naming, indentation, comment
  density).
- Keep numerical kernels backend-agnostic where possible so they run on both the
  CPU (Numba) and GPU (CUDA) backends.
- Document public functions and models with clear docstrings, and add a runnable
  example under `manapy/examples/` for new user-facing functionality.

## License

By contributing, you agree that your contributions will be licensed under the
same [MIT License](LICENCE) that covers the project.

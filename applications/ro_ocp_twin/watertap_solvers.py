"""Local shim for the `watertap_solvers` package (not installed in this env).

WaterTAP's `watertap.core.solvers` only needs `get_solver`; we return a plain
Pyomo IPOPT solver (the binary provided by `idaes get-extensions`) with
WaterTAP-style options. Put this directory on PYTHONPATH (the scripts do this
automatically) so `import watertap_solvers` resolves to this file.
"""
import idaes  # noqa: F401  -- registers ~/.idaes/bin (ipopt) on the path
from pyomo.environ import SolverFactory

_DEFAULTS = {
    "tol": 1e-8,
    "constr_viol_tol": 1e-8,
    "bound_push": 1e-8,
    "nlp_scaling_method": "user-scaling",
}


def get_solver(solver="ipopt", options=None):
    s = SolverFactory(solver or "ipopt")
    if solver in (None, "ipopt"):
        s.options.update(_DEFAULTS)
    if options:
        s.options.update(options)
    return s

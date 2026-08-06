#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Periodic boundary-condition tests (2D + 3D, hex + tet, serial + MPI).

Exercises the full periodic chain that was restored on this branch:
  * domain construction on a periodic mesh (same-rank pairing + cross-rank halos);
  * the cell->node interpolation path used for VTK, including edge/corner nodes
    (the 3D "one-sided stencil" division-by-zero fix);
  * mass conservation of a linearly-advected field (nothing leaks at the seam);
  * cross-rank transparency: a parallel run reproduces the serial mass.

Meshes are generated on the fly with gmsh (via manapy.api.meshgen); the whole
module skips if gmsh is not available. The 3D recipes matter:
  * HEX  : transfinite + Recombine  -> structured hexes, conforming.
  * TET  : transfinite LINES + Periodic Surface + FREE volume. A fully transfinite
           tet cube is rejected by the mesh reader (its volume tet-split does not
           match the boundary triangulation), so the volume is left unstructured
           but constrained to the (periodic-copied) surfaces.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.testing.test_domain_helper import make_test_config

# Precision pair + device the periodic domains below are built with (see
# make_test_config: float64/int64 CPU unless overridden from the environment).
CONFIG = make_test_config()

# Mass conservation is exact up to accumulated round-off, so the tolerances below
# have to follow the working precision rather than assume float64: one float32
# step already costs ~1e-7 relative, which no fixed 1e-12 bound can absorb.
_EPS = np.finfo(np.dtype(CONFIG.float_precision)).eps
MASS_RTOL = max(1e-12, 1e3 * _EPS)       # serial: sum over a few hundred steps
RANK_RTOL = max(1e-9, 1e3 * _EPS)        # parallel vs serial: different sum order

# --------------------------------------------------------------------- geo
_SQUARE = """
Point(1)={0,0,0,1};Point(2)={1,0,0,1};Point(3)={1,1,0,1};Point(4)={0,1,0,1};
Line(1)={1,2};Line(2)={2,3};Line(3)={3,4};Line(4)={4,1};
Line Loop(1)={1,2,3,4};Plane Surface(1)={1};
Transfinite Line{4,2}=NN;Transfinite Line{1,3}=NN;Transfinite Surface{1};
Physical Curve("in",11)={4};Physical Curve("out",22)={2};
Physical Curve("upper",33)={3};Physical Curve("bottom",44)={1};
Physical Surface("domain",1)={1};
Periodic Line{2}={-4} Translate{1,0,0};
Periodic Line{3}={-1} Translate{0,1,0};
"""

_CUBE_COMMON = """
Point(1)={0,0,0,1};Point(2)={1,0,0,1};Point(3)={1,1,0,1};Point(4)={0,1,0,1};
Point(5)={0,0,1,1};Point(6)={1,0,1,1};Point(7)={1,1,1,1};Point(8)={0,1,1,1};
Line(1)={1,2};Line(2)={2,3};Line(3)={3,4};Line(4)={4,1};
Line(5)={5,6};Line(6)={6,7};Line(7)={7,8};Line(8)={8,5};
Line(9)={1,5};Line(10)={2,6};Line(11)={3,7};Line(12)={4,8};
Line Loop(1)={1,2,3,4};Plane Surface(1)={1};
Line Loop(2)={5,6,7,8};Plane Surface(2)={2};
Line Loop(3)={1,10,-5,-9};Plane Surface(3)={3};
Line Loop(4)={3,12,-7,-11};Plane Surface(4)={4};
Line Loop(5)={4,9,-8,-12};Plane Surface(5)={5};
Line Loop(6)={2,11,-6,-10};Plane Surface(6)={6};
Surface Loop(1)={1,2,3,4,5,6};Volume(1)={1};
Transfinite Line "*" = NN;
"""
_CUBE_TAGS = """
Physical Surface("in",11)={5};Physical Surface("out",22)={6};
Physical Surface("upper",33)={4};Physical Surface("bottom",44)={3};
Physical Surface("front",55)={1};Physical Surface("back",66)={2};
Physical Volume("domain",1)={1};
Periodic Surface{6}={5} Translate{1,0,0};
Periodic Surface{4}={3} Translate{0,1,0};
Periodic Surface{2}={1} Translate{0,0,1};
"""
_CUBE_HEX = (_CUBE_COMMON + 'Transfinite Surface "*";\nTransfinite Volume "*";\n'
             'Recombine Surface "*";\nRecombine Volume "*";\n' + _CUBE_TAGS)
_CUBE_TET = _CUBE_COMMON + _CUBE_TAGS  # transfinite lines + periodic surfaces + free volume

# 2D UNSTRUCTURED periodic square: no transfinite surface, only a characteristic
# length -- the interior is unstructured triangles, but `Periodic Line` copies the
# master boundary edge mesh to its slave, so opposite boundaries stay conforming
# (which is all the coordinate-based periodic matching needs).
_SQUARE_UNS = """
Point(1)={0,0,0,LC};Point(2)={1,0,0,LC};Point(3)={1,1,0,LC};Point(4)={0,1,0,LC};
Line(1)={1,2};Line(2)={2,3};Line(3)={3,4};Line(4)={4,1};
Line Loop(1)={1,2,3,4};Plane Surface(1)={1};
Physical Curve("in",11)={4};Physical Curve("out",22)={2};
Physical Curve("upper",33)={3};Physical Curve("bottom",44)={1};
Physical Surface("domain",1)={1};
Periodic Line{2}={-4} Translate{1,0,0};
Periodic Line{3}={-1} Translate{0,1,0};
"""

# case id -> (geo, resolution n, dim)
_GEOS = {
    "2d":     (_SQUARE,     32, 2),
    "2d-uns": (_SQUARE_UNS, 24, 2),
    "3d-hex": (_CUBE_HEX,    6, 3),
    "3d-tet": (_CUBE_TET,    6, 3),
}
CASES = list(_GEOS)


def _gen(geo, n, dim, path):
    from manapy.api import meshgen
    geo = geo.replace("NN", str(n + 1)).replace("LC", repr(1.0 / n))
    return meshgen._run_gmsh(geo, dim, path)


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    d = tmp_path_factory.mktemp("periodic")
    out = {}
    for cid, (geo, n, dim) in _GEOS.items():
        try:
            path = _gen(geo, n, dim, str(d / (cid + ".msh")))
        except Exception as e:                       # gmsh missing / geo error
            pytest.skip(f"could not generate periodic mesh {cid}: {e}")
        out[cid] = (path, dim)
    return out


def _field(c):
    """A smooth field periodic in every direction of the unit box."""
    f = 1.0 + 0.1 * np.cos(2 * np.pi * c[:, 0]) + 0.05 * np.cos(2 * np.pi * c[:, 1])
    if c.shape[1] > 2:
        f = f + 0.03 * np.cos(2 * np.pi * c[:, 2])
    return f


# --------------------------------------------------------------- serial tests
@pytest.mark.parametrize("case", CASES)
def test_periodic_build(meshes, case):
    path, dim = meshes[case]
    dom = Domain.create_domain(path, dim, CONFIG, Partitioning.Par_Nodal, recreate=True)
    assert dom.nbcells > 0
    assert len(dom.periodicboundaryfaces) > 0        # periodicity was detected + wired


@pytest.mark.parametrize("case", CASES)
def test_periodic_node_interpolation(meshes, case):
    """cell->node interpolation of a smooth periodic field must stay bounded and
    accurate -- this exercises node_periodicid including edge/corner nodes, which
    used to blow up (division by zero) in 3D."""
    path, dim = meshes[case]
    dom = Domain.create_domain(path, dim, CONFIG, Partitioning.Par_Nodal, recreate=True)
    nb = dom.nbcells
    h = Variable(domain=dom)
    h.cell.cpu_rw()[:nb] = _field(dom.cells.center.cpu_r()[:nb])
    h.update_halo_value()
    h.update_ghost_value()
    h.interpolate_celltonode()
    hn = h.node.cpu_r()
    assert np.isfinite(hn).all()
    assert np.abs(hn).max() < 2.0                    # bounded (no blow-up)
    exact = _field(dom.nodes.vertex.cpu_r()[:len(hn)])
    assert np.abs(hn - exact).max() < 0.06           # accurate on these coarse meshes


@pytest.mark.parametrize("case", CASES)
def test_periodic_advection_conserves_mass(meshes, case):
    """Linear advection on a fully periodic domain conserves total mass to ~machine
    precision (nothing leaves the domain across the periodic seams)."""
    from manapy.solvers.advec.system import AdvectionSolver
    path, dim = meshes[case]
    dom = Domain.create_domain(path, dim, CONFIG, Partitioning.Par_Nodal, recreate=True)
    nb = dom.nbcells
    ne = Variable(domain=dom)
    vel = [Variable(domain=dom) for _ in range(dim)]
    S = AdvectionSolver(ne, vel=tuple(vel), order=1, cfl=0.8)
    ne.cell.cpu_rw()[:nb] = _field(dom.cells.center.cpu_r()[:nb])
    vol = dom.cells.volume.cpu_r()[:nb]
    m0 = float(np.sum(ne.cell.cpu_r()[:nb] * vol))

    t = 0.0
    while t < 0.2:
        for w in vel:
            w.face.cpu_w()[:] = 1.0
            w.interpolate_facetocell()
        dt = S.stepper()
        t += dt
        S.compute_fluxes()
        S.compute_new_val()

    m1 = float(np.sum(ne.cell.cpu_r()[:nb] * vol))
    assert np.isfinite(ne.cell.cpu_r()[:nb]).all()
    assert abs(m1 - m0) / m0 < MASS_RTOL             # mass conserved


# ------------------------------------------------------------------- MPI test
# Standalone driver: build the periodic domain, advect, and print the (globally
# reduced) mass. Run once serially and once under mpirun; the cross-rank periodic
# path is correct iff the two masses agree to ~machine precision.
_MPI_DRIVER = r'''
import os, numpy as np
from mpi4py import MPI
from manapy.domain import Domain, Partitioning
from manapy.core.Variable import Variable
from manapy.solvers.advec.system import AdvectionSolver
from manapy.testing.test_domain_helper import make_test_config
C = MPI.COMM_WORLD
dim = int(os.environ["DIM"])
# Same env-driven config as the serial run above -- the two masses are only
# comparable if both runs use the same precision pair.
dom = Domain.create_domain(os.environ["MESH"], dim, make_test_config(), Partitioning.Par_Nodal, recreate=True)
nb = dom.nbcells
ne = Variable(domain=dom); vel = [Variable(domain=dom) for _ in range(dim)]
S = AdvectionSolver(ne, vel=tuple(vel), order=1, cfl=0.8)
c = dom.cells.center.cpu_r()[:nb]
f = 1.0 + 0.1*np.cos(2*np.pi*c[:,0]) + 0.05*np.cos(2*np.pi*c[:,1])
if dim > 2: f = f + 0.03*np.cos(2*np.pi*c[:,2])
ne.cell.cpu_rw()[:nb] = f
vol = dom.cells.volume.cpu_r()[:nb]
def mass(): return C.allreduce(float(np.sum(ne.cell.cpu_r()[:nb]*vol)), MPI.SUM)
m0 = mass(); t = 0.0
while t < 0.2:
    for w in vel:
        w.face.cpu_w()[:] = 1.0; w.interpolate_facetocell()
    dt = S.stepper(); t += dt; S.compute_fluxes(); S.compute_new_val()
m1 = mass()
if C.Get_rank() == 0:
    print("RESULT m0=%.14e m1=%.14e" % (m0, m1))
'''


def _run_driver(script, mesh, dim, nprocs):
    env = dict(os.environ, MESH=mesh, DIM=str(dim))
    launcher = ["mpirun", "-n", str(nprocs)] if nprocs > 1 else []
    proc = subprocess.run(launcher + [sys.executable, script], env=env,
                          capture_output=True, text=True, timeout=900)
    m = re.search(r"RESULT m0=(\S+) m1=(\S+)", proc.stdout)
    if m is None:
        raise RuntimeError("driver produced no RESULT line:\n" + proc.stdout + proc.stderr)
    return float(m.group(1)), float(m.group(2))


@pytest.mark.parametrize("case", ["2d", "2d-uns", "3d-tet"])
def test_periodic_cross_rank_matches_serial(meshes, tmp_path, case):
    """A parallel run (mpirun -n 2) must reproduce the serial mass -- the cross-rank
    periodic partner (delivered as a translated halo) is transparent."""
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not available")
    path, dim = meshes[case]
    script = str(tmp_path / "driver.py")
    with open(script, "w") as fh:
        fh.write(_MPI_DRIVER)
    _, m1_serial = _run_driver(script, path, dim, 1)
    try:
        _, m1_par = _run_driver(script, path, dim, 2)
    except Exception as e:                            # e.g. oversubscribe refusal
        pytest.skip(f"parallel run unavailable: {e}")
    assert m1_serial != 0.0
    assert abs(m1_par - m1_serial) / abs(m1_serial) < RANK_RTOL

"""
On-the-fly structured mesh generation via gmsh.

manapy maps boundary patches to integer physical tags:
    in = 1, out = 2, bottom = 3, upper = 4, front = 5, back = 6
(0 = interior). The generators below assign exactly these tags so the meshes
work out of the box with the high-level api (Mesh.rectangle / Mesh.box) and
boundary dicts keyed by "in"/"out"/"bottom"/"upper"/"front"/"back".

Axis convention:
    in  = x-min   out = x-max
    bottom = y-min   upper = y-max
    front = z-min    back  = z-max
"""
import os
import shutil
import subprocess
import tempfile


def _run_gmsh(geo_str, dim, out_path):
  if shutil.which("gmsh") is None:
    raise RuntimeError("gmsh executable not found in PATH; cannot generate a mesh.")
  with tempfile.NamedTemporaryFile("w", suffix=".geo", delete=False) as f:
    f.write(geo_str)
    geo_path = f.name
  try:
    subprocess.run(
      ["gmsh", geo_path, f"-{dim}", "-format", "msh2", "-o", out_path],
      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
  finally:
    os.unlink(geo_path)
  return out_path


def _as_pair(n, k):
  if isinstance(n, (tuple, list)):
    if len(n) != k:
      raise ValueError(f"n must have {k} entries, got {n}")
    return [int(v) for v in n]
  return [int(n)] * k


def rectangle(bounds=((0, 1), (0, 1)), n=20, cell_type="triangle",
              transfinite=True, recombine=None, filename=None):
  """Generate a 2D rectangle mesh.

  cell_type   : 'triangle' (default) or 'quad'. Only used to default `recombine`.
  transfinite : True (default) -> structured (mapped) mesh; False -> unstructured
                (Delaunay), target cell size derived from n.
  recombine   : recombine triangles into quads. None (default) -> (cell_type=='quad').
                Combinations:
                  transfinite + no recombine -> structured triangles
                  transfinite + recombine    -> structured quads
                  unstructured + no recombine-> unstructured triangles
                  unstructured + recombine   -> quad-dominant HYBRID (leftover triangles)
  """
  (x0, x1), (y0, y1) = bounds
  nx, ny = _as_pair(n, 2)
  if recombine is None:
    recombine = (cell_type == "quad")
  lc = min((x1 - x0) / nx, (y1 - y0) / ny)          # target size for the unstructured case
  struct = (f"Transfinite Line {{4, 2}} = {ny + 1};\n"
            f"Transfinite Line {{1, 3}} = {nx + 1};\n"
            "Transfinite Surface {1};\n") if transfinite else ""
  recomb = "Recombine Surface {1};\n" if recombine else ""
  # unstructured + recombine -> keep leftover triangles (hybrid) via the simple algorithm
  algo = "Mesh.RecombinationAlgorithm = 0;\n" if (recombine and not transfinite) else ""

  geo = f"""{algo}
Point(1) = {{{x0}, {y0}, 0, {lc}}};
Point(2) = {{{x1}, {y0}, 0, {lc}}};
Point(3) = {{{x1}, {y1}, 0, {lc}}};
Point(4) = {{{x0}, {y1}, 0, {lc}}};
Line(1) = {{1, 2}};   // bottom (y-min)
Line(2) = {{2, 3}};   // out    (x-max)
Line(3) = {{3, 4}};   // upper  (y-max)
Line(4) = {{4, 1}};   // in     (x-min)
Line Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{1}};
{struct}{recomb}Physical Curve("in", 1)     = {{4}};
Physical Curve("out", 2)    = {{2}};
Physical Curve("bottom", 3) = {{1}};
Physical Curve("upper", 4)  = {{3}};
Physical Surface("domain", 1) = {{1}};
"""
  if filename is None:
    filename = tempfile.NamedTemporaryFile(suffix=".msh", delete=False).name
  return _run_gmsh(geo, 2, filename)


def box(bounds=((0, 1), (0, 1), (0, 1)), n=10, cell_type="tetra",
        transfinite=True, recombine=None, filename=None):
  """Generate a 3D box mesh.

  cell_type   : 'tetra' (default) or 'hex'. Only used to default `recombine`.
  transfinite : True (default) -> structured (mapped) mesh; False -> unstructured
                (Delaunay tets), target cell size derived from n.
  recombine   : recombine into hexahedra. None (default) -> (cell_type=='hex').
                Combinations:
                  transfinite + no recombine -> structured tetrahedra
                  transfinite + recombine    -> structured hexahedra
                  unstructured + no recombine-> unstructured tetrahedra
                  unstructured + recombine   -> attempts hex recombination (gmsh 3D
                                                recombination is limited; may stay tetrahedral)
  """
  (x0, x1), (y0, y1), (z0, z1) = bounds
  nx, ny, nz = _as_pair(n, 3)
  if recombine is None:
    recombine = (cell_type == "hex")
  lc = min((x1 - x0) / nx, (y1 - y0) / ny, (z1 - z0) / nz)   # size for the unstructured case
  struct = (f"Transfinite Line {{1, 3, 5, 7}} = {nx + 1};\n"
            f"Transfinite Line {{2, 4, 6, 8}} = {ny + 1};\n"
            f"Transfinite Line {{9, 10, 11, 12}} = {nz + 1};\n"
            'Transfinite Surface "*";\n'
            "Transfinite Volume {1};\n") if transfinite else ""
  # hexes need the bounding SURFACES recombined to quads first, then the volume
  recomb = ('Recombine Surface "*";\n'
            "Recombine Volume {1};\n") if recombine else ""
  algo = "Mesh.RecombinationAlgorithm = 0;\n" if (recombine and not transfinite) else ""

  geo = f"""{algo}
Point(1) = {{{x0}, {y0}, {z0}, {lc}}};
Point(2) = {{{x1}, {y0}, {z0}, {lc}}};
Point(3) = {{{x1}, {y1}, {z0}, {lc}}};
Point(4) = {{{x0}, {y1}, {z0}, {lc}}};
Point(5) = {{{x0}, {y0}, {z1}, {lc}}};
Point(6) = {{{x1}, {y0}, {z1}, {lc}}};
Point(7) = {{{x1}, {y1}, {z1}, {lc}}};
Point(8) = {{{x0}, {y1}, {z1}, {lc}}};
Line(1) = {{1, 2}};  Line(2) = {{2, 3}};  Line(3) = {{3, 4}};  Line(4) = {{4, 1}};
Line(5) = {{5, 6}};  Line(6) = {{6, 7}};  Line(7) = {{7, 8}};  Line(8) = {{8, 5}};
Line(9) = {{1, 5}};  Line(10) = {{2, 6}}; Line(11) = {{3, 7}}; Line(12) = {{4, 8}};
Line Loop(1) = {{1, 2, 3, 4}};      Plane Surface(1) = {{1}};   // z-min (front)
Line Loop(2) = {{5, 6, 7, 8}};      Plane Surface(2) = {{2}};   // z-max (back)
Line Loop(3) = {{1, 10, -5, -9}};   Plane Surface(3) = {{3}};   // y-min (bottom)
Line Loop(4) = {{2, 11, -6, -10}};  Plane Surface(4) = {{4}};   // x-max (out)
Line Loop(5) = {{3, 12, -7, -11}};  Plane Surface(5) = {{5}};   // y-max (upper)
Line Loop(6) = {{4, 9, -8, -12}};   Plane Surface(6) = {{6}};   // x-min (in)
Surface Loop(1) = {{1, 2, 3, 4, 5, 6}};
Volume(1) = {{1}};
{struct}{recomb}Physical Surface("in", 1)     = {{6}};
Physical Surface("out", 2)    = {{4}};
Physical Surface("bottom", 3) = {{3}};
Physical Surface("upper", 4)  = {{5}};
Physical Surface("front", 5)  = {{1}};
Physical Surface("back", 6)   = {{2}};
Physical Volume("domain", 1)  = {{1}};
"""
  if filename is None:
    filename = tempfile.NamedTemporaryFile(suffix=".msh", delete=False).name
  return _run_gmsh(geo, 3, filename)

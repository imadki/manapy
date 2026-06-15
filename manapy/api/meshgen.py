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


def rectangle(bounds=((0, 1), (0, 1)), n=20, cell_type="triangle", filename=None):
  """Generate a 2D rectangle mesh. cell_type: 'triangle' (default) or 'quad'."""
  (x0, x1), (y0, y1) = bounds
  nx, ny = _as_pair(n, 2)
  recombine = "Recombine Surface {1};" if cell_type == "quad" else ""

  geo = f"""
Point(1) = {{{x0}, {y0}, 0, 1}};
Point(2) = {{{x1}, {y0}, 0, 1}};
Point(3) = {{{x1}, {y1}, 0, 1}};
Point(4) = {{{x0}, {y1}, 0, 1}};
Line(1) = {{1, 2}};   // bottom (y-min)
Line(2) = {{2, 3}};   // out    (x-max)
Line(3) = {{3, 4}};   // upper  (y-max)
Line(4) = {{4, 1}};   // in     (x-min)
Line Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{1}};
Transfinite Line {{4, 2}} = {ny + 1};
Transfinite Line {{1, 3}} = {nx + 1};
Transfinite Surface {{1}};
{recombine}
Physical Curve("in", 1)     = {{4}};
Physical Curve("out", 2)    = {{2}};
Physical Curve("bottom", 3) = {{1}};
Physical Curve("upper", 4)  = {{3}};
Physical Surface("domain", 1) = {{1}};
"""
  if filename is None:
    filename = tempfile.NamedTemporaryFile(suffix=".msh", delete=False).name
  return _run_gmsh(geo, 2, filename)


def box(bounds=((0, 1), (0, 1), (0, 1)), n=10, cell_type="tetra", filename=None):
  """Generate a 3D box mesh. cell_type: 'tetra' (default) or 'hex'."""
  (x0, x1), (y0, y1), (z0, z1) = bounds
  nx, ny, nz = _as_pair(n, 3)
  recombine = "Recombine Volume {1};\nTransfinite Volume {1};" if cell_type == "hex" else ""

  geo = f"""
Point(1) = {{{x0}, {y0}, {z0}, 1}};
Point(2) = {{{x1}, {y0}, {z0}, 1}};
Point(3) = {{{x1}, {y1}, {z0}, 1}};
Point(4) = {{{x0}, {y1}, {z0}, 1}};
Point(5) = {{{x0}, {y0}, {z1}, 1}};
Point(6) = {{{x1}, {y0}, {z1}, 1}};
Point(7) = {{{x1}, {y1}, {z1}, 1}};
Point(8) = {{{x0}, {y1}, {z1}, 1}};
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
Transfinite Line {{1, 3, 5, 7}} = {nx + 1};
Transfinite Line {{2, 4, 6, 8}} = {ny + 1};
Transfinite Line {{9, 10, 11, 12}} = {nz + 1};
Transfinite Surface "*";
{recombine}
Physical Surface("in", 1)     = {{6}};
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

"""
Built-in structured mesh generator — no gmsh or pygmsh required.

Writes gmsh 2.2 ASCII files (.msh) that manapy can read directly.

Boundary tag convention (matches existing manapy meshes):
  2D:  1=in (x=0), 2=out (x=Lx), 3=upper (y=Ly), 4=bottom (y=0)
  3D:  + 5=front (z=Lz), 6=back (z=0)

Cell types:
  2D: "triangle" (default) — each quad split into 2 triangles
      "quad"               — pure structured quad mesh
  3D: "tetra"   (default) — each hex split into 6 tetrahedra (Freudenthal)
      "hex"                — pure structured hexahedral mesh
"""

import os
import tempfile


# ---------------------------------------------------------------------------
# Low-level writer helpers
# ---------------------------------------------------------------------------

def _write_header(f):
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")


def _write_physical_names_2d(f):
    f.write("$PhysicalNames\n5\n")
    f.write('1 1 "in"\n')
    f.write('1 2 "out"\n')
    f.write('1 3 "upper"\n')
    f.write('1 4 "bottom"\n')
    f.write('2 1 "surface"\n')
    f.write("$EndPhysicalNames\n")


def _write_physical_names_3d(f):
    f.write("$PhysicalNames\n7\n")
    f.write('2 1 "in"\n')
    f.write('2 2 "out"\n')
    f.write('2 3 "upper"\n')
    f.write('2 4 "bottom"\n')
    f.write('2 5 "front"\n')
    f.write('2 6 "back"\n')
    f.write('3 1 "volume"\n')
    f.write("$EndPhysicalNames\n")


def _temp_file(prefix):
    fd, path = tempfile.mkstemp(suffix=".msh", prefix=prefix)
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# 2D rectangle  [0,Lx] x [0,Ly]
# ---------------------------------------------------------------------------

def rectangle(Lx=1.0, Ly=1.0, nx=10, ny=10,
              cell_type="triangle", filename=None):
    """
    Structured rectangle mesh.

    Parameters
    ----------
    Lx, Ly    : float — dimensions
    nx, ny    : int   — cells per direction
    cell_type : str   — "triangle" (default) or "quad"
    filename  : str, optional — output path; temp file if None

    Returns
    -------
    str — path to the .msh file
    """
    if cell_type not in ("triangle", "quad"):
        raise ValueError(f"cell_type must be 'triangle' or 'quad', got '{cell_type}'")
    if filename is None:
        filename = _temp_file("manapy_rect_")

    dx = Lx / nx
    dy = Ly / ny

    def nid(i, j):          # 1-based
        return i * (ny + 1) + j + 1

    nnodes = (nx + 1) * (ny + 1)

    # boundary edges (type 1) — same for both cell types
    edges = []
    for j in range(ny):
        edges.append((nid(0,  j), nid(0,  j+1), 1))   # left   "in"
        edges.append((nid(nx, j), nid(nx, j+1), 2))   # right  "out"
    for i in range(nx):
        edges.append((nid(i, ny), nid(i+1, ny), 3))   # top    "upper"
        edges.append((nid(i,  0), nid(i+1,  0), 4))   # bottom "bottom"

    # interior cells
    cells = []
    if cell_type == "triangle":
        gmsh_type = 2           # 3-node triangle
        for i in range(nx):
            for j in range(ny):
                cells.append((nid(i,j),   nid(i+1,j),   nid(i+1,j+1)))
                cells.append((nid(i,j),   nid(i+1,j+1), nid(i,  j+1)))
    else:  # quad
        gmsh_type = 3           # 4-node quad
        for i in range(nx):
            for j in range(ny):
                # CCW: bottom-left, bottom-right, top-right, top-left
                cells.append((nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)))

    with open(filename, "w") as f:
        _write_header(f)
        _write_physical_names_2d(f)

        f.write(f"$Nodes\n{nnodes}\n")
        for i in range(nx + 1):
            for j in range(ny + 1):
                f.write(f"{nid(i,j)} {i*dx:.10g} {j*dy:.10g} 0\n")
        f.write("$EndNodes\n")

        total = len(edges) + len(cells)
        f.write(f"$Elements\n{total}\n")
        eid = 1
        for n1, n2, tag in edges:
            f.write(f"{eid} 1 2 {tag} {tag} {n1} {n2}\n")
            eid += 1
        for cell in cells:
            nodes_str = " ".join(map(str, cell))
            f.write(f"{eid} {gmsh_type} 2 1 1 {nodes_str}\n")
            eid += 1
        f.write("$EndElements\n")

    return filename


def square(L=1.0, n=10, cell_type="triangle", filename=None):
    """Square mesh — shortcut for rectangle(L, L, n, n, cell_type)."""
    return rectangle(Lx=L, Ly=L, nx=n, ny=n,
                     cell_type=cell_type, filename=filename)


# ---------------------------------------------------------------------------
# 3D box  [0,Lx] x [0,Ly] x [0,Lz]
# ---------------------------------------------------------------------------

def box(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5,
        cell_type="tetra", filename=None):
    """
    Structured box mesh.

    Parameters
    ----------
    Lx, Ly, Lz : float — dimensions
    nx, ny, nz : int   — cells per direction
    cell_type  : str   — "tetra" (default) or "hex"
    filename   : str, optional

    Returns
    -------
    str — path to the .msh file
    """
    if cell_type not in ("tetra", "hex"):
        raise ValueError(f"cell_type must be 'tetra' or 'hex', got '{cell_type}'")
    if filename is None:
        filename = _temp_file("manapy_box_")

    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    def nid(i, j, k):          # 1-based
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k + 1

    nnodes = (nx + 1) * (ny + 1) * (nz + 1)

    # boundary faces — triangles for "tetra", quads for "hex"
    if cell_type == "tetra":
        bnd_faces = []   # (n1, n2, n3, tag)
        bnd_gmsh_type = 2

        for j in range(ny):
            for k in range(nz):
                bnd_faces += [
                    (nid(0,j,k),  nid(0,j+1,k),   nid(0,j+1,k+1), 1),
                    (nid(0,j,k),  nid(0,j+1,k+1), nid(0,j,  k+1), 1),
                    (nid(nx,j,k), nid(nx,j+1,k),  nid(nx,j+1,k+1),2),
                    (nid(nx,j,k), nid(nx,j+1,k+1),nid(nx,j,  k+1),2),
                ]
        for i in range(nx):
            for k in range(nz):
                bnd_faces += [
                    (nid(i,ny,k),  nid(i+1,ny,k),  nid(i+1,ny,k+1), 3),
                    (nid(i,ny,k),  nid(i+1,ny,k+1),nid(i,  ny,k+1), 3),
                    (nid(i,0, k),  nid(i+1,0, k),  nid(i+1,0, k+1), 4),
                    (nid(i,0, k),  nid(i+1,0, k+1),nid(i,  0, k+1), 4),
                ]
        for i in range(nx):
            for j in range(ny):
                bnd_faces += [
                    (nid(i,j,nz), nid(i+1,j,nz),  nid(i+1,j+1,nz), 5),
                    (nid(i,j,nz), nid(i+1,j+1,nz),nid(i,  j+1,nz), 5),
                    (nid(i,j,0),  nid(i+1,j,0),   nid(i+1,j+1,0),  6),
                    (nid(i,j,0),  nid(i+1,j+1,0), nid(i,  j+1,0),  6),
                ]
    else:  # hex → quad boundary faces
        bnd_faces = []   # (n1, n2, n3, n4, tag)
        bnd_gmsh_type = 3

        for j in range(ny):
            for k in range(nz):
                bnd_faces.append((nid(0,j,k),  nid(0,j+1,k),  nid(0,j+1,k+1),nid(0,j,  k+1), 1))
                bnd_faces.append((nid(nx,j,k), nid(nx,j+1,k), nid(nx,j+1,k+1),nid(nx,j,  k+1),2))
        for i in range(nx):
            for k in range(nz):
                bnd_faces.append((nid(i,ny,k), nid(i+1,ny,k), nid(i+1,ny,k+1),nid(i,ny,k+1), 3))
                bnd_faces.append((nid(i,0, k), nid(i+1,0, k), nid(i+1,0, k+1),nid(i,0, k+1), 4))
        for i in range(nx):
            for j in range(ny):
                bnd_faces.append((nid(i,j,nz), nid(i+1,j,nz), nid(i+1,j+1,nz),nid(i,j+1,nz), 5))
                bnd_faces.append((nid(i,j,0),  nid(i+1,j,0),  nid(i+1,j+1,0), nid(i,j+1,0),  6))

    # interior cells
    if cell_type == "tetra":
        vol_gmsh_type = 4     # 4-node tetra
        vol_cells = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n000=nid(i,j,k);     n100=nid(i+1,j,k);   n010=nid(i,j+1,k)
                    n001=nid(i,j,k+1);   n110=nid(i+1,j+1,k); n101=nid(i+1,j,k+1)
                    n011=nid(i,j+1,k+1); n111=nid(i+1,j+1,k+1)
                    vol_cells += [
                        (n000, n100, n110, n111),
                        (n000, n100, n101, n111),
                        (n000, n010, n110, n111),
                        (n000, n001, n101, n111),
                        (n000, n010, n011, n111),
                        (n000, n001, n011, n111),
                    ]
    else:  # hex
        vol_gmsh_type = 5     # 8-node hex
        vol_cells = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # gmsh hex ordering: bottom face CCW then top face CCW
                    vol_cells.append((
                        nid(i,j,k),     nid(i+1,j,k),   nid(i+1,j+1,k), nid(i,j+1,k),
                        nid(i,j,k+1),   nid(i+1,j,k+1), nid(i+1,j+1,k+1), nid(i,j+1,k+1),
                    ))

    with open(filename, "w") as f:
        _write_header(f)
        _write_physical_names_3d(f)

        f.write(f"$Nodes\n{nnodes}\n")
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    f.write(f"{nid(i,j,k)} {i*dx:.10g} {j*dy:.10g} {k*dz:.10g}\n")
        f.write("$EndNodes\n")

        total = len(bnd_faces) + len(vol_cells)
        f.write(f"$Elements\n{total}\n")
        eid = 1
        for face in bnd_faces:
            *nodes, tag = face
            nodes_str = " ".join(map(str, nodes))
            f.write(f"{eid} {bnd_gmsh_type} 2 {tag} {tag} {nodes_str}\n")
            eid += 1
        for cell in vol_cells:
            nodes_str = " ".join(map(str, cell))
            f.write(f"{eid} {vol_gmsh_type} 2 1 1 {nodes_str}\n")
            eid += 1
        f.write("$EndElements\n")

    return filename


def cube(L=1.0, n=5, cell_type="tetra", filename=None):
    """Cube mesh — shortcut for box(L, L, L, n, n, n, cell_type)."""
    return box(Lx=L, Ly=L, Lz=L, nx=n, ny=n, nz=n,
               cell_type=cell_type, filename=filename)

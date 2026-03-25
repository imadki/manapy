"""
Built-in structured mesh generator — no gmsh or pygmsh required.

Writes gmsh 2.2 ASCII files (.msh) that manapy can read directly.

Boundary tag convention (matches existing manapy meshes):
  2D:  1=in (x=0), 2=out (x=Lx), 3=upper (y=Ly), 4=bottom (y=0)
  3D:  + 5=front (z=Lz), 6=back (z=0)
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


# ---------------------------------------------------------------------------
# 2D rectangle  [0,Lx] x [0,Ly]  — triangulated structured grid
# ---------------------------------------------------------------------------

def rectangle(Lx=1.0, Ly=1.0, nx=10, ny=10, filename=None):
    """
    Generate a structured triangulated rectangle mesh.

    Parameters
    ----------
    Lx, Ly   : float — dimensions
    nx, ny   : int   — number of cells in each direction
    filename : str, optional — output path (.msh).
               If None, a temp file is created.

    Returns
    -------
    str — path to the generated .msh file
    """
    if filename is None:
        fd, filename = tempfile.mkstemp(suffix=".msh", prefix="manapy_rect_")
        os.close(fd)

    nnodes = (nx + 1) * (ny + 1)
    dx = Lx / nx
    dy = Ly / ny

    def nid(i, j):          # 1-based node index
        return i * (ny + 1) + j + 1

    # --- boundary edges (type 1) ---
    edges = []
    for j in range(ny):
        edges.append((nid(0,  j), nid(0,  j+1), 1))   # left  → "in"
        edges.append((nid(nx, j), nid(nx, j+1), 2))   # right → "out"
    for i in range(nx):
        edges.append((nid(i, ny), nid(i+1, ny), 3))   # top    → "upper"
        edges.append((nid(i,  0), nid(i+1,  0), 4))   # bottom → "bottom"

    # --- triangles (type 2) ---
    triangles = []
    for i in range(nx):
        for j in range(ny):
            triangles.append((nid(i, j),   nid(i+1, j),   nid(i+1, j+1)))
            triangles.append((nid(i, j),   nid(i+1, j+1), nid(i,   j+1)))

    total_elems = len(edges) + len(triangles)

    with open(filename, "w") as f:
        _write_header(f)
        _write_physical_names_2d(f)

        f.write(f"$Nodes\n{nnodes}\n")
        for i in range(nx + 1):
            for j in range(ny + 1):
                f.write(f"{nid(i,j)} {i*dx:.10g} {j*dy:.10g} 0\n")
        f.write("$EndNodes\n")

        f.write(f"$Elements\n{total_elems}\n")
        eid = 1
        for n1, n2, tag in edges:
            f.write(f"{eid} 1 2 {tag} {tag} {n1} {n2}\n")
            eid += 1
        for n1, n2, n3 in triangles:
            f.write(f"{eid} 2 2 1 1 {n1} {n2} {n3}\n")
            eid += 1
        f.write("$EndElements\n")

    return filename


def square(L=1.0, n=10, filename=None):
    """Square mesh — shortcut for rectangle(L, L, n, n)."""
    return rectangle(Lx=L, Ly=L, nx=n, ny=n, filename=filename)


# ---------------------------------------------------------------------------
# 3D box  [0,Lx] x [0,Ly] x [0,Lz]  — tetrahedral structured grid
# ---------------------------------------------------------------------------
# Each hex cell is split into 6 tetrahedra (Freudenthal decomposition).
# Boundary quads are split into 2 triangles.

def box(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5, filename=None):
    """
    Generate a structured tetrahedral box mesh.

    Parameters
    ----------
    Lx, Ly, Lz : float — dimensions
    nx, ny, nz : int   — number of cells in each direction
    filename   : str, optional

    Returns
    -------
    str — path to the generated .msh file
    """
    if filename is None:
        fd, filename = tempfile.mkstemp(suffix=".msh", prefix="manapy_box_")
        os.close(fd)

    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    def nid(i, j, k):          # 1-based
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k + 1

    nnodes = (nx + 1) * (ny + 1) * (nz + 1)

    # --- boundary triangles (type 2) ---
    # Each boundary quad → 2 triangles
    bnd_tris = []   # (n1, n2, n3, tag)

    for j in range(ny):
        for k in range(nz):
            # x=0  → tag 1 "in"
            bnd_tris.append((nid(0,j,k),   nid(0,j+1,k),   nid(0,j+1,k+1), 1))
            bnd_tris.append((nid(0,j,k),   nid(0,j+1,k+1), nid(0,j,  k+1), 1))
            # x=Lx → tag 2 "out"
            bnd_tris.append((nid(nx,j,k),  nid(nx,j+1,k),  nid(nx,j+1,k+1), 2))
            bnd_tris.append((nid(nx,j,k),  nid(nx,j+1,k+1),nid(nx,j,  k+1), 2))

    for i in range(nx):
        for k in range(nz):
            # y=Ly → tag 3 "upper"
            bnd_tris.append((nid(i,ny,k),  nid(i+1,ny,k),  nid(i+1,ny,k+1), 3))
            bnd_tris.append((nid(i,ny,k),  nid(i+1,ny,k+1),nid(i,  ny,k+1), 3))
            # y=0  → tag 4 "bottom"
            bnd_tris.append((nid(i,0,k),   nid(i+1,0,k),   nid(i+1,0,k+1),  4))
            bnd_tris.append((nid(i,0,k),   nid(i+1,0,k+1), nid(i,  0,k+1),  4))

    for i in range(nx):
        for j in range(ny):
            # z=Lz → tag 5 "front"
            bnd_tris.append((nid(i,j,nz),  nid(i+1,j,nz),  nid(i+1,j+1,nz), 5))
            bnd_tris.append((nid(i,j,nz),  nid(i+1,j+1,nz),nid(i,  j+1,nz), 5))
            # z=0  → tag 6 "back"
            bnd_tris.append((nid(i,j,0),   nid(i+1,j,0),   nid(i+1,j+1,0),  6))
            bnd_tris.append((nid(i,j,0),   nid(i+1,j+1,0), nid(i,  j+1,0),  6))

    # --- tetrahedra (type 4) — Freudenthal decomposition of each hex ---
    # Hex corners: n000..n111 using (di,dj,dk) offsets
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = [[[ nid(i+di, j+dj, k+dk)
                        for dk in range(2)]
                        for dj in range(2)]
                        for di in range(2)]
                n000=n[0][0][0]; n100=n[1][0][0]; n010=n[0][1][0]; n001=n[0][0][1]
                n110=n[1][1][0]; n101=n[1][0][1]; n011=n[0][1][1]; n111=n[1][1][1]

                tets.append((n000, n100, n110, n111))
                tets.append((n000, n100, n101, n111))
                tets.append((n000, n010, n110, n111))
                tets.append((n000, n001, n101, n111))
                tets.append((n000, n010, n011, n111))
                tets.append((n000, n001, n011, n111))

    total_elems = len(bnd_tris) + len(tets)

    with open(filename, "w") as f:
        _write_header(f)
        _write_physical_names_3d(f)

        f.write(f"$Nodes\n{nnodes}\n")
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    f.write(f"{nid(i,j,k)} {i*dx:.10g} {j*dy:.10g} {k*dz:.10g}\n")
        f.write("$EndNodes\n")

        f.write(f"$Elements\n{total_elems}\n")
        eid = 1
        for n1, n2, n3, tag in bnd_tris:
            f.write(f"{eid} 2 2 {tag} {tag} {n1} {n2} {n3}\n")
            eid += 1
        for n1, n2, n3, n4 in tets:
            f.write(f"{eid} 4 2 1 1 {n1} {n2} {n3} {n4}\n")
            eid += 1
        f.write("$EndElements\n")

    return filename


def cube(L=1.0, n=5, filename=None):
    """Cube mesh — shortcut for box(L, L, L, n, n, n)."""
    return box(Lx=L, Ly=L, Lz=L, nx=n, ny=n, nz=n, filename=filename)

# meshes/geo — gmsh source files

This folder holds the **gmsh `.geo` source files**, one per mesh test case.
The generated `.msh` files are **not** tracked (they are `.gitignore`d) — generate
them locally from the `.geo` before running an example:

```bash
gmsh -2 uns_square.geo        -o uns_square.msh          # 2D
gmsh -3 uns_cube.geo          -o uns_cube.msh            # 3D
gmsh -3 periodic_cube_hex.geo -o periodic_cube_hex.msh
```

By default the examples read their mesh from this folder (`MESH_DIR=.../meshes/geo`);
override with the `MESH_DIR` / `MESH_FILE` environment variables.

## Naming convention

`<element-or-type>_<shape>.geo`

- element / type: `uns` (unstructured), `struct` (structured, transfinite),
  `quad` / `hex` (structured, recombined), `periodic`, `hybrid`
- shape: `square` `[0,1]²`, `rectangle` `10×5`, `cube` `[0,1]³`

## Boundary-tag conventions (read by manapy)

- **2D non-periodic**: `Physical Line` `1`=in (x=0), `2`=out (x=1), `3`=upper (y=1),
  `4`=bottom (y=0).
- **2D periodic**: `11`=in, `22`=out, `33`=upper, `44`=bottom, plus `Periodic Line`.
- **3D**: faces named `in/out/bottom/upper/front/back`, or periodic tags
  `11`=in `22`=out `33`=upper `44`=bottom `55`=front `66`=back, plus `Periodic Surface`.

Periodic meshes need aligned boundary nodes (`Transfinite Line`) **and** a
`Periodic` directive so opposite sides share an identical discretization. For
periodic **tet** cubes do *not* add `Transfinite Surface/Volume` — the free tet
fill must keep every boundary triangle a real tet face, otherwise manapy rejects
the mesh (see the header of `periodic_cube_tet.geo`); use hexes for a fully
structured periodic cube.

## Catalogue

| `.geo` | shape | elements | used by example(s) |
|---|---|---|---|
| `uns_square` | unit square | triangles (free) | most 2D examples (darcy, laplacien, advection, euler, diffusion, shallow water, swmhd) |
| `struct_square` | unit square | triangles (structured) | — (generic / tests) |
| `quad_square` | unit square | quadrilaterals | — (generic / tests) |
| `periodic_square` | unit square | triangles, doubly periodic | `advection2d_periodic` |
| `struct_rectangle` | 10×5 | triangles (structured) | — (generic / tests) |
| `quad_rectangle` | 10×5 | quadrilaterals | — (generic / tests) |
| `hybrid2d` | tiled square | mixed tri + quad | — (hybrid-support test) |
| `uns_cube` | unit cube | tetrahedra (free) | `euler3d_gpu` |
| `hex_cube` | unit cube | hexahedra | — (generic / tests) |
| `hybrid3d` | box | mixed tet + hex/prism | 3D examples (advection, advecdiff, darcy, diffusion, laplacien, weno euler) |
| `periodic_cube_hex` | unit cube | hexahedra, triply periodic | `advection3d_periodic` |
| `periodic_cube_tet` | unit cube | tetrahedra, triply periodic | — (periodic-tet reference) |

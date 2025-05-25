//——————————————————————————————————————
// Cuboid built from tets with matching boundary triangles
// (First method: per‐face transfinite subdivisions)
//——————————————————————————————————————

// Geometry parameters
Lx = 10;   Ly = 5;    Lz = 15;
Nx = 10;   Ny = 10;   Nz = 10;

// Corner points
Point(1) = {0,   0,   0,  1.0};
Point(2) = {Lx,  0,   0,  1.0};
Point(3) = {Lx,  Ly,  0,  1.0};
Point(4) = {0,   Ly,  0,  1.0};
Point(5) = {0,   0,   Lz, 1.0};
Point(6) = {Lx,  0,   Lz, 1.0};
Point(7) = {Lx,  Ly,  Lz, 1.0};
Point(8) = {0,   Ly,  Lz, 1.0};

// Edges
Line(1)  = {1,2};
Line(2)  = {2,3};
Line(3)  = {3,4};
Line(4)  = {4,1};
Line(5)  = {5,6};
Line(6)  = {6,7};
Line(7)  = {7,8};
Line(8)  = {8,5};
Line(9)  = {1,5};
Line(10) = {2,6};
Line(11) = {3,7};
Line(12) = {4,8};

// Planar faces
// 1) bottom (z=0)
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
// 2) top (z=Lz)
Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
// 3) y=0 (x–z face)
Line Loop(3) = {1, 10, -5, -9};
Plane Surface(3) = {3};
// 4) x=Lx (y–z face)
Line Loop(4) = {2, 11, -6, -10};
Plane Surface(4) = {4};
// 5) y=Ly (x–z face)
Line Loop(5) = {12, -7, -11, 3};
Plane Surface(5) = {5};
// 6) x=0 (y–z face)
Line Loop(6) = {9, -8, -12, 4};
Plane Surface(6) = {6};

// Volume
Surface Loop(1) = {1,2,3,4,5,6};
Volume(1)       = {1};

// ——— First‐method transfinite subdivisions ———
// horizontal faces in x–y plane
Transfinite Surface {1,2} = {Nx+1, Ny+1};
// vertical faces in x–z plane
Transfinite Surface {3,5} = {Nx+1, Nz+1};
// vertical faces in y–z plane
Transfinite Surface {4,6} = {Ny+1, Nz+1};
// interior
Transfinite Volume  {1}   = {Nx+1, Ny+1, Nz+1};

// Tet‐mesh in 3D
Mesh.Algorithm3D = 1;
Mesh 3;

// Physical groups
Physical Surface("Zmin") = {1};
Physical Surface("Zmax") = {2};
Physical Surface("Ymin") = {3};
Physical Surface("Xmax") = {4};
Physical Surface("Ymax") = {5};
Physical Surface("Xmin") = {6};
Physical Volume ("Domain") = {1};


Mesh 3;


Save "tetrahedron.msh";

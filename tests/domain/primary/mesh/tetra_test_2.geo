Lx = 10;  // Length of the rectangle in the x-direction
Ly = 5;   // Length of the rectangle in the y-direction
Lz = 15;

Nx = 100;  // Number of divisions in the x-direction
Ny = 100;  // Number of divisions in the y-direction
Nz = 100;

// Create the rectangle geometry
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


Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
Line Loop(3) = {1, 10, -5, -9};
Plane Surface(3) = {3};
Line Loop(4) = {2, 11, -6, -10};
Plane Surface(4) = {4};
Line Loop(5) = {12, -7, -11, 3};
Plane Surface(5) = {5};
Line Loop(6) = {9, -8, -12, 4};
Plane Surface(6) = {6};


Surface Loop(1) = {1,2,3,4,5,6};
Volume(1)       = {1};

// Define the meshing parameters
Transfinite Line {10, 12, 9, 11, 13} = Nx + 1;
Transfinite Line {5, 7, 1, 3} = Ny + 1;
Transfinite Line {8, 6, 2, 4} = Nz + 1;

Transfinite Surface "*";
// Recombine Surface {1, 2, 3, 4, 5, 6};
Transfinite Volume {1};




Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};
Physical Surface(4) = {4};
Physical Surface(5) = {5};
Physical Surface(6) = {6};
Physical Volume (1) = {1};

Mesh.Algorithm3D = 1;
// Mesh 3;


// Save "tetrahedron_big.msh";





Lx = 10;  // Length of the rectangle in the x-direction
Ly = 5;   // Length of the rectangle in the y-direction
Lz = 15;

Nx = 10;  // Number of divisions in the x-direction
Ny = 10;  // Number of divisions in the y-direction
Nz = 10;

// Create the rectangle geometry
Point(1) = {0, 0, 0, 1.0};
Point(2) = {Lx, 0, 0, 1.0};
Point(3) = {Lx, Ly, 0, 1.0};
Point(4) = {0, Ly, 0, 1.0};
Point(5) = {0, 0, Lz, 1.0};
Point(6) = {Lx, 0, Lz, 1.0};
Point(7) = {Lx, Ly, Lz, 1.0};
Point(8) = {0, Ly, Lz, 1.0};

//Bottom
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

//Top
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

//Left
Line(9) = {2, 6};
Line(10) = {1, 5};

//Right
Line(11) = {3, 7};
Line(12) = {4, 8};

Curve Loop(1) = {5, -9, -1, 10};
Plane Surface(1) = {1};
Curve Loop(2) = {7, -12, -3, 11};
Plane Surface(2) = {2};
Curve Loop(3) = {8, -10, -4, 12};
Plane Surface(3) = {3};
Curve Loop(4) = {6, -11, -2, 9};
Plane Surface(4) = {4};
Curve Loop(5) = {5, 6, 7, 8};
Plane Surface(5) = {5};
Curve Loop(6) = {1, 2, 3, 4};
Plane Surface(6) = {6};


Surface Loop(1) = {2, 3, 4, 5, 6, 1};
Volume(1) = {1};

// Define the meshing parameters
Transfinite Line {10, 12, 9, 11} = Nx + 1;
Transfinite Line {5, 7, 1, 3} = Ny + 1;
Transfinite Line {8, 6, 2, 4} = Nz + 1;

Transfinite Surface "*";
Recombine Surface {1, 2, 3, 4, 5, 6};
Transfinite Volume {1};

// Physical Line("1") = {4};
// Physical Line("2") = {2};
// Physical Line("3") = {1};
// Physical Line("4") = {3};
// Physical Surface("1") = {1};

Physical Surface(1) = {1};
Physical Surface(2) = {8};
Physical Surface(3) = {2,11};
Physical Surface(4) = {6,9};
Physical Surface(5) = {5,7};
Physical Surface(6) = {3,10};
Physical Volume(1) = {1,2};

Mesh 3;


Save "cuboid.msh";






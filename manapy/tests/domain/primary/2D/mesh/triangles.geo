Lx = 10;  // Length of the rectangle in the x-direction
Ly = 5;   // Length of the rectangle in the y-direction

Nx = 10;  // Number of divisions in the x-direction
Ny = 10;  // Number of divisions in the y-direction

Point(1) = {0, 0, 0, 1.0};
Point(2) = {Lx, 0, 0, 1.0};
Point(3) = {Lx, Ly, 0, 1.0};
Point(4) = {0, Ly, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Define the meshing parameters
Transfinite Line {1, 3} = Nx + 1;  // Divide x-boundaries into Nx segments
Transfinite Line {2, 4} = Ny + 1;  // Divide y-boundaries into Ny segments
Transfinite Surface {1};            // Apply structured meshing to the surface


Physical Line("1") = {4};
Physical Line("2") = {2};
Physical Line("3") = {1};
Physical Line("4") = {3};
Physical Surface("1") = {1};

Mesh 2;


Save "triangles.msh";



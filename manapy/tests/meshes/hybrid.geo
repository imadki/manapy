Lx = 10;  // Length of the rectangle in the x-direction
Ly = 5;   // Length of the rectangle in the y-direction

Nx = 10;  // Number of divisions in the x-direction
Ny = 10;  // Number of divisions in the y-direction

// Create the rectangle geometry
Point(1) = {0, 0, 0, 1.0};
Point(2) = {Lx, 0, 0, 1.0};
Point(3) = {Lx, Ly, 0, 1.0};
Point(4) = {0, Ly, 0, 1.0};



// Line Loop(1) = {1, 2, 3, 4};
// Plane Surface(1) = {1};

// Define the meshing parameters
// Transfinite Line {1, 3} = Nx + 1;  // Divide x-boundaries into Nx segments
// Transfinite Line {2, 4} = Ny + 1;  // Divide y-boundaries into Ny segments
// Transfinite Surface {1};            // Apply structured meshing to the surface


// Recombine Surface {1};

// Physical Line("1") = {4};
// Physical Line("2") = {2};
// Physical Line("3") = {1};
// Physical Line("4") = {3};
// Physical Surface("1") = {1};


// Mesh 2;





//+
Point(5) = {5, 5, 0, 1.0};
//+
Point(6) = {5, -0, 0, 1.0};


//+
Point(7) = {0, 2.5, 0, 1.0};
//+
Point(8) = {5, 2.5, 0, 1.0};
//+
Point(9) = {10, 2.5, 0, 1.0};
//+
Point(10) = {5, 0, 0, 1.0};
//+
Point(11) = {5, -0, 0, 1.0};
//+
Line(1) = {4, 5};
//+
Line(2) = {5, 8};
//+
Line(3) = {8, 7};
//+
Line(4) = {7, 4};
//+
Line(5) = {8, 6};
//+
Line(6) = {6, 1};
//+
Line(7) = {1, 7};
//+
Line(8) = {5, 3};
//+
Line(9) = {3, 9};
//+
Line(10) = {9, 8};
//+
Line(11) = {9, 2};
//+
Line(12) = {2, 6};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, -10, -9, -8};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {7, -3, 5, 6};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {10, 5, -12, -11};
//+
Plane Surface(4) = {4};

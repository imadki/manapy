Lx = 5;  // Length of the tile in the x-direction
Ly = 5;   // Length of the tile in the y-direction

Div = 3;  // Number of divisions

// Create the rectangle geometry
Point(4) =  {0*Lx, 3*Ly, 0, 1.0};
Point(16) = {1*Lx, 3*Ly, 0, 1.0};
Point(19) = {2*Lx, 3*Ly, 0, 1.0};
Point(3) =  {3*Lx, 3*Ly, 0, 1.0};
Point(21) = {3*Lx, 2*Ly, 0, 1.0};
Point(23) = {3*Lx, 1*Ly, 0, 1.0};
Point(2) =  {3*Lx, 0*Ly, 0, 1.0};
Point(18) = {2*Lx, 0*Ly, 0, 1.0};
Point(17) = {1*Lx, 0*Ly, 0, 1.0};
Point(1) =  {0*Lx, 0*Ly, 0, 1.0};
Point(22) = {0*Lx, 1*Ly, 0, 1.0};
Point(20) = {0*Lx, 2*Ly, 0, 1.0};
Point(13) = {1*Lx, 2*Ly, 0, 1.0};
Point(14) = {2*Lx, 2*Ly, 0, 1.0};
Point(15) = {2*Lx, 1*Ly, 0, 1.0};
Point(12) = {1*Lx, 1*Ly, 0, 1.0};



//+
Line(1) = {4, 16};
//+
Line(2) = {16, 13};
//+
Line(3) = {13, 20};
//+
Line(4) = {20, 4};
//+
Line(5) = {16, 19};
//+
Line(6) = {19, 14};
//+
Line(7) = {14, 13};
//+
Line(8) = {19, 3};
//+
Line(9) = {3, 21};
//+
Line(10) = {21, 14};
//+
Line(11) = {21, 23};
//+
Line(12) = {23, 15};
//+
Line(13) = {15, 14};
//+
Line(14) = {15, 12};
//+
Line(15) = {12, 13};
//+
Line(16) = {12, 22};
//+
Line(17) = {22, 20};
//+
Line(18) = {23, 2};
//+
Line(19) = {2, 18};
//+
Line(20) = {18, 17};
//+
Line(21) = {17, 1};
//+
Line(22) = {1, 22};
//+
Line(23) = {17, 12};
//+
Line(24) = {18, 15};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {5, 6, 7, -2};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {8, 9, 10, -6};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {10, -13, -12, -11};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {7, -15, -14, 13};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {3, -17, -16, 15};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {12, -24, -19, -18};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {14, -23, -20, 24};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {16, -22, -21, 23};
//+
Plane Surface(9) = {9};

Transfinite Line {1:24} = Div;
Transfinite Surface {1:9};
Recombine Surface {1, 2, 3, 5};

Physical Line("1") = {1, 5, 8};
Physical Line("2") = {9, 11, 18};
Physical Line("3") = {19, 20, 21};
Physical Line("4") = {22, 17, 4};
//Physical Surface("1") = {1};
Physical Surface(25) = {1, 2, 3, 4, 5, 6, 7, 8, 9};

Mesh 2;


Save "hybrid.msh";


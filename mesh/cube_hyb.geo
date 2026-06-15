lc = 1;  // Characteristic length for the mesh

// Define points for the cube
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};
Point(5) = {0, 0, 1, lc};
Point(6) = {1, 0, 1, lc};
Point(7) = {1, 1, 1, lc};
Point(8) = {0, 1, 1, lc};

// Define points for a smaller subcube
Point(9) = {0.25, 0.25, 0.25, lc};
Point(10) = {0.75, 0.25, 0.25, lc};
Point(11) = {0.75, 0.75, 0.25, lc};
Point(12) = {0.25, 0.75, 0.25, lc};
Point(13) = {0.25, 0.25, 0.75, lc};
Point(14) = {0.75, 0.25, 0.75, lc};
Point(15) = {0.75, 0.75, 0.75, lc};
Point(16) = {0.25, 0.75, 0.75, lc};

// Define lines for the main cube
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Define line loops and surfaces for the main cube
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8};
Line Loop(3) = {1, 10, -5, -9};
Line Loop(4) = {2, 11, -6, -10};
Line Loop(5) = {3, 12, -7, -11};
Line Loop(6) = {4, 9, -8, -12};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};

// Define the volume of the main cube
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// Define lines for the subcube
Line(13) = {9, 10};
Line(14) = {10, 11};
Line(15) = {11, 12};
Line(16) = {12, 9};
Line(17) = {13, 14};
Line(18) = {14, 15};
Line(19) = {15, 16};
Line(20) = {16, 13};
Line(21) = {9, 13};
Line(22) = {10, 14};
Line(23) = {11, 15};
Line(24) = {12, 16};

// Define line loops and surfaces for the subcube
Line Loop(7) = {13, 14, 15, 16};
Line Loop(8) = {17, 18, 19, 20};
Line Loop(9) = {13, 22, -17, -21};
Line Loop(10) = {14, 23, -18, -22};
Line Loop(11) = {15, 24, -19, -23};
Line Loop(12) = {16, 21, -20, -24};

Plane Surface(7) = {7};
Plane Surface(8) = {8};
Plane Surface(9) = {9};
Plane Surface(10) = {10};
Plane Surface(11) = {11};
Plane Surface(12) = {12};

// Define the volume of the subcube
Surface Loop(2) = {7, 8, 9, 10, 11, 12};
Volume(2) = {2};

// Mesh settings
Transfinite Volume {1};
Recombine Volume {1};
Transfinite Volume {2};
Recombine Volume {2};

// Physical groups
Physical Volume("MainCube") = {1};
Physical Volume("SubCube") = {2};

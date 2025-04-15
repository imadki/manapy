// Set mesh options
Mesh.ElementOrder = 1;
Mesh.Algorithm3D = 1;
Mesh.MeshSizeMin = 0.2;
Mesh.MeshSizeMax = 0.2;

// Create three separate boxes with small gaps between them

// Box 1 - For hexahedral elements (0,0,0) to (1,1,1)
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Point(5) = {0, 0, 1};
Point(6) = {1, 0, 1};
Point(7) = {1, 1, 1};
Point(8) = {0, 1, 1};


// Box 2 - For pyramids/tets (1.2,0,0) to (2.2,1,1)
Point(9) = {1.2, 0, 0};
Point(10) = {2.2, 0, 0};
Point(11) = {2.2, 1, 0};
Point(12) = {1.2, 1, 0};
Point(13) = {1.2, 0, 1};
Point(14) = {2.2, 0, 1};
Point(15) = {2.2, 1, 1};
Point(16) = {1.2, 1, 1};

// Box 3 - For prisms (2.4,0,0) to (3.4,1,1)
Point(17) = {2.4, 0, 0};
Point(18) = {3.4, 0, 0};
Point(19) = {3.4, 1, 0};
Point(20) = {2.4, 1, 0};
Point(21) = {2.4, 0, 1};
Point(22) = {3.4, 0, 1};
Point(23) = {3.4, 1, 1};
Point(24) = {2.4, 1, 1};


// Lines for box 1
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


// Lines for box 2
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

// Lines for box 3
Line(25) = {17, 18};
Line(26) = {18, 19};
Line(27) = {19, 20};
Line(28) = {20, 17};
Line(29) = {21, 22};
Line(30) = {22, 23};
Line(31) = {23, 24};
Line(32) = {24, 21};
Line(33) = {17, 21};
Line(34) = {18, 22};
Line(35) = {19, 23};
Line(36) = {20, 24};

// Surfaces for box 1
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
Line Loop(3) = {1, 10, -5, -9};
Plane Surface(3) = {3};
Line Loop(4) = {2, 11, -6, -10};
Plane Surface(4) = {4};
Line Loop(5) = {3, 12, -7, -11};
Plane Surface(5) = {5};
Line Loop(6) = {4, 9, -8, -12};
Plane Surface(6) = {6};

// Surfaces for box 2
Line Loop(7) = {13, 14, 15, 16};
Plane Surface(7) = {7};
Line Loop(8) = {17, 18, 19, 20};
Plane Surface(8) = {8};
Line Loop(9) = {13, 22, -17, -21};
Plane Surface(9) = {9};
Line Loop(10) = {14, 23, -18, -22};
Plane Surface(10) = {10};
Line Loop(11) = {15, 24, -19, -23};
Plane Surface(11) = {11};
Line Loop(12) = {16, 21, -20, -24};
Plane Surface(12) = {12};

// Surfaces for box 3
Line Loop(13) = {25, 26, 27, 28};
Plane Surface(13) = {13};
Line Loop(14) = {29, 30, 31, 32};
Plane Surface(14) = {14};
Line Loop(15) = {25, 34, -29, -33};
Plane Surface(15) = {15};
Line Loop(16) = {26, 35, -30, -34};
Plane Surface(16) = {16};
Line Loop(17) = {27, 36, -31, -35};
Plane Surface(17) = {17};
Line Loop(18) = {28, 33, -32, -36};
Plane Surface(18) = {18};

// Create volumes
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Surface Loop(2) = {7, 8, 9, 10, 11, 12};
Surface Loop(3) = {13, 14, 15, 16, 17, 18};
Volume(1) = {1};
Volume(2) = {2};
Volume(3) = {3};

// Structured mesh for box 1 (Pyramid)
Transfinite Line {1,2,3,4,5,6,7,8} = 10;
Transfinite Line {9,10,11,12} = 10;
Transfinite Surface {1,2,3,4,5,6};
Recombine Surface {1,2,3,4,5,6};

// Structured mesh for box 2 (hex)
Transfinite Line {13, 14, 15, 16, 17, 18, 19, 20} = 10;
Transfinite Line {21, 22, 23, 24} = 10;
Transfinite Surface {7, 8, 9, 10, 11, 12};
Recombine Surface {7, 8, 9, 10, 11, 12};
Transfinite Volume {2};



// Extrude settings for box 3 to create prisms
Extrude {0,0,0.2} {
  Surface{13}; Layers{10}; Recombine;
}

Mesh 3;
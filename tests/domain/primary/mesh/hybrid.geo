// Set mesh options
// Mesh.ElementOrder = 1;
Mesh.Algorithm3D = 1;
Mesh.MeshSizeMin = 0.1;
Mesh.MeshSizeMax = 0.1;

// Create points for left box (will be structured/hex mesh)
Point(1) = {0, 0, 0};
Point(2) = {0.5, 0, 0};
Point(3) = {0.5, 1, 0};
Point(4) = {0, 1, 0};
Point(5) = {0, 0, 1};
Point(6) = {0.5, 0, 1};
Point(7) = {0.5, 1, 1};
Point(8) = {0, 1, 1};

// Create points for right box (will be unstructured mesh)
Point(9) = {1, 0, 0};
Point(10) = {1, 1, 0};
Point(11) = {1, 0, 1};
Point(12) = {1, 1, 1};

// Create lines for left box
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

// Create lines for right box
Line(13) = {2, 9};
Line(14) = {9, 10};
Line(15) = {10, 3};
Line(16) = {9, 11};
Line(17) = {11, 12};
Line(18) = {12, 7};
Line(19) = {6, 11};
Line(20) = {10, 12};

// Create surfaces for left box
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

// Create surfaces for right box
Line Loop(7) = {13, 14, 15, -2};
Plane Surface(7) = {7};
Line Loop(8) = {14, 20, -17, -16};
Plane Surface(8) = {8};
Line Loop(9) = {13, 16, -19, -10};
Plane Surface(9) = {9};
Line Loop(10) = {15, 11, -18, -20};
Plane Surface(10) = {10};
Line Loop(11) = {19, 17, 18, -6};
Plane Surface(11) = {11};

// Create volumes
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};
Surface Loop(2) = {7, 8, 9, 10, 11, 4};
Volume(2) = {2};

// Structured mesh for left volume
Transfinite Line {1,2,3,4,5,6,7,8} = 20;
Transfinite Line {9,10,11,12} = 20;

// Apply transfinite meshing to left volume surfaces
Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};
Transfinite Surface {6};

// Recombine surfaces to get quadrangles
Recombine Surface {1,2,3,4,5,6};
Transfinite Volume {1};

Mesh 3;


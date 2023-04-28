// Définition des points
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};
Point(5) = {0.5, 0.5, 0, 1.0};
Point(6) = {0.5, 0.5, 1, 1.0};

// Définition des éléments
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {1, 5};
Line(6) = {2, 5};
Line(7) = {3, 5};
Line(8) = {4, 5};
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Ruled Surface(2) = {5, 6};
Ruled Surface(3) = {6, 7};
Ruled Surface(4) = {7, 8};
Ruled Surface(5) = {8, 5};
Volume(1) = {1, 2, 3, 4};


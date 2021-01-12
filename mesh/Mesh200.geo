// Gmsh project created on Mon Apr 27 01:35:00 2020
//SetFactory("OpenCASCADE");

lc= 1.5;

//+
Point(1) = {0, 0, 0, lc};
//+
Point(2) = {90, 0, 0, lc};
//+
Point(3) = {90, 95, 0, lc};
//+
Point(4) = {100, 95, 0, lc};
//+
Point(5) = {100, 0, 0, lc};
//+
Point(6) = {200, 0, 0, lc};
//+
Point(7) = {200, 200, 0, lc};
//+
Point(8) = {100, 200, 0, lc};
//+
Point(9) = {100, 170, 0, lc};
//+
Point(10) = {90, 170, 0, lc};
//+
Point(11) = {90, 200, 0, lc};
//+
Point(12) = {0, 200, 0, lc};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 9};
//+
Line(9) = {9, 10};
//+
Line(10) = {10, 11};
//+
Line(11) = {11, 12};
//+
Line(12) = {12, 1};
//+
Line Loop(1) = {12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//+
Physical Line("1")={12};
//+
Physical Line("2")={1,2,3,4,5,7,8,9,10,11};
//+
Physical Line("3")={6};
//+
Plane Surface(1) = {1};
//+
Physical Surface("surf") = {1};

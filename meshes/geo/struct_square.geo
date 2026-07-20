// Structured TRIANGLE mesh of the unit square [0,1]^2 (transfinite, split quads).
// manapy boundary tags: 1=in(x=0) 2=out(x=1) 3=upper(y=1) 4=bottom(y=0).
//   gmsh -2 struct_square.geo -o struct_square.msh
N = 48;   // cells per direction

Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Line(1) = {1, 2};   // bottom (y=0)
Line(2) = {2, 3};   // right  (x=1)
Line(3) = {3, 4};   // top    (y=1)
Line(4) = {4, 1};   // left   (x=0)

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Transfinite Line {1, 3} = N + 1;
Transfinite Line {2, 4} = N + 1;
Transfinite Surface {1};

Physical Line("1") = {4};   // in     (x=0)
Physical Line("2") = {2};   // out    (x=1)
Physical Line("3") = {3};   // upper  (y=1)
Physical Line("4") = {1};   // bottom (y=0)
Physical Surface("1") = {1};

// Doubly-PERIODIC unit square [0,1]^2, triangles with matching opposite sides.
// manapy periodic tags: in=11(x=0) out=22(x=1) upper=33(y=1) bottom=44(y=0).
// Transfinite Line (aligned boundary nodes) + Periodic Line (slave side is an exact
// copy of its master) => conforming periodic pairing; free interior triangulation.
//   gmsh -2 periodic_square.geo -o periodic_square.msh
N = 48;   // cells per side

Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Line(1) = {1, 2};   // bottom (y=0)  44
Line(2) = {2, 3};   // right  (x=1)  22
Line(3) = {3, 4};   // top    (y=1)  33
Line(4) = {4, 1};   // left   (x=0)  11

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Transfinite Line {1, 2, 3, 4} = N + 1;   // aligned boundary nodes only

Physical Line("in",     11) = {4};
Physical Line("out",    22) = {2};
Physical Line("upper",  33) = {3};
Physical Line("bottom", 44) = {1};
Physical Surface("domain", 1) = {1};

Periodic Line {2} = {4} Translate {1, 0, 0};   // out   = in     + x
Periodic Line {3} = {1} Translate {0, 1, 0};   // upper = bottom + y

// Triply-PERIODIC unit cube [0,1]^3, structured HEXAHEDRA (transfinite + recombine).
// manapy periodic tags: in=11(x=0) out=22(x=1) upper=33(y=1) bottom=44(y=0)
//                       front=55(z=0) back=66(z=1).
// Hexes are conforming across opposite faces, so the periodic centroid pairing works
// (unlike a fully-transfinite TET cube, which manapy rejects -- see periodic_cube_tet.geo).
//   gmsh -3 periodic_cube_hex.geo -o periodic_cube_hex.msh
N = 16;   // cells per edge

Point(1) = {0,0,0, 1}; Point(2) = {1,0,0, 1};
Point(3) = {1,1,0, 1}; Point(4) = {0,1,0, 1};
Point(5) = {0,0,1, 1}; Point(6) = {1,0,1, 1};
Point(7) = {1,1,1, 1}; Point(8) = {0,1,1, 1};

Line(1) = {1,2}; Line(2) = {2,3}; Line(3) = {3,4}; Line(4) = {4,1};
Line(5) = {5,6}; Line(6) = {6,7}; Line(7) = {7,8}; Line(8) = {8,5};
Line(9) = {1,5}; Line(10) = {2,6}; Line(11) = {3,7}; Line(12) = {4,8};

Line Loop(1) = {1,2,3,4};        Plane Surface(1) = {1};   // z=0  front  (55)
Line Loop(2) = {5,6,7,8};        Plane Surface(2) = {2};   // z=1  back   (66)
Line Loop(3) = {1,10,-5,-9};     Plane Surface(3) = {3};   // y=0  bottom (44)
Line Loop(4) = {3,12,-7,-11};    Plane Surface(4) = {4};   // y=1  upper  (33)
Line Loop(5) = {4,9,-8,-12};     Plane Surface(5) = {5};   // x=0  in     (11)
Line Loop(6) = {2,11,-6,-10};    Plane Surface(6) = {6};   // x=1  out     (22)

Surface Loop(1) = {1,2,3,4,5,6};  Volume(1) = {1};

Transfinite Line "*" = N + 1;
Transfinite Surface "*";
Recombine Surface "*";
Transfinite Volume "*";

Physical Surface("in",     11) = {5};
Physical Surface("out",    22) = {6};
Physical Surface("upper",  33) = {4};
Physical Surface("bottom", 44) = {3};
Physical Surface("front",  55) = {1};
Physical Surface("back",   66) = {2};
Physical Volume("domain",   1) = {1};

Periodic Surface {6} = {5} Translate {1,0,0};   // out   = in     + x
Periodic Surface {4} = {3} Translate {0,1,0};   // upper = bottom + y
Periodic Surface {2} = {1} Translate {0,0,1};   // back  = front  + z

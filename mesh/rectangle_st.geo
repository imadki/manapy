lc = 0.2;
lcc = 0.2;

Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0,  0, lc} ;
Point(3) = {1, .5, 0, lc} ;
Point(4) = {0,  .5, 0, lc} ;
Point(5) = {1, .2, 0, lcc} ;
Point(6) = {0,  .2, 0, lcc} ;
Point(7) = {1, .3, 0, lcc} ;
Point(8) = {0,  .3, 0, lcc} ;

//+
Line(1) = {1, 6};
//+
Line(2) = {6, 5};
//+
Line(3) = {5, 2};
//+
Line(4) = {2, 1};
//+
Line(5) = {6, 8};
//+
Line(6) = {8, 7};
//+
Line(7) = {7, 5};
//+
Line(8) = {8, 4};
//+
Line(9) = {4, 3};
//+
Line(10) = {3, 7};
//+
Line Loop(1) = {9, 10, -6, 8};
//+
Plane Surface(1) = {1};
//+
Line Loop(2) = {6, 7, -2, 5};
//+
Plane Surface(2) = {2};
//+
Line Loop(3) = {4, 1, 2, 3};
//+
Plane Surface(3) = {3};

Periodic Line {4} ={9};

Physical Line("1") = {1,5, 8};
Physical Line("2") = {10,7, 3};
Physical Line("3") = {9};
Physical Line("4") = {4};

Physical Surface("1") = {1, 2, 3};

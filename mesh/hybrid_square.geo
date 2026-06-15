lc = 1;

Point(1) = {0, 0, 0, lc};
Point(2) = {.5, 0,  0, lc} ;
Point(3) = {.5, .5, 0, lc} ;
Point(4) = {0,  .5, 0, lc} ;
Point(5) = {1, .5, 0, lc} ;
Point(6) = {1,  0, 0, lc} ;

Point(7) = {.15, .25, 0, lc} ;
Point(8) = {0.15,  .15, 0, lc} ;
Point(9) = {.25, .15, 0, lc} ;
Point(10) = {.25,  .25, 0, lc} ;

//+
Line(1) = {4, 3};
//+
Line(2) = {3, 2};
//+
Line(3) = {2, 1};
//+
Line(4) = {1, 4};

Line Loop(1) = {1, 2, 3, 4};

//Plane Surface(1) = {1};

Line(5) = {3, 5};
//+
Line(6) = {5, 6};
//+
Line(7) = {6, 2};
//+
Line Loop(2) = {5, 6, 7, -2};
//+
//Plane Surface(2) = {2};

//+
Line(8) = {7, 10};
//+
Line(9) = {10, 9};
//+
Line(10) = {9, 8};
//+
Line(11) = {8, 7};

//+
Line Loop(3) = {8, 9, 10, 11};
//+
//Plane Surface(3) = {1,3};
//Plane Surface(1) = {3};



//+
Plane Surface(1) = {3};
//+
Plane Surface(2) = {1, 3};
//+
Plane Surface(3) = {2};

Transfinite Surface {1};
Recombine Surface {1};

Transfinite Surface {3};


Physical Line("1") = {4};
Physical Line("2") = {6};
Physical Line("3") = {1,5};
Physical Line("4") = {3,7};

Physical Surface("10") = {1,2,3} ;



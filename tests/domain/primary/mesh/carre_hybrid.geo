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


Point (15) = {0.7 , 0.25 ,0}; // centre
Point (11) = { .65 , .2 ,0};
Point (12) = {.75 , .2 ,0};
Point (13) = {.75 ,.3 ,0};
Point (14) = { .65 ,.3 ,0};

Circle (12) = {11 ,15 ,12};
Circle (13) = {12 ,15 ,13};
Circle (14) = {13 ,15 ,14};
Circle (15) = {14 ,15 ,11};


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
Plane Surface(3) = {1,3};
Plane Surface(1) = {3};

//+
Line Loop (4) = {12, 13, 14, 15};
//+
Plane Surface(4) = {2,4};
//Plane Surface(1) = {4};

Transfinite Surface {1};
Recombine Surface {1};

//Transfinite Surface {4};
//Recombine Surface {4};

Transfinite Line { 5,6,7,-2, 12, 13, 14,15} = 20 Using Progression 1;
Recombine Surface {4};

//Periodic Line {1} ={3};
//Periodic Line {2} ={4};

Physical Line("1", 1) = {4};
Physical Line("2", 2) = {6};
Physical Line("3", 3) = {1,5,12,13,14,15};
Physical Line("4", 4) = {3,7};

Physical Surface("10") = {1,2,3,4} ;


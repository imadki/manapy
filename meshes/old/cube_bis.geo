lc=.1;

Point(1) = {1, 1, 1, lc};
Point(2) = {1, 1, 0, lc};
Point(3) = {1, 0, 0, lc};
Point(4) = {0, 1, 1, lc};
Point(5) = {0, 1, 0, lc};
Point(6) = {0, 0, 1, lc};
Point(7) = {1, 0, 1, lc};
Point(8) = {0, 0, 0, lc};


Point(9) = {2, 0, 0, lc};
Point(10) = {2, 1, 1, lc};
Point(11) = {2, 0, 1, lc};
Point(12) = {2, 1, 0, lc};



Line(1) = {4, 1};
//+
Line(2) = {1, 1};
//+
Line(3) = {1, 7};
//+
Line(4) = {7, 6};
//+
Line(5) = {6, 8};
//+
Line(6) = {8, 5};
//+
Line(7) = {5, 5};
//+
Line(8) = {5, 2};
//+
Line(9) = {2, 2};
//+
Line(10) = {2, 2};
//+
Line(11) = {2, 1};
//+
Line(12) = {5, 4};
//+
Line(13) = {4, 6};
//+
Line(14) = {8, 3};
//+
Line(15) = {3, 2};
//+
Line(16) = {2, 2};
//+
Line(17) = {2, 2};
//+
Line(18) = {3, 3};
//+
Line(19) = {3, 7};

//+
Curve Loop(1) = {12, 13, 5, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {8, 11, -1, -12};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {15, -8, -6, 14};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {15, 11, 3, -19};


//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {1, 3, 4, -13};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {14, 19, 4, 5};
//+
Plane Surface(6) = {6};

//+
Surface Loop(1) = {2, 3, 4, 5, 6, 1};
//+
Volume(1) = {1};


//+
Line(20) = {10, 12};
//+
Line(21) = {12, 2};
//+
Line(22) = {10, 1};
//+
Line(23) = {11, 7};
//+
Line(24) = {10, 11};
//+
Line(25) = {9, 12};
//+
Line(26) = {3, 9};
//+
Line(27) = {9, 11};
//+
Curve Loop(7) = {23, -3, -22, 24};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {27, -24, 20, -25};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {26, 27, 23, -19};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {26, 25, 21, -15};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {22, -11, -21, -20};
//+
Plane Surface(11) = {11};
//+
Surface Loop(2) = {11, 7, 9, 10, 8, 4};
//+
Volume(2) = {2};

Transfinite Line "*" = 10;// Using Bump 0.25;
Transfinite Surface "*";
Recombine Surface {1,2,3,4,5,6};
Transfinite Volume {1};


Physical Surface(1) = {1};
//+
Physical Surface(2) = {8};
//+
Physical Surface(3) = {2,11};
//+
Physical Surface(4) = {6,9};

Physical Surface(5) = {5,7};

Physical Surface(6) = {3,10};


//+
Physical Volume(1) = {1,2};
//Physical Volume(2) = {2};



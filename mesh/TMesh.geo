lc = 0.001 ;
t = 1e-3 ;
//+
Point(1) = {0, 0, 0, lc};
//+
Point(2) = {83 * t, 0, 0, lc};
//+
Point(3) = {83 * t, 3 * t, 0, lc};
//+
Point(4) = {0, 3 * t, 0, lc};
//+
Point(5) = {34 * t, 3 * t, 0, lc};
//+
Point(6) = {34 *t , 9 * t, 0, lc};
//+
Point(7) = {40 *t , 9 * t, 0, lc};
//+
Point(8) = {49 *t , 9 * t, 0, lc};
//+
Point(9) = {43 *t , 9 * t, 0, lc};
//+
Point(10) = {49 *t , 3 * t, 0, lc};
//+
Point(11) = {43 *t , 49 * t, 0, lc};
//+
Point(12) = {40 *t , 49 * t, 0, lc};
//+
//+
Point(13) = {35.5*t, 1.4*t, 0, lc};
//+
Point(14) = {40*t, 0, 0, lc};
//+
Point(15) = {35.5*t, 7.4*t, 0, lc};
//+
Point(16) = {47.5*t, 7.4*t, 0, lc};
//+
Point(17) = {47.5*t, 1.4*t, 0, lc};
//+
Point(18) = {41.5*t, 7.4*t, 0, lc/2};
//+
Line(4) = {3, 10};
//+
Line(5) = {9, 11};
//+
Line(6) = {11, 12};
//+
Line(7) = {12, 7};
//+
Line(8) = {5, 4};
//+
Circle(9) = {5, 6, 7};
//+
Circle(10) = {9, 8, 10};
//+
Circle(12) = {13, 15, 18};
//+
Circle(13) = {18, 16, 17};
//+
Line(14) = {13, 17};
//+
Point(19) = {41.8*t, 0, 0, lc};
//+
Point(20) = {41.2*t, 0, 0, lc};
//+
Line(15) = {1, 20};
//+
Line(16) = {20, 19};
//+
Line(17) = {19, 2};
//+

//+
Point(21) = {0, 0.0014, 0, lc};
//+
Point(22) = {0.083, 0.0014, 0, lc};
//+
Line(18) = {1, 21};
//+
Line(19) = {21, 13};
//+
Line(20) = {17, 22};
//+
Line(21) = {22, 2};
//+
Line(22) = {4, 21};
//+
Line(23) = {3, 22};
//+
Curve Loop(1) = {7, -9, 8, 22, 19, 12, 13, 20, -23, 4, -10, 5, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {18, 19, 14, 20, 21, -17, -16, -15};
//+
Plane Surface(2) = {2};


//Make one surface structured.
//Transfinite Line{18, 19, 14, 20, 21, 17, 16, 15};
//Transfinite Surface{2};
//Recombine Surface{2};
//+
Physical Curve(1) = {6};
//+
//Physical Curve(2) = {};
//+
Physical Curve(3) = {7, 5, 9, 10, 12, 13, 14, 4, 17, 15, 8, 22, 23};
//+
Physical Curve(2) = {16, 18, 21};
//+
Physical Surface(4) = {1, 2};

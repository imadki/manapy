lc = 0.1; // Adjust the characteristic length as needed

// Define Points
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

// Create Transfinite Curve
Transfinite Line {1, 2, 3, 4} = 20 Using Progression 1;

// Create Transfinite Surface
Transfinite Surface {1};

// Recombine the Surface into Quadrilateral Elements
Recombine Surface {1};


SetFactory("OpenCASCADE");

// ======================
// PARAMETERS
// ======================
cubeSize  = 0.3;
cylRadius = 0.01;
cylHeight = 0.02;

cx = cubeSize/2;
cy = cubeSize/2;

// ======================
// GEOMETRY
// ======================
Box(1) = {0, 0, 0, cubeSize, cubeSize, cylHeight};
Cylinder(2) = {cx, cy, 0, 0, 0, cylHeight, cylRadius};

fluid[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// ======================
// MESH CONTROL
// ======================
Mesh.CharacteristicLengthMax = 0.2;
Mesh.CharacteristicLengthMin = 0.1;

Mesh.Algorithm3D = 10;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;

Mesh 3;


// PHYSICAL GROUPS (TO UPDATE AFTER GUI CHECK)

Physical Surface("1") = {7};   // inlet (cylinder wall)
Physical Surface("2") = {8, 9, 11, 13};    
Physical Surface("3") = {10};    // font
Physical Surface("4") = {12};    // back

Physical Volume(66) = {fluid(0)};

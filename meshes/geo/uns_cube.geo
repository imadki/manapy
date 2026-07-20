// Unstructured TETRAHEDRAL mesh of the unit cube [0,1]^3.
// manapy boundary faces: in(x=0) out(x=1) bottom(y=0) upper(y=1) front(z=0) back(z=1).
//   gmsh -3 uns_cube.geo -o uns_cube.msh
SetFactory("OpenCASCADE");
Mesh.MeshSizeMax = 0.06;   // target edge length

Box(1) = {0, 0, 0, 1, 1, 1};

Physical Surface("in")     = {1};   // x=0
Physical Surface("out")    = {2};   // x=1
Physical Surface("bottom") = {3};   // y=0
Physical Surface("upper")  = {4};   // y=1
Physical Surface("front")  = {5};   // z=0
Physical Surface("back")   = {6};   // z=1
Physical Volume("fluid")   = {1};

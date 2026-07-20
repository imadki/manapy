// Shared unit-cube mesh for Manapy and OpenFOAM (3D Laplace benchmark).
// 6 named faces -> Dirichlet patches; physical volume "fluid" -> OpenFOAM cellZone.
// 3D is native: gmshToFoam imports the tets directly (no extrusion/empty hack).
SetFactory("OpenCASCADE");
Box(1) = {0,0,0, 1,1,1};
Physical Surface("in")     = {1};   // x=0  (hot face, P=20)
Physical Surface("out")    = {2};   // x=1
Physical Surface("bottom") = {3};   // y=0
Physical Surface("upper")  = {4};   // y=1
Physical Surface("front")  = {5};   // z=0
Physical Surface("back")   = {6};   // z=1
Physical Volume("fluid")   = {1};

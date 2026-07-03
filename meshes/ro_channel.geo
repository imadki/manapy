// Reverse-osmosis feed channel.
//   x : cross-flow direction (feed in -> concentrate out)
//   y : channel height (membrane wall at y=0, top wall/symmetry at y=H)
//
// Physical-curve numbering matches manapy's face_name convention
// (1->in, 2->out, 3->upper, 4->bottom); see LocalDomainClass.py:653.
//   in     (1) = left edge  x=0      (feed inlet)
//   out    (2) = right edge x=L      (concentrate outlet)
//   upper  (3) = top edge   y=H      (impermeable wall / symmetry)
//   bottom (4) = bottom edge y=0     (MEMBRANE wall)

L  = 0.10;     // channel length  [m]
H  = 0.02;     // channel height  [m]
lc = 0.0016;   // target cell size [m]

Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, H, 0, lc};
Point(4) = {0, H, 0, lc};

Line(1) = {1, 2};   // bottom (y=0)  -> membrane
Line(2) = {2, 3};   // right  (x=L)  -> out
Line(3) = {3, 4};   // top    (y=H)  -> upper
Line(4) = {4, 1};   // left   (x=0)  -> in

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Curve(1)   = {4};   // in     = left
Physical Curve(2)   = {2};   // out    = right
Physical Curve(3)   = {3};   // upper  = top
Physical Curve(4)   = {1};   // bottom = membrane
Physical Surface(5) = {1};

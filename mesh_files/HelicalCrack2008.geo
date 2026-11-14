// Composite tube in traction with a half-disk notch, the helical crack of
// B. Bourdin, G. A. Francfort, J.-J. Marigo, The variational approach to fracture,
// J. Elasticity 91 (2008), Section 6, Fig. 30.
//
// A cylinder along e1 of length L = 30 and diameter 2, with an unbreakable core of
// diameter 1, perfectly bonded (Section 6 and Fig. 23). A half-disk hairline crack of
// radius a = .4, centered on the outer edge of the brittle cylinder, lies in its
// middle cross-section (Section 6, text of Fig. 30). Clamped at the left
// cross-section, pulled by a displacement at the right one.
//
// The coordinates are those of the mef90 run HelicalCrack of ~/runs/ATTEMPTS.md:
// x from -L/2 to L/2, the notch in x = 0 centered at (0, 0, R). With h = .15 and
// gmsh 4.15.2 the mesh has the counts of that run: 27,963 vertices, 106,863
// matrix and 34,463 core tetrahedra, 366 / 366 / 34 triangles.
//
// h = .15 is that of the mef90 runs of ~/runs/ATTEMPTS.md, not given in the paper,
// whose mesh has over 1.7 million elements (Section 8).
//
// The mesh is written in the Gmsh 2.2 format, the one MFEM reads; it is not
// kept in git. From the project directory:
//    gmsh -3 -setnumber h 0.15 mesh_files/HelicalCrack2008.geo -o mesh_files/HelicalCrack2008-h015.msh
//
// Physical tags, the element and the boundary attributes in MFEM:
//    1  brittle matrix
//    2  unbreakable core
//    10 left cross-section, x = -L/2 (clamped)
//    20 right cross-section, x = L/2 (displacement)
//    40 initial crack, inside the matrix
SetFactory("OpenCASCADE");

DefineConstant[ L = 30, R = 1., Rc = .5, a = .4, h = .15 ];

Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;
Mesh.MshFileVersion = 2.2;

eps = 1.e-3;

Cylinder(1) = {-L/2, 0, 0, L, 0, 0, R};
Cylinder(2) = {-L/2, 0, 0, L, 0, 0, Rc};

// disk of radius a centered on the outer edge, turned into the cross-section
// x = 0, then trimmed to the matrix, a half disk
disk = news;
Disk(disk) = {0, 0, R, a};
Rotate {{0, 1, 0}, {0, 0, R}, Pi/2} { Surface{disk}; }
crack() = BooleanIntersection{ Surface{disk}; Delete; }{ Volume{1}; };

v() = BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Surface{crack()}; Delete; };

core()   = Volume In BoundingBox{-L/2-eps, -Rc-eps, -Rc-eps, L/2+eps, Rc+eps, Rc+eps};
all()    = Volume{:};
matrix() = all();
matrix() -= core();

left()  = Surface In BoundingBox{-L/2-eps, -R-eps, -R-eps, -L/2+eps, R+eps, R+eps};
right() = Surface In BoundingBox{L/2-eps, -R-eps, -R-eps, L/2+eps, R+eps, R+eps};
// the bounding boxes of trimmed surfaces are loose, so leave some margin
crk()   = Surface In BoundingBox{-.2, -a-.2, R-a-.2, .2, a+.2, R+.2};
If (#crk() != 1)
   Error("Expected 1 surface for the initial crack, found %g", #crk());
EndIf

Physical Volume(1) = {matrix()};
Physical Volume(2) = {core()};
Physical Surface(10) = {left()};
Physical Surface(20) = {right()};
Physical Surface(40) = {crk()};

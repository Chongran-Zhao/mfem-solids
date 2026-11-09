// Cylinder with a central unbreakable shaft, capped at both ends, in traction under a body force:
// B. Bourdin, The variational formulation of brittle fracture: numerical implementation and extensions (2007), Section 3.
//
// A hairline crack, a disk of radius a centered on the outer edge (0, -R, L/2), lies in a plane containing
// the radial direction e2 and tilted by 30 degrees from the xy-plane: z - L/2 = x tan(tilt).
//
// R = 1 is read from Figs. 1 and 3 of the paper; its text says radius 2. The
// cap thickness tc is not given in the paper. The geometry is that of the
// mef90 runs in ~/runs/HelicalCrack2007L016.
//
// The mesh is written in the Gmsh 2.2 format, the one MFEM reads; it is not
// kept in git. From the project directory:
//    gmsh -3 -setnumber h 0.08 mesh_files/HelicalCrack2007.geo -o mesh_files/HelicalCrack2007-h008.msh
//
// Physical tags, the element and the boundary attributes in MFEM (mef90 sets):
//    1  (cs0001): brittle matrix
//    2  (cs0002): unbreakable shaft
//    3  (cs0003): bottom cap (z < 0)
//    4  (cs0004): top cap (z > L), carries the body force
//    20 (fs0020): bottom face of the bottom cap (clamped)
//    30 (fs0030): both faces of the top cap (ux = uy = 0 in the mef90 runs, so that it cannot rotate, not in the paper)
//    40 (fs0040): initial crack, inside the matrix
SetFactory("OpenCASCADE");

// tc is not given in the paper
DefineConstant[ L = 10, R = 1., Rc = .5, a = .4, tilt = 30., tc = .5, h = .08 ];

Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;
Mesh.MshFileVersion = 2.2;

eps = 1.e-3;

Cylinder(1) = {0, 0, 0, 0, 0, L, R};
Cylinder(2) = {0, 0, 0, 0, 0, L, Rc};
Cylinder(3) = {0, 0, -tc, 0, 0, tc, R};
Cylinder(4) = {0, 0, L, 0, 0, tc, R};

// disk of radius a centered on the outer edge, tilted about e2, then trimmed to the matrix
disk = news;
Disk(disk) = {0, -R, L/2, a};
Rotate {{0, 1, 0}, {0, -R, L/2}, -tilt*Pi/180} { Surface{disk}; }
crack() = BooleanIntersection{ Surface{disk}; Delete; }{ Volume{1}; };

v() = BooleanFragments{ Volume{1}; Delete; }{ Volume{2,3,4}; Surface{crack()}; Delete; };

shaft()  = Volume In BoundingBox{-Rc-eps, -Rc-eps, -eps, Rc+eps, Rc+eps, L+eps};
bottom() = Volume In BoundingBox{-R-eps, -R-eps, -tc-eps, R+eps, R+eps, eps};
top()    = Volume In BoundingBox{-R-eps, -R-eps, L-eps, R+eps, R+eps, L+tc+eps};
all()    = Volume{:};
matrix() = all();
matrix() -= shaft();
matrix() -= bottom();
matrix() -= top();

clamped() = Surface In BoundingBox{-R-eps, -R-eps, -tc-eps, R+eps, R+eps, -tc+eps};
topface() = Surface In BoundingBox{-R-eps, -R-eps, L+tc-eps, R+eps, R+eps, L+tc+eps};
topface() += Surface In BoundingBox{-R-eps, -R-eps, L-eps, R+eps, R+eps, L+eps};
// the bounding boxes of trimmed surfaces are loose, so leave some margin
crk()     = Surface In BoundingBox{-a-.2, -R-.2, L/2-a-.2, a+.2, -R+a+.2, L/2+a+.2};
If (#crk() != 1)
   Error("Expected 1 surface for the initial crack, found %g", #crk());
EndIf

Physical Volume(1) = {matrix()};
Physical Volume(2) = {shaft()};
Physical Volume(3) = {bottom()};
Physical Volume(4) = {top()};
Physical Surface(20) = {clamped()};
Physical Surface(30) = {topface()};
Physical Surface(40) = {crk()};

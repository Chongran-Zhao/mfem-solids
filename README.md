# Learning MFEM

A step-by-step record of me learning [MFEM](https://mfem.org/), starting from the
simplest problems I could find.

Each step is a small, self-contained program that solves one problem and introduces as
few new ideas as possible. Every step has a companion write-up on my blog that walks
through the code line by line; this repository holds the code those posts describe.

## Steps

| Program | Problem | Introduces | Write-up |
|---|---|---|---|
| `1d-elastostatics.cpp` | A bar fixed at one end, pulled by a uniform axial traction at the other | `Mesh`, `H1_FECollection`, `FiniteElementSpace`, essential and natural boundary conditions, `BilinearForm`, `LinearForm`, `FormLinearSystem`, `CGSolver` | [MFEM 01](https://chongran-zhao.github.io/notes/mfem-01/) |
| `3d-elastostatics.cpp` | A compressible Neo-Hookean beam clamped at one end, its other end moved vertically (finite deformation, Total Lagrangian) | reading a mesh file, vector-valued `FiniteElementSpace`, component-wise essential boundary conditions, a hand-written material model and `NonlinearFormIntegrator`, `NonlinearForm`, `NewtonSolver`, `ParaViewDataCollection` | [MFEM 02](https://chongran-zhao.github.io/notes/mfem-02/) |

`3d-elastostatics.cpp` uses the header-only classes in `include/`:

| Header | Content |
|---|---|
| `Vector_3D.hpp`, `Tensor2_3D.hpp`, `Tensor4_3D.hpp` | first-, second- and fourth-order tensors in 3D |
| `HyperelasticMaterialModel.hpp` | interface of a hyperelastic material: $\bm S$, $\mathbb C$, and from them $\bm P$ and $\mathbb A = \partial\bm P/\partial\bm F$ |
| `CompressibleNeoHookean.hpp` | the compressible Neo-Hookean model |
| `CompressibleHyperelasticIntegrator.hpp` | element residual and tangent, `AssembleElementVector` and `AssembleElementGrad` |

## Building

Requires CMake 3.20 or newer and MFEM built with CMake (developed against 4.10.1, with
MPI, hypre and METIS). The build looks for MFEM's build tree in `../../lib/mfem/build`;
change the `HINTS` in `CMakeLists.txt` if yours is elsewhere.

```bash
cmake -B build
```

```bash
cmake --build build
```

Each step builds its own executable:

```bash
./build/1d-elastostatics
```

```bash
./build/3d-elastostatics
```

`3d-elastostatics` reads the beam mesh from the path set in `mesh_file` at the top of
`main`; point it to your copy of `Beam_coarse-hex.mesh`. It writes the displacement to
`ParaView/3d-elastostatics/`; open the `.pvd` file and apply *Warp By Vector* to see the
deformed beam.

# mfem-solids

Finite-strain solid mechanics with [MFEM](https://mfem.org/): a Total Lagrangian
formulation with a hand-written material model and element integrator, Dirichlet
and traction conditions on named faces, load stepping and Newton's method.

## Programs

| Program | Reads | Writes |
|---|---|---|
| `read_mesh` | the mesh in `config.yaml` | `beam.mesh` with the six box faces named `left`, `right`, `front`, `back`, `bottom`, `top`, and a 3D view of them, `beam_boundary.html` |
| `driver` (`static_disp/`) | `beam.mesh`, the boundary conditions, loading and solver settings | the displacement of each load step, `results_gf/disp_XXXX.gf` |
| `driver` (`static_mixed/`) | the same, in the mixed displacement-pressure form with Taylor-Hood elements (`space.order` >= 2) | displacement `disp_XXXX.gf` and nodal pressure `pres_XXXX.gf` at each load step |
| `driver` (`dynamic_disp/`) | `beam.mesh`, the time steps, the boundary conditions and solver settings | displacement, velocity and acceleration of each time step, `disp_XXXX.gf`, `velo_XXXX.gf`, `acce_XXXX.gf`, and the time and Newton iterations of each step, `time.csv` |
| `driver` (`dynamic_mixed/`) | the same, in the mixed displacement-pressure form with Taylor-Hood elements (`space.order` >= 2) | the files of `dynamic_disp/` and nodal pressure `pres_XXXX.gf` at each time step |
| `vtu_writer` | `results_gf/` with its `mesh.mesh`, and the material | `results_vtu/`: the deformed mesh with displacement, pressure and element-center first and second Piola-Kirchhoff stresses; pressure is p(J) in the displacement form and the saved nodal field in the mixed form; in `dynamic_disp/` and `dynamic_mixed/` also velocity and acceleration, at the physical times of `time.csv`; open `results_vtu.pvd` in ParaView |
| `csv_writer` | `results_gf/` with its `mesh.mesh`, and the material | `results_csv/<face>.csv`: mean displacement, reaction force F, reference face area and face-mean pressure p on the faces and directions of `csv_writer` in `config.yaml`, at each step; the reaction is the formulation's residual on the constrained dofs of the face |

`scripts/plot_csv.m` (MATLAB) plots the CSV files of `csv_writer` against the load factor,
overlaying the result folders listed at its top.

`dynamic_disp/` solves the displacement dynamics, M a + F_int(u) = F_ext(t), with the
generalized-alpha method in physical time. Its nonlinear solver owns the mass matrix and
`TimeMethod_GenAlpha`, and solves each step for u_{n+1} by Newton's method at the
intermediate states; Newmark's formulas then give a_{n+1} and v_{n+1}. The initial
acceleration solves the equation of motion at t = 0. A prescribed displacement and a
traction may act together. The initial displacement is zero and the initial velocity is
`LoadData::initial_velo`. On the displacement-driven faces, the initial velocity is
`LoadData::velo_loading`, kept consistent with `disp_loading` by hand, and the initial
acceleration is zero, as in MixPERIGEE. Its `csv_writer` writes the face columns of the static one at the
physical times, with reactions M a + F_int - F_ext that include the inertia, and
`energy.csv`, the kinetic, strain and total energies of the whole body.

`dynamic_mixed/` solves the mixed dynamics, M a + F_int(u,p) = F_ext(t) with J(u) = J(p)
weakly, by the same second-order generalized-alpha method: u_{n+1} and p_{n+1} are the
unknowns of Newton's method, both equations hold at the intermediate states, and the
pressure has no inertia. The initial pressure is zero, consistent with the zero initial
displacement. Its `csv_writer`
writes the columns of the one of `dynamic_disp/`, with the strain energy of the mixed form,
int Psi_vol(J(p)) + Psi_ich(F) dV. Its default `rho_inf` is 0.5: with 1, the pressure
keeps the oscillations of the high frequencies of Q2 that dt does not resolve.

Each formulation has its own `config.yaml`; the material, the density included, is in `include/material/MaterialModelData.hpp`,
and the prescribed displacements and tractions are in `include/boundary/LoadData.hpp`.

The CSV column `area` is the undeformed reference area of the reported face.
The CSV column `p` is the reference-area average of pressure on each reported
face, with compression positive. The displacement writer evaluates p(J) from
the adjacent volume element at boundary quadrature points; the mixed writer
integrates the saved pressure field. Pressure is scalar and is written once per
face, independently of the reported directions.

Both drivers follow the same structure: global assembly, a nonlinear solver for
one load step, and a time solver for load stepping and output. In the mixed form,
the nonlinear solver packs the displacement and the pressure into one block vector
for Newton's method, and the block tangent is copied into one matrix for MUMPS.

## Parallel runs

The drivers and `csv_writer` run on MPI ranks, e.g. `mpirun -np 2 ./driver`; without
`mpirun` they run on one rank. Each rank holds part of the mesh, and Newton's method works
on the dofs each rank owns; MUMPS, the parallel direct solver, solves the tangents and
analyzes their sparsity once. Rank 0 gathers the results onto the mesh of all ranks and
saves it as `results_gf/mesh.mesh`; its element order is that of the ranks, so
`vtu_writer`, which is serial, and `csv_writer` read this mesh instead of `beam.mesh`.

## Setting up a problem

- **Faces.** `read_mesh` names the six faces of the box; the boundary conditions refer
  to them by name.
- **Loading.** The load is raised in `loading.load_steps` equal steps of the pseudo time
  t = n / N. It is either a prescribed displacement or a traction, not both:
  - displacement: list the driven faces and directions in `Dirichlet.disp_bc`, leave
    `Neumann.faces` empty, and give the displacement in `LoadData::disp_loading`;
  - traction: list the loaded faces in `Neumann.faces`, leave `Dirichlet.disp_bc`
    empty, and give the nominal traction in `LoadData::surface_traction`.

  `loading.type` (`displacement` or `traction`) is checked against these two lists.
- **Supports.** `Dirichlet.fixed_bc` fixes one direction of a face per entry.
- **Output.** `csv_writer` lists the faces and directions to report, and `output` the
  three result folders.

## Notes

- `vtu_writer` writes the values at the vertices only: with `space.order: 2` the midside
  nodes are left out and ParaView draws the elements as linear.
- The mixed driver needs a volumetric model with the pressure form J(p): `Quadratic`, or
  `Incompressible` for J = 1, set in `MaterialModelData.hpp`. `SimoPister` has none and
  aborts.
- With `Incompressible`, the pressure is fixed only up to a constant when displacements
  are prescribed on every face; the drivers do not handle that case.
- Linear hexahedra lock in bending: on a coarse mesh the stresses are poor. Quadratic
  elements (`space.order: 2`) remove most of this.
- Large prescribed displacements on a refined mesh may need more load steps for Newton
  to converge.

## Building and running

### Dependencies

The project was developed on macOS (Apple silicon) with the versions below. The libraries
are built in `../../lib`, next to `projects/`; change the paths in `cmake/mfem-solids.cmake`
if yours are elsewhere.

| Package | Version | Where |
|---|---|---|
| CMake | 3.20 or newer | Homebrew |
| Open MPI | 5.0.10 | Homebrew |
| gfortran, OpenBLAS, ScaLAPACK | Homebrew `gcc`, `openblas`, `scalapack` | Homebrew, for MUMPS |
| METIS | 5.1 | `lib/metis`, library in `lib/metis/lib/libmetis.a` |
| hypre | 3.2.0 | `lib/hypre`, installed in `lib/hypre/src/hypre` |
| MUMPS | 5.8.1 | sources in `lib/mumps-src`, libraries and headers in `lib/mumps` |
| yaml-cpp | | sources in `lib/yaml-cpp/src`, installed into `lib/yaml-cpp` |
| MFEM | 4.10.1 (development) | `lib/mfem`, built in `lib/mfem/build` |
| ADIOS2 | optional | Homebrew `adios2`; MFEM is built with it for other projects, mfem-solids does not use it |

```
MFEM/
├── lib/
│   ├── metis/  hypre/  mumps-src/  mumps/  yaml-cpp/
│   └── mfem/           (built in mfem/build)
└── projects/
    └── mfem-solids/
```

`cmake/mfem-solids.cmake` finds MFEM in `lib/mfem/build` and yaml-cpp in
`lib/yaml-cpp/lib/cmake/yaml-cpp`, and stops unless MFEM has MPI and MUMPS.

### Building the dependencies

The commands below are those the libraries of this machine were built with.

**Homebrew packages**

```bash
brew install cmake open-mpi gcc openblas scalapack adios2
```

**METIS 5.1**, in `lib/metis`, a static library with the GKlib it ships with. Its old
`CMakeLists.txt` needs `CMAKE_POLICY_VERSION_MINIMUM` with CMake 4:

```bash
cd lib/metis && cmake -S . -B lib -DCMAKE_BUILD_TYPE=Release -DGKLIB_PATH=$PWD/GKlib -DSHARED=FALSE -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build lib && cp lib/libmetis/libmetis.a lib/
```

MFEM and MUMPS take `lib/metis/lib/libmetis.a` and the headers of `lib/metis/include`.

**hypre 3.2.0**, in `lib/hypre/src`, installed into `lib/hypre/src/hypre`:

```bash
cd lib/hypre/src && ./configure --disable-fortran CC=mpicc CXX=mpicxx && make -j8 && make install
```

**MUMPS 5.8.1**, in `lib/mumps-src/MUMPS_5.8.1`, single and double precision, with METIS
and PORD orderings. Its `Makefile.inc` sets

```make
LMETISDIR = /Users/chongran/MFEM/lib/metis/lib
IMETIS    = -I/Users/chongran/MFEM/lib/metis/include
ORDERINGSF = -Dmetis -Dpord
CC = mpicc
FC = mpif90
FL = mpif90
LAPACK  = -L/opt/homebrew/opt/openblas/lib -lopenblas
SCALAP  = -L/opt/homebrew/opt/scalapack/lib -lscalapack
LIBPAR  = $(SCALAP) $(LAPACK)
LIBBLAS = -L/opt/homebrew/opt/openblas/lib -lopenblas
OPTF    = -O2 -fPIC -fallow-argument-mismatch
OPTC    = -O2 -fPIC
```

then the libraries and the headers go to `lib/mumps`:

```bash
cd lib/mumps-src/MUMPS_5.8.1 && make s d && mkdir -p ../../mumps && cp -R lib include ../../mumps/
```

**yaml-cpp**, with its sources in `lib/yaml-cpp/src`, static, installed into
`lib/yaml-cpp`:

```bash
cd lib/yaml-cpp && cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD -DYAML_BUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build build && cmake --install build
```

**MFEM**, in `lib/mfem`, with MPI, METIS, hypre, MUMPS and ADIOS2, built in
`lib/mfem/build`. MUMPS needs the Fortran runtime, ScaLAPACK and OpenBLAS named
explicitly. MFEM's `FindADIOS2.cmake` looks for the library names of old ADIOS2 releases,
so the libraries of the current one are given directly, with `ADIOS2_USE_MPI`:

```bash
cd lib/mfem && mkdir -p build && cd build && A=/opt/homebrew/opt/adios2 && cmake .. -DCMAKE_BUILD_TYPE=Release -DMFEM_USE_MPI=ON -DMFEM_USE_METIS_5=ON -DMETIS_DIR=$PWD/../../metis -DHYPRE_DIR=$PWD/../../hypre/src/hypre -DMFEM_USE_MUMPS=ON -DMUMPS_DIR=$PWD/../../mumps "-DMUMPS_REQUIRED_PACKAGES=MPI;MPI_Fortran;METIS" "-DMUMPS_REQUIRED_LIBRARIES=/opt/homebrew/opt/scalapack/lib/libscalapack.dylib;/opt/homebrew/opt/openblas/lib/libopenblas.dylib;/opt/homebrew/opt/gcc/lib/gcc/current/libgfortran.dylib" -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran -DMFEM_USE_ADIOS2=ON -DADIOS2_DIR=$A "-DADIOS2_LIBRARY=$A/lib/libadios2_cxx_mpi.dylib;$A/lib/libadios2_cxx.dylib;$A/lib/libadios2_core_mpi.dylib;$A/lib/libadios2_core.dylib" -DCMAKE_CXX_FLAGS=-DADIOS2_USE_MPI && make -j8 mfem
```

Without ADIOS2, leave out the four `ADIOS2` options and `CMAKE_CXX_FLAGS`. The MFEM sources
here carry a local change to `linalg/mumps.cpp` and `linalg/mumps.hpp`, which mfem-solids
does not use.

### Building and running the cases

The programs of each case are in their own folder, a CMake project of its own:
`static_disp/`, `static_mixed/`, `dynamic_disp/`, `dynamic_mixed/` and `sphere_rotation/`;
the settings they share are in `cmake/mfem-solids.cmake`. Each is built and run in its
folder, e.g. `static_disp/`:

```bash
cd static_disp && cmake -B build && cmake --build build
```

```bash
cd build && ./read_mesh && mpirun -np 2 ./driver && ./vtu_writer && mpirun -np 2 ./csv_writer
```

`static_mixed/`, `dynamic_disp/` and `dynamic_mixed/` build and run the same four programs
in the same way. The mixed folders need `space.order >= 2`; their drivers save displacement
and pressure, and their postprocessors read both fields.

Each folder has its own `config.yaml`, which CMake copies into its `build/`, again whenever
it changes; each program reads the `config.yaml` of the directory it runs in, or the file
given as its first argument. `mesh.file` is relative to the project directory, whose
`mesh_files/` the beam cases share.

**`sphere_rotation/`** is a gelatin ball in an acrylic skull shell, spun about z by its
outer surface, in the mixed form. Its folder holds its own `LoadData.hpp`,
`MaterialModelData.hpp`, nonlinear solver, `VTK_Tools.hpp` and writers, which replace the
shared ones for its programs:

- `read_mesh` generates the cubed-sphere mesh from the `mesh` section of its `config.yaml`;
  `inner_elements`, `radial_elements` and `skull_elements` set the mesh size.
- `driver` solves the linear systems by GMRES with a block preconditioner instead of MUMPS,
  which gets faster with more ranks; 4 ranks are about three times faster than 1.
- `vtu_writer` cuts each element into `output.vtu_refine`^3 hexahedra and writes every
  `output.vtu_interval`-th step, with the vectors and tensors in spherical components; the
  files grow as (`vtu_refine` + 1)^3.

```bash
cd sphere_rotation && cmake -B build && cmake --build build
```

```bash
cd build && ./read_mesh && mpirun -np 4 ./driver && ./vtu_writer && mpirun -np 2 ./csv_writer
```

With the default mesh and `dt: 1.0e-4`, the 300 steps take about 25 minutes on 4 ranks of
an Apple M2; every step is saved, about 7 MB of `results_gf/` and, with all steps written,
25 MB of `results_vtu/` each.

### Troubleshooting

- **`Could not find a package configuration file provided by "MFEM"`**: `lib/mfem/build` is
  missing, for example removed by a disk cleaner that deletes build folders; build MFEM
  again with the command above. The programs already built still run, since they link MFEM
  statically.
- **`Could NOT find ADIOS2 (missing: ADIOS2_LIBRARY)`**: give the ADIOS2 libraries
  directly, as in the MFEM command above.
- **METIS or yaml-cpp: `Compatibility with CMake < 3.5 has been removed`**: add
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
- **MUMPS gets slower with more ranks**: on the Apple M2 used here, its factorization took
  longer on 2 and 4 ranks than on 1, with about a quarter of the time spent waiting on MPI;
  use one rank there, or the iterative solver of `sphere_rotation/`.

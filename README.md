# mfem-solids

Finite-strain solid mechanics and phase-field fracture with [MFEM](https://mfem.org/).

## Dependencies

The project was developed on macOS (Apple silicon) with the versions below. The libraries
are built in `../lib`, next to `mfem-solids/`; change the paths in `cmake/mfem-solids.cmake`
if yours are elsewhere.

| Package | Version | Usage |
|---|---|---|
| CMake | 3.20 or newer | builds the libraries and the examples |
| Open MPI | 5.0.10 | parallel runs: the MPI compilers and `mpirun` |
| gfortran, OpenBLAS, ScaLAPACK | Homebrew `gcc`, `openblas`, `scalapack` | for PETSc and MUMPS: MUMPS is written in Fortran, and its dense kernels use BLAS, LAPACK and ScaLAPACK |
| PETSc | 3.24.6 | the solver of the phase field of fracture with the bounds 0 <= alpha <= 1, `SNESVINEWTONRSLS`; it also provides METIS, hypre and MUMPS below, which it downloads and builds, so that one copy of each is linked |
| METIS | 5.1.0, from PETSc | splits the mesh among the ranks (`ParMesh`), and orders the matrix for MUMPS |
| hypre | 3.0.0, `da9f93f8d`, from PETSc | the parallel matrices of MFEM (`HypreParMatrix`), and the BoomerAMG preconditioners of the GMRES solvers of the linear systems |
| MUMPS | 5.8.1, from PETSc | parallel direct solver of the linear systems, instead of GMRES with `linear_solver: mumps` in a solver section of the input; MFEM has only its interface |
| yaml-cpp | 0.9.0 | reads the inputs, `config*.yaml` |
| MFEM | 4.10.1, master `45799eb` | finite element spaces, meshes, assembly, the Krylov solvers and the interface to PETSc |

```
MFEM/
├── lib/
│   ├── perigee/petsc-3.24.6-opt/   (with METIS, hypre and MUMPS)
│   ├── yaml-cpp/
│   └── mfem/           (built in mfem/build_petsc)
└── mfem-solids/
```

`cmake/mfem-solids.cmake` finds MFEM in `lib/mfem/build_petsc` and yaml-cpp in
`lib/yaml-cpp/lib/cmake/yaml-cpp`, and stops unless MFEM has MPI, MUMPS and PETSc.

## Installing the dependencies

Run the commands below in order, all from `MFEM/`, the folder that will hold `lib/` and
`mfem-solids/`.

### 1. Homebrew packages

CMake, Open MPI, and the Fortran compiler, BLAS and ScaLAPACK that PETSc and MUMPS need:

```bash
brew install cmake open-mpi gcc openblas scalapack
```

### 2. PETSc 3.24.6, with METIS, hypre and MUMPS

Download it into `lib/petsc-src`:

```bash
mkdir -p lib/petsc-src && (cd lib/petsc-src && curl -LO https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.24.6.tar.gz && tar xzf petsc-3.24.6.tar.gz)
```

Configure it, downloading METIS, ParMETIS, hypre, MUMPS and HDF5, and install it into
`lib/perigee/petsc-3.24.6-opt`, then remove the sources:

```bash
(cd lib/petsc-src/petsc-3.24.6 && ./configure --prefix=$PWD/../../perigee/petsc-3.24.6-opt --with-mpi-dir=/opt/homebrew/opt/open-mpi --with-blaslapack-dir=/opt/homebrew/opt/openblas --with-scalapack-dir=/opt/homebrew/opt/scalapack --download-hypre --download-mumps --download-metis --download-parmetis --download-hdf5 --with-x=0 --with-debugging=no COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 && make all && make install) && rm -rf lib/petsc-src
```

### 3. yaml-cpp 0.9.0

Download it into `lib/yaml-cpp/src`:

```bash
git clone --branch yaml-cpp-0.9.0 https://github.com/jbeder/yaml-cpp.git lib/yaml-cpp/src
```

Build it as a static library, install it into `lib/yaml-cpp`, and remove the build folder:

```bash
(cd lib/yaml-cpp && cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD -DYAML_BUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build build && cmake --install build) && rm -rf lib/yaml-cpp/build
```

### 4. MFEM 4.10.1

Download it into `lib/mfem`, at the commit used here:

```bash
(cd lib && git clone https://github.com/mfem/mfem.git && git -C mfem checkout 45799eb)
```

Configure it with MPI and with PETSc, and with the METIS, hypre and MUMPS of PETSc, and build
it in `lib/mfem/build_petsc`:

```bash
P=$PWD/lib/perigee/petsc-3.24.6-opt; mkdir -p lib/mfem/build_petsc && (cd lib/mfem/build_petsc && cmake .. -DCMAKE_BUILD_TYPE=Release -DMFEM_USE_MPI=ON -DMFEM_USE_METIS_5=ON -DMETIS_DIR=$P -DHYPRE_DIR=$P -DMFEM_USE_MUMPS=ON -DMUMPS_DIR=$P "-DMUMPS_REQUIRED_PACKAGES=MPI;MPI_Fortran;METIS" "-DMUMPS_REQUIRED_LIBRARIES=/opt/homebrew/opt/scalapack/lib/libscalapack.dylib;/opt/homebrew/opt/openblas/lib/libopenblas.dylib;/opt/homebrew/opt/gcc/lib/gcc/current/libgfortran.dylib" -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran -DMFEM_USE_PETSC=ON -DPETSC_DIR=$P -DPETSC_ARCH= && make -j8 mfem)
```

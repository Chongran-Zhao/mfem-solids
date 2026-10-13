# mfem-solids

Finite-strain solid mechanics with [MFEM](https://mfem.org/).

## Dependencies

The project was developed on macOS (Apple silicon) with the versions below. The libraries
are built in `../lib`, next to `mfem-solids/`; change the paths in `cmake/mfem-solids.cmake`
if yours are elsewhere.

| Package | Version | Usage |
|---|---|---|
| CMake | 3.20 or newer | builds the libraries and the cases |
| Open MPI | 5.0.10 | parallel runs: the MPI compilers and `mpirun` |
| gfortran, OpenBLAS, ScaLAPACK | Homebrew `gcc`, `openblas`, `scalapack` | for MUMPS only: it is written in Fortran, and its dense kernels use BLAS, LAPACK and ScaLAPACK |
| METIS | 5.1.0 | splits the mesh among the ranks (`ParMesh`), and orders the matrix for MUMPS |
| hypre | 3.2.0, master `5b58261` | the parallel matrices of MFEM (`HypreParMatrix`), and the BoomerAMG preconditioner of `sphere_rotation/` |
| MUMPS | 5.8.1 | parallel direct solver of the linear systems of the beam cases; MFEM has only its interface |
| yaml-cpp | 0.9.0 | reads `config.yaml` |
| MFEM | 4.10.1, master `45799eb` | finite element spaces, meshes, assembly and the Krylov solvers |

```
MFEM/
├── lib/
│   ├── metis/  hypre/  mumps-src/  mumps/  yaml-cpp/
│   └── mfem/           (built in mfem/build)
└── mfem-solids/
```

`cmake/mfem-solids.cmake` finds MFEM in `lib/mfem/build` and yaml-cpp in
`lib/yaml-cpp/lib/cmake/yaml-cpp`, and stops unless MFEM has MPI and MUMPS.

## Installing the dependencies

Install them in the order below: MUMPS needs METIS, and MFEM needs all the others. Each
command starts in `MFEM/`, the folder that holds `lib/` and `mfem-solids/`; these are the
commands and options the libraries of this machine were built with.

### 1. Homebrew packages

CMake, Open MPI, and gfortran, OpenBLAS and ScaLAPACK for MUMPS:

```bash
brew install cmake open-mpi gcc openblas scalapack
```

### 2. METIS 5.1.0

Download it into `lib/metis`:

```bash
mkdir -p lib && cd lib && curl -LO https://github.com/mfem/tpls/raw/gh-pages/metis-5.1.0.tar.gz && tar xzf metis-5.1.0.tar.gz && mv metis-5.1.0 metis && rm metis-5.1.0.tar.gz
```

Build it as a static library, with the GKlib it ships with; its old `CMakeLists.txt` needs
`CMAKE_POLICY_VERSION_MINIMUM` with CMake 4:

```bash
cd lib/metis && cmake -S . -B lib -DCMAKE_BUILD_TYPE=Release -DGKLIB_PATH=$PWD/GKlib -DSHARED=FALSE -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build lib && cp lib/libmetis/libmetis.a lib/
```

This gives `lib/metis/lib/libmetis.a`, with the headers in `lib/metis/include`.

### 3. hypre 3.2.0

Download it into `lib/hypre`, at the commit used here:

```bash
cd lib && git clone https://github.com/hypre-space/hypre.git && git -C hypre checkout 5b58261
```

Build it with the MPI compilers, without Fortran:

```bash
cd lib/hypre/src && ./configure --disable-fortran CC=mpicc CXX=mpicxx && make -j8 && make install
```

This installs it into `lib/hypre/src/hypre`, with `lib/libHYPRE.a` and `include/`.

### 4. MUMPS 5.8.1

Download it into `lib/mumps-src`:

```bash
mkdir -p lib/mumps-src && cd lib/mumps-src && curl -LO https://mumps-solver.org/MUMPS_5.8.1.tar.gz && tar xzf MUMPS_5.8.1.tar.gz
```

Start its `Makefile.inc` from the generic one:

```bash
cd lib/mumps-src/MUMPS_5.8.1 && cp Make.inc/Makefile.inc.generic Makefile.inc
```

and set these lines in it, with the path of your `MFEM/` folder: METIS and PORD orderings,
the MPI compilers, OpenBLAS and ScaLAPACK of Homebrew, and `-fallow-argument-mismatch`,
which recent gfortran needs for the MPI calls of MUMPS:

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

Build the single and double precision libraries, and copy them and the headers to
`lib/mumps`:

```bash
cd lib/mumps-src/MUMPS_5.8.1 && make s d && mkdir -p ../../mumps && cp -R lib include ../../mumps/
```

This gives `lib/mumps/lib/libdmumps.a`, `libmumps_common.a` and `libpord.a`, with the
headers in `lib/mumps/include`; `lib/mumps-src` may then be removed.

### 5. yaml-cpp 0.9.0

Download it into `lib/yaml-cpp/src`:

```bash
git clone --branch yaml-cpp-0.9.0 https://github.com/jbeder/yaml-cpp.git lib/yaml-cpp/src
```

Build it as a static library and install it into `lib/yaml-cpp`:

```bash
cd lib/yaml-cpp && cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD -DYAML_BUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build build && cmake --install build
```

This gives `lib/yaml-cpp/lib/libyaml-cpp.a` and `lib/yaml-cpp/include`;
`lib/yaml-cpp/build` may then be removed.

### 6. MFEM 4.10.1

Download it into `lib/mfem`, at the commit used here:

```bash
cd lib && git clone https://github.com/mfem/mfem.git && git -C mfem checkout 45799eb
```

Configure it in `lib/mfem/build` with MPI, and the paths of METIS, hypre and MUMPS. MUMPS
also needs ScaLAPACK, OpenBLAS and the Fortran runtime named explicitly:

```bash
cd lib/mfem && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DMFEM_USE_MPI=ON -DMFEM_USE_METIS_5=ON -DMETIS_DIR=$PWD/../../metis -DHYPRE_DIR=$PWD/../../hypre/src/hypre -DMFEM_USE_MUMPS=ON -DMUMPS_DIR=$PWD/../../mumps "-DMUMPS_REQUIRED_PACKAGES=MPI;MPI_Fortran;METIS" "-DMUMPS_REQUIRED_LIBRARIES=/opt/homebrew/opt/scalapack/lib/libscalapack.dylib;/opt/homebrew/opt/openblas/lib/libopenblas.dylib;/opt/homebrew/opt/gcc/lib/gcc/current/libgfortran.dylib" -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran
```

The output should list `Found HYPRE`, `Found METIS` and `Found MUMPS`. Then build it:

```bash
cd lib/mfem/build && make -j8 mfem
```

This gives `lib/mfem/build/libmfem.a`. `lib/mfem/build/MFEMTargets.cmake` records the
absolute paths of hypre, METIS and MUMPS, so mfem-solids only needs to find MFEM; after
moving `lib/`, configure and build MFEM again.

## Building the cases

Each case is a CMake project of its own: `static_disp/`, `static_mixed/`, `dynamic_disp/`,
`dynamic_mixed/` and `sphere_rotation/`; the settings they share are in
`cmake/mfem-solids.cmake`. Each is built in its folder, e.g. `static_disp/`:

```bash
cd static_disp && cmake -B build && cmake --build build
```

and run in its `build/`, which CMake gives a copy of the case's `config.yaml`:

```bash
cd build && ./read_mesh && mpirun -np 2 ./driver && ./vtu_writer
```

## Troubleshooting

- **`Could not find a package configuration file provided by "MFEM"`**: `lib/mfem/build` is
  missing, for example removed by a disk cleaner that deletes build folders; build MFEM
  again with the command above. The programs already built still run, since they link MFEM
  statically.
- **METIS or yaml-cpp: `Compatibility with CMake < 3.5 has been removed`**: add
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.

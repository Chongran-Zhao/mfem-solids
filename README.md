# mfem-solids

Finite-strain solid mechanics with [MFEM](https://mfem.org/).

## Dependencies

The project was developed on macOS (Apple silicon) with the versions below. The libraries
are built in `../lib`, next to `mfem-solids/`; change the paths in `cmake/mfem-solids.cmake`
if yours are elsewhere.

| Package | Version | Where |
|---|---|---|
| CMake | 3.20 or newer | Homebrew |
| Open MPI | 5.0.10 | Homebrew |
| gfortran, OpenBLAS, ScaLAPACK | Homebrew `gcc`, `openblas`, `scalapack` | Homebrew, for MUMPS |
| METIS | 5.1.0 | `lib/metis`, library in `lib/metis/lib/libmetis.a` |
| hypre | 3.2.0, master `5b58261` | `lib/hypre`, installed in `lib/hypre/src/hypre` |
| MUMPS | 5.8.1 | sources in `lib/mumps-src`, libraries and headers in `lib/mumps` |
| yaml-cpp | 0.9.0 | sources in `lib/yaml-cpp/src`, installed into `lib/yaml-cpp` |
| MFEM | 4.10.1, master `45799eb` | `lib/mfem`, built in `lib/mfem/build` |
| ADIOS2 | optional | Homebrew `adios2`; MFEM is built with it for other projects, mfem-solids does not use it |

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

The commands below are those the libraries of this machine were built with; each starts
in `MFEM/`, the folder that holds `lib/` and `mfem-solids/`.

**Homebrew packages**

```bash
brew install cmake open-mpi gcc openblas scalapack adios2
```

**Sources**, at the versions above; hypre and MFEM at the commits used here:

```bash
mkdir -p lib && cd lib && curl -LO https://github.com/mfem/tpls/raw/gh-pages/metis-5.1.0.tar.gz && tar xzf metis-5.1.0.tar.gz && mv metis-5.1.0 metis
```

```bash
cd lib && git clone https://github.com/hypre-space/hypre.git && git -C hypre checkout 5b58261
```

```bash
mkdir -p lib/mumps-src && cd lib/mumps-src && curl -LO https://mumps-solver.org/MUMPS_5.8.1.tar.gz && tar xzf MUMPS_5.8.1.tar.gz
```

```bash
git clone --branch yaml-cpp-0.9.0 https://github.com/jbeder/yaml-cpp.git lib/yaml-cpp/src
```

```bash
cd lib && git clone https://github.com/mfem/mfem.git && git -C mfem checkout 45799eb
```

**METIS 5.1.0**, in `lib/metis`, a static library with the GKlib it ships with. Its old
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
and PORD orderings. Copy `Make.inc/Makefile.inc.generic` to `Makefile.inc` and set in it,
with the path of your `MFEM/` folder,

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

**yaml-cpp 0.9.0**, with its sources in `lib/yaml-cpp/src`, static, installed into
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

Without ADIOS2, leave out the four `ADIOS2` options and `CMAKE_CXX_FLAGS`.

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
- **`Could NOT find ADIOS2 (missing: ADIOS2_LIBRARY)`**: give the ADIOS2 libraries
  directly, as in the MFEM command above.
- **METIS or yaml-cpp: `Compatibility with CMake < 3.5 has been removed`**: add
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.

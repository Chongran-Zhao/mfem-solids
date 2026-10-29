# mfem-solids

Finite-strain solid mechanics with [MFEM](https://mfem.org/).

## Cases

Each case in `examples/` is a CMake project of its own, with one driver and one input per
formulation and time method: the driver sets the nonlinear solver, and the input the mesh,
the material settings, the time steps, the boundary conditions and the results folders,
`results_<suffix of the driver>_{gf,vtu,csv}`.

| Case | Driver | Input | Problem |
|---|---|---|---|
| `hyperelasticity_disp` | `driver_static` | `config_static.yaml` | beam, displacement form, statics |
| | `driver_dynamic` | `config_dynamic.yaml` | beam, displacement form, generalized-alpha of second order |
| `hyperelasticity_mixed` | `driver_TaylorHood_static` | `config_TaylorHood_static.yaml` | beam, mixed u/p form, Taylor-Hood elements, statics |
| | `driver_TaylorHood_dynamic` | `config_TaylorHood_dynamic.yaml` | beam, mixed u/p form, Taylor-Hood elements, generalized-alpha of second order |
| | `driver_vms_dynamic` | `config_vms_dynamic.yaml` | beam, rate form of the mixed u/p form, equal-order elements with VMS, generalized-alpha of first order |
| `sphere_rotation` | `driver_TaylorHood_dynamic` | `config_TaylorHood_dynamic.yaml` | gelatin ball in a skull shell, spun up by the measured rotation of its surface, Taylor-Hood elements |
| | `driver_vms_dynamic` | `config_vms_dynamic.yaml` | the same, equal-order elements with VMS |

Besides the drivers, each case has:
- `read_mesh`, which prepares the mesh of the input;
- `vtu_writer`, which writes the saved fields and the stresses as a VTU series for ParaView,
  on MPI ranks;
- in the beam cases, `csv_writer`: mean displacement and pressure, reaction forces and
  energies;
- in `sphere_rotation`, `probe_disp`: u_phi at the measured radii, and the pvpython scripts
  of `scripts/`.

The drivers read their own input by default; the other programs take it as first argument.
For example, the Taylor-Hood statics of the beam, from `examples/hyperelasticity_mixed`:

```bash
cmake -B build && cmake --build build
```

```bash
cd build && ./read_mesh config_TaylorHood_static.yaml && mpirun -np 2 ./driver_TaylorHood_static
```

```bash
mpirun -np 2 ./vtu_writer config_TaylorHood_static.yaml && mpirun -np 2 ./csv_writer config_TaylorHood_static.yaml
```

The ParaView file is `results_TaylorHood_static_vtu/results_TaylorHood_static_vtu.pvd`.

The code the cases share is in `include/`:

| Folder | Content |
|---|---|
| `assembly` | local (element) and global assemblies of the displacement and the mixed forms, Taylor-Hood and VMS |
| `boundary` | Dirichlet and Neumann boundary conditions on named faces |
| `material` | isochoric and volumetric hyperelastic models |
| `solver` | the bases of the nonlinear and time solvers of the cases, the generalized-alpha methods, and the time steps |
| `system` | tensors and printing and file tools |
| `visualization` | the VTU writers, in Cartesian and spherical components |

Each case has in its own `include/` the loading (`LoadData.hpp`), the material
(`MaterialModelData.hpp`), its nonlinear solvers and its time solver.

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
| hypre | 3.2.0, master `5b58261` | the parallel matrices of MFEM (`HypreParMatrix`), and the BoomerAMG preconditioners of the GMRES solvers of the linear systems |
| MUMPS | 5.8.1 | parallel direct solver of the linear systems, instead of GMRES with `linear_solver: mumps` in the `solver` section of the input of a case; MFEM has only its interface |
| yaml-cpp | 0.9.0 | reads the inputs of the cases, `config_*.yaml` |
| MFEM | 4.10.1, master `45799eb` | finite element spaces, meshes, assembly and the Krylov solvers |

```
MFEM/
├── lib/
│   ├── metis/  hypre/  mumps/  yaml-cpp/
│   └── mfem/           (built in mfem/build)
└── mfem-solids/
```

`cmake/mfem-solids.cmake` finds MFEM in `lib/mfem/build` and yaml-cpp in
`lib/yaml-cpp/lib/cmake/yaml-cpp`, and stops unless MFEM has MPI and MUMPS.

## Installing the dependencies

Run the commands below in order, all from `MFEM/`, the folder that will hold `lib/` and
`mfem-solids/`.

### 1. Homebrew packages

CMake, Open MPI, and the Fortran compiler, BLAS and ScaLAPACK that MUMPS needs:

```bash
brew install cmake open-mpi gcc openblas scalapack
```

### 2. METIS 5.1.0

Download it into `lib/metis`:

```bash
mkdir -p lib && (cd lib && curl -LO https://github.com/mfem/tpls/raw/gh-pages/metis-5.1.0.tar.gz && tar xzf metis-5.1.0.tar.gz && mv metis-5.1.0 metis && rm metis-5.1.0.tar.gz)
```

Build it as a static library, `lib/metis/lib/libmetis.a`:

```bash
(cd lib/metis && cmake -S . -B lib -DCMAKE_BUILD_TYPE=Release -DGKLIB_PATH=$PWD/GKlib -DSHARED=FALSE -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build lib && cp lib/libmetis/libmetis.a lib/)
```

### 3. hypre 3.2.0

Download it into `lib/hypre`, at the commit used here:

```bash
(cd lib && git clone https://github.com/hypre-space/hypre.git && git -C hypre checkout 5b58261)
```

Build it and install it into `lib/hypre/src/hypre`:

```bash
(cd lib/hypre/src && ./configure --disable-fortran CC=mpicc CXX=mpicxx && make -j8 && make install)
```

### 4. MUMPS 5.8.1

Download it into `lib/mumps-src`:

```bash
mkdir -p lib/mumps-src && (cd lib/mumps-src && curl -LO https://mumps-solver.org/MUMPS_5.8.1.tar.gz && tar xzf MUMPS_5.8.1.tar.gz)
```

Write its build settings, `Makefile.inc`: METIS and PORD orderings, the MPI compilers,
and OpenBLAS and ScaLAPACK of Homebrew:

```bash
(cd lib/mumps-src/MUMPS_5.8.1 && cp Make.inc/Makefile.inc.generic Makefile.inc && cat >> Makefile.inc <<'EOF'
LMETISDIR  = $(topdir)/../../metis/lib
IMETIS     = -I$(topdir)/../../metis/include
LMETIS     = -L$(LMETISDIR) -lmetis
ORDERINGSF = -Dmetis -Dpord
CC      = mpicc
FC      = mpif90
FL      = mpif90
LAPACK  = -L/opt/homebrew/opt/openblas/lib -lopenblas
SCALAP  = -L/opt/homebrew/opt/scalapack/lib -lscalapack
INCPAR  =
LIBPAR  = $(SCALAP) $(LAPACK)
LIBBLAS = -L/opt/homebrew/opt/openblas/lib -lopenblas
OPTF    = -O2 -fPIC -fallow-argument-mismatch
OPTC    = -O2 -fPIC -I.
OPTL    = -O2
EOF
)
```

Build the single and double precision libraries, copy them and the headers to `lib/mumps`,
and remove the sources:

```bash
(cd lib/mumps-src/MUMPS_5.8.1 && make s d && mkdir -p ../../mumps && cp -R lib include ../../mumps/) && rm -rf lib/mumps-src
```

### 5. yaml-cpp 0.9.0

Download it into `lib/yaml-cpp/src`:

```bash
git clone --branch yaml-cpp-0.9.0 https://github.com/jbeder/yaml-cpp.git lib/yaml-cpp/src
```

Build it as a static library, install it into `lib/yaml-cpp`, and remove the build folder:

```bash
(cd lib/yaml-cpp && cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD -DYAML_BUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build build && cmake --install build) && rm -rf lib/yaml-cpp/build
```

### 6. MFEM 4.10.1

Download it into `lib/mfem`, at the commit used here:

```bash
(cd lib && git clone https://github.com/mfem/mfem.git && git -C mfem checkout 45799eb)
```

Configure it with MPI, METIS, hypre and MUMPS, and build it in `lib/mfem/build`:

```bash
mkdir -p lib/mfem/build && (cd lib/mfem/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DMFEM_USE_MPI=ON -DMFEM_USE_METIS_5=ON -DMETIS_DIR=$PWD/../../metis -DHYPRE_DIR=$PWD/../../hypre/src/hypre -DMFEM_USE_MUMPS=ON -DMUMPS_DIR=$PWD/../../mumps "-DMUMPS_REQUIRED_PACKAGES=MPI;MPI_Fortran;METIS" "-DMUMPS_REQUIRED_LIBRARIES=/opt/homebrew/opt/scalapack/lib/libscalapack.dylib;/opt/homebrew/opt/openblas/lib/libopenblas.dylib;/opt/homebrew/opt/gcc/lib/gcc/current/libgfortran.dylib" -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran && make -j8 mfem)
```

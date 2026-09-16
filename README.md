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

## Building

Requires CMake 3.20 or newer and an installed MFEM (developed against 4.9.1, serial,
without PETSc or SuiteSparse). The build looks for MFEM under `~/.local/mfem`; change
the `HINTS` in `CMakeLists.txt` if yours is elsewhere.

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

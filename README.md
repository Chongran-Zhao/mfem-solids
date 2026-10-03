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
| `vtu_writer` | `results_gf/` and the material | `results_vtu/`: the deformed mesh with displacement, pressure and element-center first and second Piola-Kirchhoff stresses; pressure is p(J) in the displacement form and the saved nodal field in the mixed form; in `dynamic_disp/` and `dynamic_mixed/` also velocity and acceleration, at the physical times of `time.csv`; open `results_vtu.pvd` in ParaView |
| `csv_writer` | `results_gf/` and the material | `results_csv/<face>.csv`: mean displacement, reaction force F, reference face area and face-mean pressure p on the faces and directions of `csv_writer` in `config.yaml`, at each step; the reaction is the formulation's residual on the constrained dofs of the face |

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
acceleration is zero, as in MixPERIGEE. `ctest` runs the checks of
`tests/dynamic_disp.cpp`. Its `csv_writer` writes the face columns of the static one at the
physical times, with reactions M a + F_int - F_ext that include the inertia, and
`energy.csv`, the kinetic, strain and total energies of the whole body.

`dynamic_mixed/` solves the mixed dynamics, M a + F_int(u,p) = F_ext(t) with J(u) = J(p)
weakly, by the same second-order generalized-alpha method: u_{n+1} and p_{n+1} are the
unknowns of Newton's method, both equations hold at the intermediate states, and the
pressure has no inertia. The initial pressure is zero, consistent with the zero initial
displacement. `ctest` runs the checks of `tests/dynamic_mixed.cpp`. Its `csv_writer`
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
for Newton's method, and the block tangent is copied into one sparse matrix for UMFPACK.

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

Requires CMake 3.20 or newer, MFEM built with CMake and SuiteSparse (developed against
4.10.1; the linear systems are solved with UMFPACK) and yaml-cpp. The build looks for
them in `../../lib`; change the paths in `cmake/mfem-solids.cmake` if yours are elsewhere.

The programs of each formulation are in their own folder, `static_disp/`,
`static_mixed/`, `dynamic_disp/` and `dynamic_mixed/`, a CMake project of its own; the settings they share are in
`cmake/mfem-solids.cmake`. The displacement form is built and run in `static_disp/`:

```bash
cd static_disp && cmake -B build && cmake --build build
```

```bash
cd build && ./read_mesh && ./driver && ./vtu_writer && ./csv_writer
```

`static_mixed/` builds and runs the same four programs, using its own configuration
with `space.order >= 2`. Its driver saves displacement and pressure; its
postprocessors read both fields to compute stresses and reactions.

`dynamic_disp/` and `dynamic_mixed/` build `read_mesh`, `driver`, `vtu_writer`, `csv_writer` and the checks run by `ctest`:

```bash
cd build && ctest && ./read_mesh && ./driver && ./vtu_writer && ./csv_writer
```

Each folder has its own `config.yaml`, which CMake copies into its `build/`,
again whenever it changes; each program reads the `config.yaml` of the directory it runs in,
or the file given as its first argument. `mesh.file` is relative to the project directory,
whose `mesh_files/` all formulations share.

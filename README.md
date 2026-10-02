# mfem-solids

Finite-strain hyperelastostatics with [MFEM](https://mfem.org/): a Total Lagrangian
formulation with a hand-written material model and element integrator, Dirichlet
and traction conditions on named faces, load stepping and Newton's method.

## Programs

| Program | Reads | Writes |
|---|---|---|
| `read_mesh` | the mesh in `config.yaml` | `beam.mesh` with the six box faces named `left`, `right`, `front`, `back`, `bottom`, `top`, and a 3D view of them, `beam_boundary.html` |
| `driver` (`static_disp/`) | `beam.mesh`, the boundary conditions, loading and solver settings | the displacement of each load step, `results_gf/disp_XXXX.gf` |
| `driver` (`static_mixed/`) | the same, in the mixed displacement-pressure form with Taylor-Hood elements (`space.order` >= 2) | displacement `disp_XXXX.gf` and nodal pressure `pres_XXXX.gf` at each load step |
| `vtu_writer` | `results_gf/` and the material | `results_vtu/`: the deformed mesh with displacement, pressure and element-center first and second Piola-Kirchhoff stresses; pressure is p(J) in the displacement form and the saved nodal field in the mixed form; open `results_vtu.pvd` in ParaView |
| `csv_writer` | `results_gf/` and the material | `results_csv/<face>.csv`: mean displacement, reaction force F and F / A_0 on the faces and directions of `csv_writer` in `config.yaml`, at each step; the reaction is the formulation's residual on the constrained dofs of the face |

`scripts/plot_csv.m` (MATLAB) plots the CSV files of `csv_writer` against the load factor,
overlaying the result folders listed at its top.

Each formulation has its own `config.yaml`; the material is in `include/material/MaterialModelData.hpp`,
and the prescribed displacements and tractions are in `include/boundary/LoadData.hpp`.

Both drivers follow the same structure: global assembly, a nonlinear solver for
one load step, and a time solver for load stepping and output. The mixed structure
and ownership are described in `static_mixed/README.md`.

## Setting up a problem

- **Faces.** `read_mesh` names the six faces of the box; the boundary conditions refer
  to them by name.
- **Loading.** The load is raised in `loading.load_steps` equal steps of the pseudo time
  t = n / N. It is either a prescribed displacement or a traction, not both:
  - displacement: list the driven faces and directions in `Dirichlet.disp_bc`, leave
    `Neumann.faces` empty, and give the displacement in `LoadData::disp_driven`;
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

The programs of each formulation are in their own folder, `static_disp/` and
`static_mixed/`, a CMake project of its own; the settings they share are in
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

Each folder has its own `config.yaml`, which CMake copies into its `build/`,
again whenever it changes; each program reads the `config.yaml` of the directory it runs in,
or the file given as its first argument. `mesh.file` is relative to the project directory,
whose `mesh_files/` both formulations share.

# mfem-solids

Finite-strain hyperelastostatics with [MFEM](https://mfem.org/): a Total Lagrangian
formulation with a hand-written material model and element integrator, Dirichlet
and traction conditions on named faces, load stepping and Newton's method.

## Programs

| Program | Reads | Writes |
|---|---|---|
| `read_mesh` | the mesh in `config.yaml` | `beam.mesh` with the six box faces named `left`, `right`, `front`, `back`, `bottom`, `top`, and a 3D view of them, `beam_boundary.html` |
| `driver_displacement` | `beam.mesh`, the boundary conditions, loading and solver settings | the displacement and the nodal internal force of each load step, `results_gf/disp_XXXX.gf` and `internal_force_XXXX.gf` |
| `vtu_writer` | `results_gf/` | `results_vtu/`: the deformed mesh with the displacement and the first and second Piola-Kirchhoff stresses; open `results_vtu.pvd` in ParaView |
| `csv_writer` | `results_gf/` | `results_csv/<face>.csv`: mean displacement, resultant force and mean traction on the faces and directions of `csv_writer` in `config.yaml`, at each step |

All settings are in `config.yaml`; the material is in `include/material/MaterialModelData.hpp`,
and the prescribed displacements and tractions are in `include/boundary/LoadData.hpp`.

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

- The resultant force F of `csv_writer` is the sum of the nodal internal forces on the
  face. It is the reaction on a constrained face and the applied load on a traction
  face, exact up to the Newton tolerance on any mesh. On a free face it picks up the
  reactions of the edges it shares with constrained faces, so it has no meaning there.
- F is a resultant only: applying F / A_0 as a uniform traction does not reproduce a
  prescribed displacement, since the distribution of the reaction is lost.
- `vtu_writer` writes the vertices only, so it needs linear elements (`space.order: 1`).
- Linear hexahedra lock in bending: on a coarse mesh the stresses are poor. Quadratic
  elements (`space.order: 2`) remove most of this.
- Large prescribed displacements on a refined mesh may need more load steps for Newton
  to converge.

## Building and running

Requires CMake 3.20 or newer, MFEM built with CMake and SuiteSparse (developed against
4.10.1; the linear systems are solved with UMFPACK) and yaml-cpp. The build looks for
them in `../../lib`; change the paths in `CMakeLists.txt` if yours are elsewhere.

```bash
cmake -B build
```

```bash
cmake --build build
```

The programs run in `build/`, in this order:

```bash
cd build && ./read_mesh && ./driver_displacement && ./vtu_writer && ./csv_writer
```

Each program reads `config.yaml` from the source directory, or the file given as its
first argument.

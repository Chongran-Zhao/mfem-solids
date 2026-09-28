# mfem-solids

Finite-strain hyperelastostatics with [MFEM](https://mfem.org/): a Total Lagrangian
formulation with a hand-written material model and element integrator, Dirichlet
and traction conditions on named faces, load stepping and Newton's method.

## Programs

| Program | Reads | Writes |
|---|---|---|
| `read_mesh` | the mesh in `config.yaml` | `beam.mesh` with the six box faces named `left`, `right`, `front`, `back`, `bottom`, `top`, and a 3D view of them, `beam_boundary.html` |
| `driver` | `beam.mesh`, the boundary conditions, loading and solver settings | the displacement of each load step, `results_gf/disp_XXXX.gf` |
| `vtu_writer` | `results_gf/` | `results_vtu/`: the deformed mesh with the displacement and the first and second Piola-Kirchhoff stresses |
| `csv_writer` | `results_gf/` | `results_csv/<face>.csv`: mean displacement, resultant force and mean traction on the faces and directions of `csv_writer` in `config.yaml`, at each step |

All settings are in `config.yaml`; the material is in `include/material/MaterialModel.hpp`,
and the prescribed displacements and tractions are in `include/boundary/LoadData.hpp`.

## Building and running

Requires CMake 3.20 or newer, MFEM built with CMake (developed against 4.10.1) and
yaml-cpp. The build looks for them in `../../lib`; change the paths in `CMakeLists.txt`
if yours are elsewhere.

```bash
cmake -B build
```

```bash
cmake --build build
```

The programs run in `build/`, in this order:

```bash
cd build && ./read_mesh && ./driver && ./vtu_writer && ./csv_writer
```

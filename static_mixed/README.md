# Mixed formulation

Structure, following `static_disp`:

- `LocalAssembly_Mixed`: own the material and compute the element residual
  and tangent.
- `GlobalAssembly_Mixed`: own local assembly and boundary conditions, assemble
  the full residual (internal minus external force) and tangent, and eliminate
  essential displacement dofs from the residual or a sparse tangent copy.
- `NonlinearSolver_Static_Mixed`: own UMFPACK, Newton and its block monitor;
  solve one load step with a consistent displacement-pressure predictor.
- `TimeSolver_Static_Mixed`: manage load stepping and save displacement and
  pressure.
- `static_mixed/driver.cpp`: read configuration, construct the mesh and finite
  element spaces, connect the components and start the solve.

Both solvers use a residual equation with a zero Newton right-hand side. The
predictor uses the same full residual and tangent; boundary elimination moves
prescribed increments to the right-hand side. The mixed tangent is assembled
in blocks, converted to one sparse matrix and solved by UMFPACK, just as the
displacement tangent is. Only one block nonlinear form is needed.

The driver owns the finite element spaces and separate displacement and pressure
grid functions. The public interfaces are `solve(t, disp, pres)` and
`run(disp, pres)`. The nonlinear solver packs the fields into its internal block
vector for Newton and copies the converged fields back after each step. The block
form owns its local integrator, which owns the material. MFEM's
`BlockNonlinearForm` does not offer the external-integrator ownership option used by `NonlinearForm` in
the displacement assembly. The nonlinear solver owns global assembly, and the
time solver owns the nonlinear solver. These objects are destroyed before the
spaces and grid functions.

The driver saves only `disp_XXXX.gf` and `pres_XXXX.gf`, including the initial
state. Stress is computed by `vtu_writer` from the saved displacement and nodal
pressure; it is no longer saved as `stress_XXXX.gf` by the solver. `csv_writer`
uses the full mixed residual to report support reactions. Both postprocessors
use the same material as the driver.

Build and run the same four programs as `static_disp`:

```bash
cmake -B build && cmake --build build
cd build && ./read_mesh && ./driver && ./vtu_writer && ./csv_writer
```

`read_mesh` reuses the displacement formulation's mesh preparation source and
HTML template. The mixed equations, Taylor-Hood spaces and loading behavior are
unchanged. General configuration instructions are in the project README.

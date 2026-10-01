# Mixed formulation

Structure, following `static_disp`:

- `LocalAssembly_Mixed`: own the material and compute the element residual,
  tangent and element-center stress.
- `GlobalAssembly_Mixed`: own local assembly and boundary conditions, assemble
  the full residual (internal minus external force) and tangent, and eliminate
  essential displacement dofs from the residual or a sparse tangent copy.
- `NonlinearSolver_Static_Mixed`: own UMFPACK, Newton and its block monitor;
  solve one load step with a consistent displacement-pressure predictor.
- `TimeSolver_Static_Mixed`: manage load stepping and save displacement, pressure
  and element-center stress, delegating stress calculation to assembly.
- `static_mixed/driver.cpp`: read configuration, construct the mesh and finite
  element spaces, connect the components and start the solve.

Both solvers use a residual equation with a zero Newton right-hand side. The
predictor uses the same full residual and tangent; boundary elimination moves
prescribed increments to the right-hand side. The mixed tangent is assembled
in blocks, converted to one sparse matrix and solved by UMFPACK, just as the
displacement tangent is. Only one block nonlinear form is needed.

The driver owns the finite element spaces and the block solution. `solve(t, sol)`
and `run(sol)` create displacement and pressure views internally. The block form
owns its local integrator, which owns the material; global assembly borrows a
pointer to that integrator for stress output. MFEM's `BlockNonlinearForm` does
not offer the external-integrator ownership option used by `NonlinearForm` in
the displacement assembly. The nonlinear solver owns global assembly, and the
time solver owns the nonlinear solver. These objects are destroyed before the
spaces and solution.

The equations, Taylor-Hood spaces, loading behavior and result file formats are
unchanged. Saved fields include the initial state. The build and mesh preparation
instructions are in the project README.

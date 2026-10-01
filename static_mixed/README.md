# Mixed formulation refactor

Planned structure, following `static_disp`:

- `LocalAssembly_Mixed`: retain the element residual and tangent integrator.
- `GlobalAssembly_Mixed`: own the block nonlinear forms and boundary conditions,
  assemble internal and external forces, and expose the residual and tangent.
- `NonlinearSolver_Static_Mixed`: own the block linear solver, Newton solver and
  monitor, and solve one load step using the consistent displacement-pressure
  predictor.
- `TimeSolver_Static_Mixed`: manage load stepping and save displacement, pressure
  and element-center stress at each step.
- `static_mixed/driver.cpp`: read configuration, construct the mesh and finite
  element spaces, connect the components and start the solve.

Preserve the existing equations, Taylor-Hood spaces, loading behavior and result
file formats. Validate the refactor by building both formulations and comparing
the mixed driver's results before and after the change for the default case.

This initial document opens the refactor; implementation is pending.

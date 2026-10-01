# Mixed formulation

Structure, following `static_disp`:

- `LocalAssembly_Mixed`: compute the element residual and tangent.
- `GlobalAssembly_Mixed`: own the material, block nonlinear forms and boundary conditions,
  assemble internal and external forces, and expose the residual and tangent.
- `NonlinearSolver_Static_Mixed`: own the block linear solver, Newton solver and
  monitor, and solve one load step using the consistent displacement-pressure
  predictor.
- `TimeSolver_Static_Mixed`: manage load stepping and save displacement, pressure
  and element-center stress at each step.
- `static_mixed/driver.cpp`: read configuration, construct the mesh and finite
  element spaces, connect the components and start the solve.

The driver owns the finite element spaces and the block solution. The displacement
and pressure grid functions are views into that solution. Global assembly owns
the material; the two forms own their local integrators, which borrow it. The
nonlinear solver owns global assembly, and the time solver owns the nonlinear
solver. These objects are destroyed before the spaces and solution.

The equations, Taylor-Hood spaces, loading behavior and result file formats are
unchanged. The time solver saves displacement, pressure and element-center first
Piola-Kirchhoff stress, including the initial state.

The build and mesh preparation instructions are in the project README.

# Displacement dynamics

Finite-strain Total Lagrangian dynamics using the existing displacement material
and element assembly. The default case is a cantilever beam with an initial
transverse velocity and no external load.

```bash
cmake -S dynamic_disp -B dynamic_disp/build
cmake --build dynamic_disp/build
ctest --test-dir dynamic_disp/build --output-on-failure
cd dynamic_disp/build
./read_mesh
./driver
```

`driver` accepts an optional YAML path. Run it in the directory containing the
mesh named by `mesh.output`. Density is reference density; `dt` and `final_time`
are physical time in seconds. Initial displacement and velocity have three
components multiplied by the `uniform` or `linear_x` profile. Material constants
remain in `MaterialModelData.hpp`.

## Ownership and time integration

- The driver owns displacement, velocity and acceleration grid functions.
- `TimeSolver_Dynamic_Disp` owns the nonlinear solver and integration parameters.
  It computes stage/end times, integration weights and known predictor vectors,
  requests a solve for that step, and updates velocity and acceleration through
  Newmark kinematics. It manages the physical-time loop and output, and never
  accesses global assembly or the mass matrix.
- `NonlinearSolver_Dynamic_Disp` owns global assembly, the consistent mass matrix,
  the linear/Newton solvers and monitor. Like the static nonlinear solver, it
  handles loading, prescribed boundaries, a consistent initial guess, residual
  and tangent assembly, and Newton convergence. Its `solve` receives stage/end
  times, scalar weights and known vectors; it constructs its own MFEM equation.
  It computes compatible initial acceleration and kinetic energy internally.
  It neither includes nor receives `TimeMethod_GenAlpha` and does not expose
  assembly or mass to its caller.
- `GlobalAssembly_Disp` assembles mass, internal force and material tangent;
  element and material assembly have no time-integration dependency.

`TimeMethod_GenAlpha` uses the new-time weight convention of PERIGEE's
[second-order generalized-alpha parameters](https://github.com/APSIS-ANALYSIS/PERIGEE/blob/329fe3433b6edc3363f107b7096efdf2fc77c4a7/src/Solver/TimeMethod_GenAlpha.cpp):

```
alpha_m = (2 - rho_inf) / (1 + rho_inf)
alpha_f = 1 / (1 + rho_inf)
gamma   = 1/2 + alpha_m - alpha_f
beta    = (1 + alpha_m - alpha_f)^2 / 4
x_alpha = (1 - alpha) x_n + alpha x_(n+1)
```

The step residual is `M a_alpha_m + R(u_alpha_f, t_alpha_f)`. Its displacement
Jacobian is `alpha_m/(beta dt^2) M + alpha_f K(u_alpha_f)`. Mass is assembled once
without boundary elimination. Prescribed increments are eliminated from the
combined tangent so their inertia coupling reaches the free equations.
The time solver updates velocity and acceleration after Newton convergence.
The nonlinear solver's local step operator borrows the known vectors only
during the single-step call. Neither solver stores displacement, velocity or acceleration as members.

`rho_inf` lies in [0, 1]; 1 gives the nondissipative linear midpoint method.
Smaller values introduce damping of high frequency response. Exact nonlinear
energy conservation is not guaranteed.

## Boundaries and output

Both prescribed motion and nominal traction can be active. LoadData receives
physical time here, whereas the static drivers pass a load factor. For dynamic
initialization, keep `velo_driven` and `acce_driven` consistent with the time
derivatives of `disp_driven`. The initial acceleration solves equilibrium with
prescribed acceleration. Later boundary velocities and accelerations follow the
discrete Newmark kinematics of the imposed displacement.

`output.gf` is emptied when the time solver is constructed, following the static
workflow. It contains `disp_XXXX.gf`, `velo_XXXX.gf`, `acce_XXXX.gf` and `time.csv`.
The CSV records physical time, actual step size, Newton iterations and kinetic
energy. The initial state is step zero.

Currently the driver supports conforming 3D meshes. Dynamic stress/reaction
postprocessors, strain energy output and mixed dynamics remain subsequent work;
static reaction CSVs omit inertia and must not be used as dynamic reactions.

## Verification

CTest checks the full and constrained effective tangent by finite differences
at three values of `rho_inf`, rigid translation, prescribed linear motion,
traction impulse versus total momentum, final-step shortening and all three
saved fields. A small-amplitude longitudinal mode is compared against its
linearized frequency and discrete midpoint solution, with energy behavior and
second-order time convergence checks. Its energy check uses linearized strain
energy and does not assert exact nonlinear energy conservation.

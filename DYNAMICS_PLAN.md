# Displacement dynamics

Start from the current `main` after the displacement and mixed static solver
refactors. This work replaces the old dynamics attempt in PR #2; its operator
and obsolete driver and boundary interfaces are not carried into this branch.

## Goal

Add finite-strain displacement dynamics with a reference-configuration mass
matrix and implicit generalized-alpha time integration. Reuse the current
material, local assembly and boundary components with explicit ownership.
Mixed displacement-pressure dynamics is a later task.

## Implementation plan

- [x] Rename the displacement and mixed residual/tangent interfaces to
      `assemble_residual` and `assemble_tangent`, and add `assemble_mass` to
      displacement global assembly. The caller owns the unconstrained mass
      matrix on true dofs.
- [x] Define density, time step, final time, integration parameters and initial
      displacement and velocity in a dedicated `dynamic_disp` configuration.
- [x] Assemble the consistent mass matrix and obtain a compatible initial
      acceleration from the initial equilibrium and boundary conditions.
- [x] Add dynamic nonlinear and time solver components using existing global assembly,
      following the current static ownership chain. Keep material integration
      independent of the time-integration scheme.
- [x] Implement the generalized-alpha residual and effective tangent, Newmark
      state updates and prescribed motion at physical time.
- [ ] Add a `dynamic_disp` driver and postprocessors for displacement, velocity,
      acceleration, stresses, reactions and physical time.
- [ ] Add kinetic and strain energy reporting for numerical verification.

## Verification

- Check the effective tangent against finite differences of the dynamic
  residual, including prescribed displacement handling.
- Verify a small-amplitude free-vibration case against its linearized
  frequency, and check time-step convergence and energy behavior.
- Verify rigid motion and prescribed motion with consistent initial data.
- Confirm the existing static programs retain their behavior when shared
  components change.

## Current status

The single-step nonlinear solver, time loop, generalized-alpha parameters,
compatible initial acceleration, physical-time boundaries and `dynamic_disp`
driver/configuration are implemented. The driver saves displacement, velocity,
acceleration and a time history with kinetic energy. Dynamic stress/reaction
postprocessors and nonlinear strain energy output remain to be implemented.

CTest passes effective tangent finite differences (full and constrained;
`rho_inf` = 0, 0.5, 1), rigid translation, prescribed motion, traction momentum,
physical output times and final-step shortening. Small-amplitude longitudinal
vibration agrees with its linearized frequency and discrete midpoint solution,
retains linearized total energy and shows second-order time convergence. The
suite also passes AddressSanitizer with leak detection disabled. The default
80-element beam completes 50 steps to t = 0.05. Both static formulations build.
The mixed displacement-loading and traction-loading regressions reproduce all
104 saved fields byte for byte after the shared boundary changes.

Mass assembly was checked at Q1/Q2 with both vector orderings on conforming and
nonconforming meshes: total mass, density scaling, symmetry, positive quadratic
forms and ownership after assembly destruction. The dynamic driver currently
requires conforming meshes.

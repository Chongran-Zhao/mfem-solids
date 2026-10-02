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
- [ ] Define density, time step, final time, integration parameters and initial
      displacement and velocity in a dedicated `dynamic_disp` configuration.
- [ ] Assemble the consistent mass matrix and obtain a compatible initial
      acceleration from the initial equilibrium and boundary conditions.
- [ ] Add dynamic global assembly, nonlinear and time solver components,
      following the current static ownership chain. Keep material integration
      independent of the time-integration scheme.
- [ ] Implement the generalized-alpha residual and effective tangent, Newmark
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

Assembly preparation is implemented. Both static formulations build, and mass
assembly has been checked at Q1/Q2 with both vector orderings on conforming and
nonconforming meshes: total mass, density scaling, symmetry, positive quadratic
forms and ownership after assembly destruction. No dynamic solver or dynamic
runtime configuration has been implemented yet.

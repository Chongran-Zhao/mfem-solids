// ============================================================================
// NonlinearSolver_Static.hpp
//
// Solves one step of the static displacement form, R(d) = 0 at the time of
// the step, without inertia: sets the load of the step, makes the consistent
// predictor the initial guess, and runs Newton's method from it. It owns the
// global assembly and the solvers; the loop over the steps is left to its
// caller. It is also the
// mfem::Operator that its NewtonSolver solves, through Mult and GetGradient.
// The displacement is a ParGridFunction, on which the boundary values are
// set; Newton's method works on the vector of the dofs this rank owns, with
// the norms over all ranks.
//
// The linear systems are solved by GMRES, preconditioned by BoomerAMG with
// the systems options of 3 components per node, or by MUMPS, by
// linear_solver of the solver section of config.yaml.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_STATIC_HPP
#define NONLINEAR_SOLVER_STATIC_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "LoadData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver.hpp"
#include "SystemTools.hpp"

class NonlinearSolver_Static : public NonlinearSolver
{
public:
   // Takes the ownership of the global assembly; the Newton settings come
   // from the solver section of config.yaml.
   NonlinearSolver_Static(std::unique_ptr<GlobalAssembly_Disp> input_global_assembly,
                          const YAML::Node &solver)
      : NonlinearSolver(input_global_assembly->get_num_dofs(), input_global_assembly->get_comm()),
        global_assembly(std::move(input_global_assembly)),
        linear_solver(global_assembly->get_comm())
   {
      amg.SetSystemsOptions(3, false);
      amg.SetPrintLevel(0);
      set_linear_solver(linear_solver, amg);
      set_newton_solver(solver, linear_solver, newton_monitor);
   }

   // The initial state is that of the driver, zero displacement.
   void initialize(double, mfem::ParGridFunction &, mfem::ParGridFunction &,
                   mfem::ParGridFunction &) override
   {
   }

   // Solves the load at time from the converged disp of time_n, and returns
   // the number of Newton iterations; velo and acce are not used.
   int solve(double, double time, mfem::ParGridFunction &disp, mfem::ParGridFunction &,
             mfem::ParGridFunction &) override
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();

      if (neumann.is_traction_load())
         global_assembly->set_traction_load(time);

      // The unknowns of Newton's method, the values of disp on the dofs this
      // rank owns.
      mfem::Vector disp_owned(global_assembly->get_num_dofs());
      disp.GetTrueDofs(disp_owned);

      initial_guess(time, disp, disp_owned);

      // The load value: the prescribed displacement, or the traction faces.
      if (dirichlet.is_disp_load())
         dirichlet.print_disp_load_by_step(disp);
      else
         neumann.print_traction_load_by_step();
      SystemTools::print_newton_header();

      // Newton iterations for R(d) = 0.
      const int iterations = solve_newton(time, disp_owned);
      disp.SetFromTrueDofs(disp_owned);
      return iterations;
   }

   bool is_dynamic() const override { return false; }

   // Required by MFEM: overrides mfem::Operator::Mult, which NewtonSolver
   // calls at every iteration for the residual.
   // R(d), zero on the constrained dofs.
   void Mult(const mfem::Vector &disp, mfem::Vector &residual) const override
   {
      global_assembly->assemble_residual(disp, residual);
      global_assembly->set_essential_bdr(residual);
   }

   // Required by MFEM: overrides mfem::Operator::GetGradient, which
   // NewtonSolver calls at every iteration for the tangent.
   // K(d), the identity on the constrained dofs; also the operator of the
   // AMG preconditioner, unless MUMPS solves it.
   mfem::Operator &GetGradient(const mfem::Vector &disp) const override
   {
      mfem::HypreParMatrix &tangent = global_assembly->assemble_tangent(disp);
      global_assembly->set_essential_bdr(tangent);
      if (!is_direct())
         amg.SetOperator(tangent);
      return tangent;
   }

private:
   // Consistent predictor: the initial guess of Newton's method is one
   // linear step from the converged state d with the new load, so that the
   // interior follows the prescribed boundary increment g instead of only
   // the boundary nodes moving,
   //    K_ff du_f = -R_f(d) - K_fe g,
   // with K_fe g moved to the right-hand side by set_essential_bdr. disp
   // and disp_owned, its values on the dofs this rank owns, are both
   // updated.
   void initial_guess(double tt, mfem::ParGridFunction &disp, mfem::Vector &disp_owned)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();

      // Zero on the fixed faces, the prescribed values at time tt on the
      // displacement-driven ones; g is their difference from disp.
      mfem::ParGridFunction disp_target(disp.ParFESpace());
      disp_target = disp;
      dirichlet.apply_fixed_bc(disp_target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_load_bc(tt, LoadData::disp_loading, disp_target);
      mfem::Vector target_owned(disp_owned.Size());
      disp_target.GetTrueDofs(target_owned);
      mfem::Vector prescribed_increment(target_owned);
      prescribed_increment -= disp_owned;

      mfem::Vector rhs(global_assembly->get_num_dofs());
      global_assembly->assemble_residual(disp_owned, rhs);
      rhs.Neg();
      mfem::HypreParMatrix &tangent = global_assembly->assemble_tangent(disp_owned);
      global_assembly->set_essential_bdr(tangent, prescribed_increment, rhs);

      if (!is_direct())
         amg.SetOperator(tangent);
      mfem::Vector predicted_increment(global_assembly->get_num_dofs());
      predicted_increment = 0.0;
      get_linear_solver().SetOperator(tangent);
      get_linear_solver().Mult(rhs, predicted_increment);
      disp_owned += predicted_increment;

      // Set the prescribed values exactly, free of round-off.
      for (int dof : dirichlet.get_ess_tdof_list())
         disp_owned(dof) = target_owned(dof);
      disp.SetFromTrueDofs(disp_owned);
   }

   // The tangent is owned by global_assembly.
   const std::unique_ptr<GlobalAssembly_Disp> global_assembly;   // R and K
   mutable mfem::HypreBoomerAMG amg;                             // preconditioner of GMRES
   mfem::GMRESSolver linear_solver;                              // solver of the tangent
   SystemTools::NewtonMonitor newton_monitor;                    // prints the residual norms
};

#endif

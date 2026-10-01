// ============================================================================
// NonlinearSolver_Static_Disp.hpp
//
// Solves one load step of the static displacement form, R(d) = 0: sets the
// load of the step, makes the consistent predictor the initial guess, and
// runs Newton's method from it. It owns the global assembly and the solvers;
// the loop over the load steps is left to its caller. It is also the
// mfem::Operator that its NewtonSolver solves, through Mult and GetGradient.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_STATIC_DISP_HPP
#define NONLINEAR_SOLVER_STATIC_DISP_HPP

#include <memory>
#include <utility>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"

class NonlinearSolver_Static_Disp : public mfem::Operator
{
public:
   // Takes the ownership of the global assembly; the Newton settings come
   // from the solver section of config.yaml.
   NonlinearSolver_Static_Disp(std::unique_ptr<GlobalAssembly_Disp> input_global_assembly,
                               const YAML::Node &solver)
      : mfem::Operator(input_global_assembly->get_num_dofs()),
        global_assembly(std::move(input_global_assembly))
   {
      newton_solver.SetOperator(*this);
      newton_solver.SetSolver(linear_solver);
      newton_solver.SetRelTol(solver["newton_rel_tol"].as<double>());
      newton_solver.SetAbsTol(solver["newton_abs_tol"].as<double>());
      newton_solver.SetMaxIter(solver["newton_max_iter"].as<int>());
      newton_solver.SetPrintLevel(-1);
      newton_solver.iterative_mode = true;
      newton_solver.SetMonitor(newton_monitor);
   }

   // Solves the load at time tt from the converged disp of the previous
   // step, and returns the number of Newton iterations.
   int solve(double tt, mfem::GridFunction &disp)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();

      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);

      initial_guess(tt, disp);

      // The load value: the prescribed displacement, or the traction faces.
      if (dirichlet.is_disp_load())
         dirichlet.print_disp_load_by_step(disp);
      else
         neumann.print_traction_load_by_step();
      SystemTools::print_newton_header();

      // Newton iterations for R(d) = 0; the empty right-hand side means zero.
      newton_solver.Mult(mfem::Vector(), disp);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at t = " << tt << ".");
      return newton_solver.GetNumIterations();
   }

   // Required by MFEM: overrides mfem::Operator::Mult, which NewtonSolver
   // calls at every iteration for the residual.
   // R(d), zero on the constrained dofs.
   void Mult(const mfem::Vector &disp, mfem::Vector &residual) const override
   {
      global_assembly->get_residual(disp, residual);
      global_assembly->set_essential_bdr(residual);
   }

   // Required by MFEM: overrides mfem::Operator::GetGradient, which
   // NewtonSolver calls at every iteration for the tangent.
   // K(d), the identity on the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &disp) const override
   {
      tangent = std::make_unique<mfem::SparseMatrix>(global_assembly->get_tangent(disp));
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

private:
   // Consistent predictor: the initial guess of Newton's method is one
   // linear step from the converged state d with the new load, so that the
   // interior follows the prescribed boundary increment g instead of only
   // the boundary nodes moving,
   //    K_ff du_f = -R_f(d) - K_fe g,
   // with K_fe g moved to the right-hand side by set_essential_bdr.
   void initial_guess(double tt, mfem::GridFunction &disp)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();

      // Zero on the fixed faces, the prescribed values at time tt on the
      // displacement-driven ones; g is their difference from disp.
      mfem::GridFunction disp_target(disp);
      dirichlet.apply_fixed_bc(disp_target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp_target);
      mfem::Vector prescribed_increment(disp_target);
      prescribed_increment -= disp;

      mfem::Vector rhs(global_assembly->get_num_dofs());
      global_assembly->get_residual(disp, rhs);
      rhs.Neg();
      tangent = std::make_unique<mfem::SparseMatrix>(global_assembly->get_tangent(disp));
      global_assembly->set_essential_bdr(*tangent, prescribed_increment, rhs);

      mfem::Vector predicted_increment(global_assembly->get_num_dofs());
      linear_solver.SetOperator(*tangent);
      linear_solver.Mult(rhs, predicted_increment);
      disp += predicted_increment;

      // Set the prescribed values exactly, free of round-off.
      dirichlet.apply_fixed_bc(disp);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp);
   }

   // newton_solver points to this operator, the linear solver and the
   // monitor, so it is declared last and goes first.
   const std::unique_ptr<GlobalAssembly_Disp> global_assembly;   // R and K
   mfem::UMFPackSolver linear_solver;                            // direct solver of the tangent
   SystemTools::NewtonMonitor newton_monitor;                    // prints the residual norms
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;          // tangent of the predictor and of Newton's method
   mfem::NewtonSolver newton_solver;
};

#endif

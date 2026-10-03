// ============================================================================
// NonlinearSolver_Static_Mixed.hpp
//
// Solves one load step of the static mixed form, R(u,p) = 0: sets the load of
// the step, makes the consistent predictor the initial guess, and runs
// Newton's method from it. It owns the global assembly and the solvers; the
// loop over the load steps is left to its caller. It is also the
// mfem::Operator that its NewtonSolver solves, through Mult and GetGradient.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_STATIC_MIXED_HPP
#define NONLINEAR_SOLVER_STATIC_MIXED_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Mixed.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"

class NonlinearSolver_Static_Mixed : public mfem::Operator
{
public:
   // Takes the ownership of the global assembly; the Newton settings come
   // from the solver section of config.yaml.
   NonlinearSolver_Static_Mixed(std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
                                const YAML::Node &solver)
      : mfem::Operator(input_global_assembly->get_num_dofs()),
        global_assembly(std::move(input_global_assembly)),
        newton_monitor(global_assembly->get_offsets())
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

   // Solves the load at time tt from the converged disp and pres of the
   // previous step, and returns the number of Newton iterations.
   int solve(double tt, mfem::GridFunction &disp, mfem::GridFunction &pres)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();

      // Newton's method works on one vector of both fields, sol = [u; p];
      // disp_view is its displacement block, for the boundary values.
      mfem::BlockVector sol(global_assembly->get_offsets());
      MFEM_VERIFY(disp.Size() == sol.GetBlock(0).Size() &&
                  pres.Size() == sol.GetBlock(1).Size(),
                  "The displacement or pressure size differs from the solver space.");
      sol.GetBlock(0) = disp;
      sol.GetBlock(1) = pres;
      mfem::GridFunction disp_view;
      disp_view.MakeRef(disp.FESpace(), sol.GetBlock(0), 0);

      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);

      initial_guess(tt, sol, disp_view);

      // The load value: the prescribed displacement, or the traction faces.
      if (dirichlet.is_disp_load())
         dirichlet.print_disp_load_by_step(disp_view);
      else
         neumann.print_traction_load_by_step();
      SystemTools::print_block_newton_header();

      // Newton iterations for R(u,p) = 0; the empty right-hand side means zero.
      newton_solver.Mult(mfem::Vector(), sol);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at t = " << tt << ".");

      disp = sol.GetBlock(0);
      pres = sol.GetBlock(1);
      return newton_solver.GetNumIterations();
   }

   // Required by MFEM: overrides mfem::Operator::Mult, which NewtonSolver
   // calls at every iteration for the residual.
   // R(u,p), zero on the constrained dofs.
   void Mult(const mfem::Vector &sol, mfem::Vector &residual) const override
   {
      global_assembly->assemble_residual(sol, residual);
      global_assembly->set_essential_bdr(residual);
   }

   // Required by MFEM: overrides mfem::Operator::GetGradient, which
   // NewtonSolver calls at every iteration for the tangent.
   // K(u,p), the identity on the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      tangent = global_assembly->assemble_tangent(sol);
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

private:
   // Consistent predictor: the initial guess of Newton's method is one
   // linear step from the converged state (u,p) with the new load, so that
   // the interior follows the prescribed boundary increment g instead of
   // only the boundary nodes moving,
   //    K_ff d(u,p)_f = -R_f(u,p) - K_fe g,
   // with K_fe g moved to the right-hand side by set_essential_bdr. disp is
   // the displacement block of sol.
   void initial_guess(double tt, mfem::BlockVector &sol, mfem::GridFunction &disp)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();

      // Zero on the fixed faces, the prescribed values at time tt on the
      // displacement-driven ones; g is their difference from disp, and zero
      // for the pressure.
      mfem::GridFunction disp_target(disp);
      dirichlet.apply_fixed_bc(disp_target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp_target);
      mfem::BlockVector prescribed_increment(global_assembly->get_offsets());
      prescribed_increment = 0.0;
      prescribed_increment.GetBlock(0) = disp_target;
      prescribed_increment.GetBlock(0) -= disp;

      mfem::Vector rhs(global_assembly->get_num_dofs());
      global_assembly->assemble_residual(sol, rhs);
      rhs.Neg();
      tangent = global_assembly->assemble_tangent(sol);
      global_assembly->set_essential_bdr(*tangent, prescribed_increment, rhs);

      mfem::Vector predicted_increment(global_assembly->get_num_dofs());
      linear_solver.SetOperator(*tangent);
      linear_solver.Mult(rhs, predicted_increment);
      sol += predicted_increment;

      // Set the prescribed values exactly, free of round-off.
      dirichlet.apply_fixed_bc(disp);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp);
   }

   // newton_solver points to this operator, the linear solver and the
   // monitor, so it is declared last and goes first.
   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly;   // R and K
   mfem::UMFPackSolver linear_solver;                             // direct solver of the tangent
   SystemTools::BlockNewtonMonitor newton_monitor;                // prints the residual norms of u and p
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;           // tangent of the predictor and of Newton's method
   mfem::NewtonSolver newton_solver;
};

#endif

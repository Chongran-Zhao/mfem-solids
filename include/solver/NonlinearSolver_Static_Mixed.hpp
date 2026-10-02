// ============================================================================
// NonlinearSolver_Static_Mixed.hpp
//
// Solves one mixed load step, R(u,p) = 0: sets the load, makes the consistent
// predictor the initial guess, then runs Newton. Owns global assembly and
// solvers, and implements Newton's operator through Mult and GetGradient.
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
   NonlinearSolver_Static_Mixed(std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
                                const YAML::Node &solver)
      : mfem::Operator(input_global_assembly->get_num_dofs()),
        global_assembly(std::move(input_global_assembly)),
        newton_monitor(global_assembly->get_offsets()),
        sol(global_assembly->get_offsets())
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

   // The caller supplies separate fields. Pack their preceding state into
   // the internal block vector, then copy the converged fields back.
   int solve(double tt, mfem::GridFunction &disp, mfem::GridFunction &pres)
   {
      MFEM_VERIFY(disp.Size() == sol.GetBlock(0).Size() &&
                  pres.Size() == sol.GetBlock(1).Size(),
                  "The displacement or pressure size differs from the solver space.");
      sol.GetBlock(0) = disp;
      sol.GetBlock(1) = pres;
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();
      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);
      initial_guess(tt, sol);

      mfem::GridFunction disp_view, pres_view;
      global_assembly->make_solution_views(sol, disp_view, pres_view);
      if (dirichlet.is_disp_load())
         dirichlet.print_disp_load_by_step(disp_view);
      else
         neumann.print_traction_load_by_step();
      SystemTools::print_block_newton_header();

      // As in the displacement solver, the empty right-hand side means zero.
      newton_solver.Mult(mfem::Vector(), sol);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at t = " << tt << ".");
      disp = sol.GetBlock(0);
      pres = sol.GetBlock(1);
      return newton_solver.GetNumIterations();
   }

   // R, zero on the constrained displacement dofs.
   void Mult(const mfem::Vector &sol, mfem::Vector &residual) const override
   {
      global_assembly->set_residual(sol, residual);
      global_assembly->set_essential_bdr(residual);
   }
   // Owned tangent copy, with identity at the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      tangent = global_assembly->get_tangent(sol);
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

   void set_center_stress(const mfem::GridFunction &disp,
                          const mfem::GridFunction &pres,
                          mfem::GridFunction &stress) const
   {
      global_assembly->set_center_stress(disp, pres, stress);
   }

private:
   // K_ff d(u,p)_f = -R_f(u,p) - K_fe g. Boundary elimination moves the
   // constrained columns to the right-hand side, as in the displacement solver.
   void initial_guess(double tt, mfem::BlockVector &sol)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      mfem::GridFunction disp, pres;
      global_assembly->make_solution_views(sol, disp, pres);
      mfem::GridFunction disp_target(disp);
      dirichlet.apply_fixed_bc(disp_target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp_target);
      mfem::BlockVector prescribed_increment(global_assembly->get_offsets());
      prescribed_increment = 0.0;
      prescribed_increment.GetBlock(0) = disp_target;
      prescribed_increment.GetBlock(0) -= disp;

      mfem::Vector rhs(global_assembly->get_num_dofs());
      global_assembly->set_residual(sol, rhs);
      rhs.Neg();
      tangent = global_assembly->get_tangent(sol);
      global_assembly->set_essential_bdr(*tangent, prescribed_increment, rhs);

      mfem::Vector predicted_increment(global_assembly->get_num_dofs());
      linear_solver.SetOperator(*tangent);
      linear_solver.Mult(rhs, predicted_increment);
      sol += predicted_increment;
      dirichlet.apply_fixed_bc(disp);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp);
   }

   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly;
   mfem::UMFPackSolver linear_solver;
   SystemTools::BlockNewtonMonitor newton_monitor;
   mfem::BlockVector sol;  // Internal Newton state; never moved or exposed to the caller.
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;
   // Destroy first: borrows this operator, the linear solver and monitor.
   mfem::NewtonSolver newton_solver;
};

#endif

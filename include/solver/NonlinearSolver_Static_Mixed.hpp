// ============================================================================
// NonlinearSolver_Static_Mixed.hpp
//
// Solves one mixed load step: sets the load, predicts displacement and
// pressure consistently, then runs Newton. Owns global assembly and solvers;
// implements the operator of the internal force, with the external force
// supplied as Newton's right-hand side.
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
#include "GlobalAssembly_Mixed.hpp"
#include "SystemTools.hpp"

class NonlinearSolver_Static_Mixed : public mfem::Operator
{
public:
   NonlinearSolver_Static_Mixed(std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
                                const YAML::Node &solver)
      : mfem::Operator(input_global_assembly->get_num_dofs()),
        global_assembly(std::move(input_global_assembly)),
        newton_monitor(global_assembly->get_offsets()),
        rhs(global_assembly->get_offsets()),
        disp_target(&global_assembly->get_disp_space()),
        prescribed_increment(global_assembly->get_offsets()),
        coupling(global_assembly->get_offsets()),
        predictor_rhs(global_assembly->get_offsets()),
        predicted_increment(global_assembly->get_offsets())
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

   const MaterialModel &get_material() const { return global_assembly->get_material(); }

   // sol holds both fields; disp is a view of its displacement block.
   int solve(double tt, mfem::BlockVector &sol, mfem::GridFunction &disp)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();
      dirichlet.apply_fixed_bc(disp);
      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);
      global_assembly->set_external_force(rhs, false);
      initial_guess(tt, sol, disp);

      if (dirichlet.is_disp_load())
         dirichlet.print_disp_load_by_step(disp);
      else
         neumann.print_traction_load_by_step();
      SystemTools::print_block_newton_header();

      global_assembly->set_external_force(rhs, true);
      newton_solver.Mult(rhs, sol);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at t = " << tt << ".");
      return newton_solver.GetNumIterations();
   }

   void Mult(const mfem::Vector &sol, mfem::Vector &residual) const override
   {
      global_assembly->set_residual(sol, residual);
   }
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      return global_assembly->get_tangent(sol);
   }

private:
   // K_ff d(u,p)_f = [f_ext - R(u,p)]_f - K_fe du_e.
   void initial_guess(double tt, mfem::BlockVector &sol, mfem::GridFunction &disp)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      disp_target = disp;
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp_target);
      prescribed_increment = 0.0;
      prescribed_increment.GetBlock(0) = disp_target;
      prescribed_increment.GetBlock(0) -= disp;

      global_assembly->set_internal_force(sol, predictor_rhs);
      global_assembly->get_internal_tangent(sol).Mult(prescribed_increment, coupling);
      predictor_rhs.Neg();
      predictor_rhs += rhs;
      predictor_rhs -= coupling;
      global_assembly->set_essential_bdr(prescribed_increment, predictor_rhs);

      linear_solver.SetOperator(global_assembly->get_tangent(sol));
      linear_solver.Mult(predictor_rhs, predicted_increment);
      sol += predicted_increment;
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp);
   }

   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly;
   SystemTools::BlockUMFPackSolver linear_solver;
   SystemTools::BlockNewtonMonitor newton_monitor;
   mfem::BlockVector rhs;
   mfem::GridFunction disp_target;
   mfem::BlockVector prescribed_increment, coupling, predictor_rhs, predicted_increment;
   // Destroy first: borrows this operator, the linear solver and monitor.
   mfem::NewtonSolver newton_solver;
};

#endif

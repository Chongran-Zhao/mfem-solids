// Linear and Newton solves for a supplied single-step equation.
#ifndef NONLINEAR_SOLVER_DYNAMIC_DISP_HPP
#define NONLINEAR_SOLVER_DYNAMIC_DISP_HPP

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "SystemTools.hpp"

class NonlinearSolver_Dynamic_Disp
{
public:
   explicit NonlinearSolver_Dynamic_Disp(const YAML::Node &solver)
   {
      newton_solver.SetSolver(linear_solver);
      newton_solver.SetRelTol(solver["newton_rel_tol"].as<double>());
      newton_solver.SetAbsTol(solver["newton_abs_tol"].as<double>());
      newton_solver.SetMaxIter(solver["newton_max_iter"].as<int>());
      newton_solver.SetPrintLevel(-1);
      newton_solver.iterative_mode = true;
      newton_solver.SetMonitor(newton_monitor);
   }

   // Used for the initial acceleration and the consistent initial guess.
   void solve_linear(const mfem::SparseMatrix &matrix, const mfem::Vector &rhs,
                     mfem::Vector &solution)
   {
      linear_solver.SetOperator(matrix);
      linear_solver.Mult(rhs, solution);
   }

   // The caller supplies the equation and owns its initial guess and result.
   int solve(const mfem::Operator &equation, mfem::Vector &solution)
   {
      newton_solver.SetOperator(equation);
      SystemTools::print_newton_header();
      newton_solver.Mult(mfem::Vector(), solution);
      return newton_solver.GetNumIterations();
   }

   bool get_converged() const { return newton_solver.GetConverged(); }

private:
   mfem::UMFPackSolver linear_solver;
   SystemTools::NewtonMonitor newton_monitor;
   mfem::NewtonSolver newton_solver;
};

#endif

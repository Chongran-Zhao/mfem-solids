// Owns global assembly, mass and numerical solvers; solves supplied step equations.
#ifndef NONLINEAR_SOLVER_DYNAMIC_DISP_HPP
#define NONLINEAR_SOLVER_DYNAMIC_DISP_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "GlobalAssembly_Disp.hpp"
#include "SystemTools.hpp"

class NonlinearSolver_Dynamic_Disp
{
public:
   NonlinearSolver_Dynamic_Disp(std::unique_ptr<GlobalAssembly_Disp> input_assembly,
                                double density, const YAML::Node &solver)
      : global_assembly(std::move(input_assembly)),
        mass(global_assembly->assemble_mass(density))
   {
      newton_solver.SetSolver(linear_solver);
      newton_solver.SetRelTol(solver["newton_rel_tol"].as<double>());
      newton_solver.SetAbsTol(solver["newton_abs_tol"].as<double>());
      newton_solver.SetMaxIter(solver["newton_max_iter"].as<int>());
      newton_solver.SetPrintLevel(-1);
      newton_solver.iterative_mode = true;
      newton_solver.SetMonitor(newton_monitor);
   }

   // Borrowed by the time solver when constructing a single-step equation.
   GlobalAssembly_Disp &get_global_assembly() const { return *global_assembly; }
   const mfem::SparseMatrix &get_mass() const { return *mass; }

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
   const std::unique_ptr<GlobalAssembly_Disp> global_assembly;
   const std::unique_ptr<mfem::SparseMatrix> mass;
   mfem::UMFPackSolver linear_solver;
   SystemTools::NewtonMonitor newton_monitor;
   mfem::NewtonSolver newton_solver;
};

#endif

// ============================================================================
// INonlinearSolver.hpp
//
// Interface of the nonlinear solvers of the cases, each solving one load or
// time step of its case, in the NonlinearSolver.hpp of the case folder. It
// is the mfem::Operator that its NewtonSolver solves: each case gives the
// residual and the tangent through Mult and GetGradient. It holds what they
// share: the NewtonSolver, its settings from the solver section of
// config.yaml, the MUMPS settings of the direct solvers, and the Newton
// iterations of a step. Newton's method works on the vector of the dofs this
// rank owns, with the norms over all ranks.
//
// Author: Chongran Zhao
// Date: Oct. 14, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef INONLINEAR_SOLVER_HPP
#define INONLINEAR_SOLVER_HPP

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

class INonlinearSolver : public mfem::Operator
{
protected:
   // An operator on num_dofs unknowns, those this rank owns, of the ranks of
   // comm.
   INonlinearSolver(int num_dofs, MPI_Comm comm)
      : mfem::Operator(num_dofs),
        newton_solver(comm)
   {
   }

   // MUMPS without output; the tangents keep the sparsity of the first one,
   // so it orders and analyzes it once and only factorizes the later ones.
   static void set_direct_solver(mfem::MUMPSSolver &linear_solver)
   {
      linear_solver.SetPrintLevel(0);
      linear_solver.SetReorderingReuse(true);
   }

   // Newton's method on this operator, with linear_solver for its tangents
   // and monitor for the residual norms; the tolerances and the maximum of
   // iterations come from the solver section of config.yaml. It starts from
   // the vector it is given. linear_solver and monitor must outlive the
   // Newton iterations; newton_solver only points to them.
   void set_newton_solver(const YAML::Node &solver, mfem::Solver &linear_solver,
                          mfem::IterativeSolverMonitor &monitor)
   {
      newton_solver.SetOperator(*this);
      newton_solver.SetSolver(linear_solver);
      newton_solver.SetRelTol(solver["newton_rel_tol"].as<double>());
      newton_solver.SetAbsTol(solver["newton_abs_tol"].as<double>());
      newton_solver.SetMaxIter(solver["newton_max_iter"].as<int>());
      newton_solver.SetPrintLevel(-1);
      newton_solver.iterative_mode = true;
      newton_solver.SetMonitor(monitor);
   }

   // Newton iterations from sol for zero residual, the empty right-hand side
   // meaning zero; aborts unless they converge, tt being the time of the
   // step. Returns the number of iterations.
   int solve_newton(double tt, mfem::Vector &sol)
   {
      newton_solver.Mult(mfem::Vector(), sol);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at t = " << tt << ".");
      return newton_solver.GetNumIterations();
   }

private:
   mfem::NewtonSolver newton_solver;
};

#endif

// ============================================================================
// NonlinearSolver_Fracture.hpp
//
// Solves the phase-field problem of one alternate-minimization iteration,
// R(alpha) = 0 with the displacement fixed, by Newton's method from the
// current phase field. The irreversibility of the notes
// Phase-field-fracture.pdf (p. 5, 7), after Bourdin, Francfort and Marigo
// (J. Elasticity 2008, eq. 8.13), fixes alpha = 0 on the nodes of the crack
// set K_n, those where alpha_n <= eta_lim at the start of the load step;
// set_crack_set builds it. It owns the global assembly and the solvers, and
// is the mfem::Operator that its NewtonSolver solves, through Mult and
// GetGradient. Newton's method works on the vector of the dofs this rank
// owns, with the norms over all ranks.
//
// The linear systems are solved by GMRES, preconditioned by BoomerAMG, or
// by MUMPS, by linear_solver of the solver_fracture section of config.yaml.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_FRACTURE_HPP
#define NONLINEAR_SOLVER_FRACTURE_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "GlobalAssembly_Fracture.hpp"
#include "INonlinearSolver.hpp"
#include "SystemTools.hpp"

class NonlinearSolver_Fracture : public INonlinearSolver
{
public:
   // Takes the ownership of the global assembly; the Newton settings come
   // from the solver_fracture section of config.yaml.
   NonlinearSolver_Fracture(std::unique_ptr<GlobalAssembly_Fracture> input_global_assembly,
                            const YAML::Node &solver)
      : INonlinearSolver(input_global_assembly->get_num_dofs(), input_global_assembly->get_comm()),
        global_assembly(std::move(input_global_assembly)),
        linear_solver(global_assembly->get_comm()),
        newton_monitor(newton_indent)
   {
      amg.SetPrintLevel(0);
      set_linear_solver(linear_solver, amg);
      set_newton_solver(solver, linear_solver, newton_monitor);
   }

   // The crack set K_n of the load step: the dofs this rank owns where
   // alpha <= eta_lim, set to alpha = 0 and constrained. Returns the number
   // of its dofs over all ranks.
   int set_crack_set(mfem::ParGridFunction &alpha, double eta_lim)
   {
      mfem::Vector alpha_owned(global_assembly->get_num_dofs());
      alpha.GetTrueDofs(alpha_owned);

      mfem::Array<int> crack_set;
      for (int dof = 0; dof < alpha_owned.Size(); dof++)
         if (alpha_owned(dof) <= eta_lim)
         {
            crack_set.Append(dof);
            alpha_owned(dof) = 0.0;
         }
      global_assembly->set_ess_tdof_list(crack_set);
      alpha.SetFromTrueDofs(alpha_owned);

      int size = crack_set.Size();
      MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_INT, MPI_SUM, global_assembly->get_comm());
      return size;
   }

   // Solves the phase field from alpha, and returns the number of Newton
   // iterations, whose residual norms it prints, indented by newton_indent;
   // time is that of the load step, for the messages.
   int solve(double time, mfem::ParGridFunction &alpha)
   {
      // The unknowns of Newton's method, the values of alpha on the dofs
      // this rank owns.
      mfem::Vector alpha_owned(global_assembly->get_num_dofs());
      alpha.GetTrueDofs(alpha_owned);

      SystemTools::print_newton_header(newton_indent);

      // Newton iterations for R(alpha) = 0.
      const int iterations = solve_newton(time, alpha_owned);
      alpha.SetFromTrueDofs(alpha_owned);
      return iterations;
   }

   // E(alpha), the degraded elastic energy plus the surface energy, over all
   // ranks.
   double get_energy(const mfem::ParGridFunction &alpha) const
   {
      mfem::Vector alpha_owned(global_assembly->get_num_dofs());
      alpha.GetTrueDofs(alpha_owned);
      return global_assembly->get_energy(alpha_owned);
   }

   // Required by MFEM: overrides mfem::Operator::Mult, which NewtonSolver
   // calls at every iteration for the residual.
   // R(alpha), zero on the constrained dofs.
   void Mult(const mfem::Vector &alpha, mfem::Vector &residual) const override
   {
      global_assembly->assemble_residual(alpha, residual);
      global_assembly->set_essential_bdr(residual);
   }

   // Required by MFEM: overrides mfem::Operator::GetGradient, which
   // NewtonSolver calls at every iteration for the tangent.
   // K(alpha), the identity on the constrained dofs; also the operator of
   // the AMG preconditioner, unless MUMPS solves it.
   mfem::Operator &GetGradient(const mfem::Vector &alpha) const override
   {
      mfem::HypreParMatrix &tangent = global_assembly->assemble_tangent(alpha);
      global_assembly->set_essential_bdr(tangent);
      if (!is_direct())
         amg.SetOperator(tangent);
      return tangent;
   }

private:
   // The tangent is owned by global_assembly.
   const std::unique_ptr<GlobalAssembly_Fracture> global_assembly;   // E, R and K
   mutable mfem::HypreBoomerAMG amg;                                 // preconditioner of GMRES
   mfem::GMRESSolver linear_solver;                                  // solver of the tangent
   SystemTools::NewtonMonitor newton_monitor;                        // prints the residual norms
   static constexpr int newton_indent = 5;                           // spaces before the Newton lines
};

#endif

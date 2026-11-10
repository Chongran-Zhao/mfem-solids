// ============================================================================
// NonlinearSolver_Fracture.hpp
//
// Solves the phase-field problem of one alternate-minimization iteration,
// R(alpha) = 0 with the displacement fixed, from the current phase field, by
// nonlinear_solver of the solver_fracture section of config.yaml:
//    newton   Newton's method of MFEM, for AT2, whose alpha stays in [0, 1]
//             without bounds;
//    vi       Newton's method with the bounds 0 <= alpha <= 1, SNESVINEWTONRSLS
//             of PETSc, as mef90 solves the phase field
//             (m_MEF90_DefMech.F90), needed by AT1, whose w'(1) < 0 would
//             push alpha above 1.
// The irreversibility of the notes Phase-field-fracture.pdf (p. 5, 7), after
// Bourdin, Francfort and Marigo (J. Elasticity 2008, eq. 8.13), fixes
// alpha = 0 on the nodes of the crack set K_n, those where alpha_n <= eta_lim
// at the start of the load step; set_crack_set builds it, together with the
// initial crack, the dofs of the faces given by set_initial_crack, where
// alpha = 0 at every step, as the damage boundary condition of mef90
// (damageBC, boundaryDamage 1); with vi, both bounds are 0 there. It owns the
// global assembly and the solvers, and is the mfem::Operator that its
// NewtonSolver, or the SNES of PETSc, solves, through Mult and GetGradient.
// Newton's method works on the vector of the dofs this rank owns, with the
// norms over all ranks.
//
// The linear systems are solved by GMRES, or CG, the tangent being
// symmetric positive definite, preconditioned by BoomerAMG, or by MUMPS, by
// linear_solver of the solver_fracture section of config.yaml; with vi, by
// the KSP and PC of PETSc of the same names, which the PETSc options of the
// prefix fracture_ may change, e.g. -fracture_ksp_monitor.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_FRACTURE_HPP
#define NONLINEAR_SOLVER_FRACTURE_HPP

#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>
#include <petscdmshell.h>
#include <petscsnes.h>
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
        linear_solver(make_iterative_solver(get_linear_solver_name(solver),
                                            global_assembly->get_comm())),
        newton_monitor(newton_indent),
        lower_bound(global_assembly->get_num_dofs()),
        upper_bound(global_assembly->get_num_dofs())
   {
      amg.SetPrintLevel(0);
      set_linear_solver(*linear_solver, amg);
      set_newton_solver(solver, *linear_solver, newton_monitor);
      lower_bound = 0.0;
      upper_bound = 1.0;

      const std::string nonlinear_solver =
         solver["nonlinear_solver"] ? solver["nonlinear_solver"].as<std::string>() : "newton";
      MFEM_VERIFY(nonlinear_solver == "newton" || nonlinear_solver == "vi",
                  "solver_fracture.nonlinear_solver must be newton or vi, not "
                  << nonlinear_solver << ".");
      if (nonlinear_solver == "vi")
         set_vi_solver(solver);
   }

   // The initial crack: the dofs this rank owns on the faces of the crack,
   // where alpha = 0 at every step. Returns the number of its dofs over all
   // ranks.
   int set_initial_crack(const mfem::Array<int> &input_initial_crack)
   {
      initial_crack = input_initial_crack;
      initial_crack.Sort();
      initial_crack.Unique();

      int size = initial_crack.Size();
      MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_INT, MPI_SUM, global_assembly->get_comm());
      return size;
   }

   // The crack set K_n of the load step: the dofs this rank owns where
   // alpha <= eta_lim, and those of the initial crack, set to alpha = 0 and
   // constrained. Returns the number of its dofs over all ranks.
   int set_crack_set(mfem::ParGridFunction &alpha, double eta_lim)
   {
      mfem::Vector alpha_owned(global_assembly->get_num_dofs());
      alpha.GetTrueDofs(alpha_owned);

      mfem::Array<int> crack_set(initial_crack);
      for (int dof = 0; dof < alpha_owned.Size(); dof++)
         if (alpha_owned(dof) <= eta_lim)
            crack_set.Append(dof);
      crack_set.Sort();
      crack_set.Unique();
      for (int dof : crack_set)
         alpha_owned(dof) = 0.0;
      global_assembly->set_ess_tdof_list(crack_set);

      // The bounds of vi: 0 <= alpha <= 1, both 0 on the crack set.
      lower_bound = 0.0;
      upper_bound = 1.0;
      for (int dof : crack_set)
         upper_bound(dof) = 0.0;
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

      // Newton iterations for R(alpha) = 0, within the bounds with vi.
      const int iterations = vi_solver ? solve_vi(time, alpha_owned)
                                       : solve_newton(time, alpha_owned);
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
      if (!is_direct() && !vi_solver)
         amg.SetOperator(tangent);
      return tangent;
   }

private:
   // Prints the residual norm of every iteration of the SNES of PETSc, as
   // SystemTools::NewtonMonitor does for MFEM's NewtonSolver: absolute and
   // relative to the first iteration, indented by indent spaces. With the
   // bounds, the norm is that of the residual on the dofs away from them.
   class VI_Monitor : public mfem::PetscSolverMonitor
   {
   public:
      explicit VI_Monitor(int input_indent) : indent(input_indent) {}

      // Required by MFEM: overrides mfem::PetscSolverMonitor::
      // MonitorResidual, which the SNES calls at every iteration.
      void MonitorResidual(PetscInt it, PetscReal norm, const mfem::Vector &) override
      {
         if (it == 0)
            initial_norm = norm;
         mfem::out << std::string(indent, ' ') << std::left << std::setw(12) << it
                   << std::scientific << std::setprecision(6) << std::setw(18) << norm
                   << (initial_norm > 0.0 ? norm / initial_norm : 0.0)
                   << std::defaultfloat << '\n';
      }

   private:
      const int indent;            // spaces before each line
      double initial_norm = 1.0;
   };

   // The SNES of PETSc, SNESVINEWTONRSLS, on this operator, with the Newton
   // tolerances of the solver section of config.yaml, no step tolerance, so
   // that it does not stop on a small step before the residual tolerance, and
   // the KSP and PC of linear_solver: GMRES or CG with BoomerAMG of hypre, or
   // MUMPS. The PETSc options of the prefix fracture_ may change them.
   void set_vi_solver(const YAML::Node &solver)
   {
      vi_solver = std::make_unique<mfem::PetscNonlinearSolver>(global_assembly->get_comm(), *this,
                                                               "fracture_");
      vi_solver->iterative_mode = true;
      vi_solver->SetMonitor(&vi_monitor);

      SNES snes = *vi_solver;
      SNESSetType(snes, SNESVINEWTONRSLS);
      // The VI solver makes its work vectors from the DM of the SNES, a
      // DMShell, which needs a vector of the layout of the unknowns.
      vi_layout = std::make_unique<mfem::PetscParVector>(global_assembly->get_comm(), *this);
      DM dm;
      SNESGetDM(snes, &dm);
      DMShellSetGlobalVector(dm, *vi_layout);
      SNESSetTolerances(snes, solver["newton_abs_tol"].as<double>(),
                        solver["newton_rel_tol"].as<double>(), 0.0,
                        solver["newton_max_iter"].as<int>(), PETSC_CURRENT);

      KSP ksp;
      PC pc;
      SNESGetKSP(snes, &ksp);
      KSPGetPC(ksp, &pc);
      const std::string name = get_linear_solver_name(solver);
      if (name == "mumps")
      {
         KSPSetType(ksp, KSPPREONLY);
         PCSetType(pc, PCLU);
         PCFactorSetMatSolverType(pc, MATSOLVERMUMPS);
      }
      else
      {
         KSPSetType(ksp, name == "cg" ? KSPCG : KSPGMRES);
         KSPSetTolerances(ksp, 1.0e-8, 0.0, PETSC_CURRENT, 200);
         PCSetType(pc, PCHYPRE);
         PCHYPRESetType(pc, "boomeramg");
      }
   }

   // The SNES iterations from alpha_owned within the bounds; aborts unless
   // they converge, tt being the time of the step. Returns the number of
   // iterations.
   int solve_vi(double tt, mfem::Vector &alpha_owned)
   {
      const MPI_Comm comm = global_assembly->get_comm();
      mfem::PetscParVector lower(comm, lower_bound, true), upper(comm, upper_bound, true);
      SNESVISetVariableBounds(*vi_solver, lower, upper);
      vi_solver->Mult(mfem::Vector(), alpha_owned);
      MFEM_VERIFY(vi_solver->GetConverged(),
                  "The phase field did not converge within its bounds at t = " << tt << ".");
      return vi_solver->GetNumIterations();
   }

   // CG for linear_solver cg, else GMRES, also for mumps, which does not
   // use it.
   static std::unique_ptr<mfem::IterativeSolver> make_iterative_solver(const std::string &name,
                                                                       MPI_Comm comm)
   {
      if (name == "cg")
         return std::make_unique<mfem::CGSolver>(comm);
      return std::make_unique<mfem::GMRESSolver>(comm);
   }

   // The tangent is owned by global_assembly.
   const std::unique_ptr<GlobalAssembly_Fracture> global_assembly;   // E, R and K
   mfem::Array<int> initial_crack;                                   // dofs where alpha = 0 always
   mutable mfem::HypreBoomerAMG amg;                                 // preconditioner of GMRES
   const std::unique_ptr<mfem::IterativeSolver> linear_solver;       // GMRES or CG
   SystemTools::NewtonMonitor newton_monitor;                        // prints the residual norms
   static constexpr int newton_indent = 5;                           // spaces before the Newton lines
   mfem::Vector lower_bound, upper_bound;                            // bounds of alpha, with vi
   VI_Monitor vi_monitor{newton_indent};                             // prints the residual norms of vi
   std::unique_ptr<mfem::PetscParVector> vi_layout;                  // layout of the unknowns, with vi
   std::unique_ptr<mfem::PetscNonlinearSolver> vi_solver;            // SNESVINEWTONRSLS, with vi
};

#endif

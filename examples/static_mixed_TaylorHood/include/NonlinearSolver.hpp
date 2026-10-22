// ============================================================================
// NonlinearSolver.hpp
//
// Solves one load step of the static mixed form, R(u,p) = 0: sets the load of
// the step, makes the consistent predictor the initial guess, and runs
// Newton's method from it. It owns the global assembly and the solvers; the
// loop over the load steps is left to its caller. It is also the
// mfem::Operator that its NewtonSolver solves, through Mult and GetGradient.
// The displacement and the pressure are ParGridFunctions, the boundary values
// set on the displacement; Newton's method works on the vector of their dofs
// this rank owns, with the norms over all ranks.
//
// The linear systems are solved iteratively. The tangent
//    K = [A  B^T; B  C],   A = K_uu,   B = K_pu,   C = K_pp = -M_p[1 / kappa],
// with M_p[w] = int w N_c N_d dV, is solved by GMRES, preconditioned by its
// block lower triangle,
//    y_u = A^-1 x_u,   y_p = -S^-1 (x_p - B y_u),   S = B A^-1 B^T - C,
// with BoomerAMG for A and, as for the Stokes problem, the pressure mass
// for S, B A^-1 B^T ~ M_p[1 / mu], so that S ~ M_p[1 / mu + 1 / kappa],
// of the reference configuration, inverted by one Jacobi sweep.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_HPP
#define NONLINEAR_SOLVER_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Mixed.hpp"
#include "INonlinearSolver.hpp"
#include "LoadData.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"

class NonlinearSolver : public INonlinearSolver
{
public:
   // Takes the ownership of the global assembly, and assembles on space_p
   // the pressure mass of the preconditioner; the Newton settings come from
   // the solver section of config.yaml.
   NonlinearSolver(mfem::ParFiniteElementSpace &space_p,
                   std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
                   const YAML::Node &solver)
      : INonlinearSolver(input_global_assembly->get_num_dofs(), input_global_assembly->get_comm()),
        global_assembly(std::move(input_global_assembly)),
        schur_inverse(*this, space_p.GetTrueVSize()),
        preconditioner(global_assembly->get_offsets()),
        linear_solver(global_assembly->get_comm()),
        newton_monitor(global_assembly->get_offsets(), global_assembly->get_comm())
   {
      // S ~ M_p[1 / mu + 1 / kappa], inverted by its diagonal.
      mfem::ConstantCoefficient weight(1.0 / shear_modulus + 1.0 / bulk_modulus);
      mfem::ParBilinearForm form(&space_p);
      form.AddDomainIntegrator(new mfem::MassIntegrator(weight));
      form.Assemble();
      form.Finalize();
      pres_mass.reset(form.ParallelAssemble());
      pres_jacobi = std::make_unique<mfem::HypreDiagScale>(*pres_mass);

      // A is a vector Laplacian-like operator of 3 components per node.
      amg_u.SetSystemsOptions(3, false);
      amg_u.SetPrintLevel(0);
      preconditioner.SetDiagonalBlock(1, &schur_inverse);

      set_linear_solver(linear_solver, preconditioner);
      set_newton_solver(solver, linear_solver, newton_monitor);
   }

   // Solves the load at time tt from the converged disp and pres of the
   // previous step, and returns the number of Newton iterations.
   int solve(double tt, mfem::ParGridFunction &disp, mfem::ParGridFunction &pres)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();

      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);

      // Newton's method works on one vector of both fields, sol = [u; p],
      // their values on the dofs this rank owns.
      mfem::BlockVector sol(global_assembly->get_offsets());
      disp.GetTrueDofs(sol.GetBlock(0));
      pres.GetTrueDofs(sol.GetBlock(1));

      initial_guess(tt, disp, sol);

      // The load value: the prescribed displacement, or the traction faces.
      if (dirichlet.is_disp_load())
         dirichlet.print_disp_load_by_step(disp);
      else
         neumann.print_traction_load_by_step();
      SystemTools::print_block_newton_header();

      // Newton iterations for R(u,p) = 0.
      const int iterations = solve_newton(tt, sol);

      disp.SetFromTrueDofs(sol.GetBlock(0));
      pres.SetFromTrueDofs(sol.GetBlock(1));
      return iterations;
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
   // K(u,p), the identity on the constrained dofs; also sets the blocks A
   // and B of the preconditioner.
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      set_tangent(sol);
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

private:
   // -S^-1 ~ -M_p[1 / mu + 1 / kappa]^-1, one Jacobi sweep.
   class SchurInverse : public mfem::Solver
   {
   public:
      // size: the pressure dofs this rank owns.
      SchurInverse(const NonlinearSolver &input_owner, int size)
         : mfem::Solver(size), owner(input_owner) {}

      void SetOperator(const mfem::Operator &op) override {}

      void Mult(const mfem::Vector &xx, mfem::Vector &yy) const override
      {
         owner.pres_jacobi->Mult(xx, yy);
         yy.Neg();
      }

   private:
      const NonlinearSolver &owner;
   };

   // K(u,p) on the dofs this rank owns, without constraints, into tangent,
   // with its blocks A and B, constrained, for the preconditioner: A the
   // identity on the displacement constraints, B without their columns;
   // with MUMPS, no preconditioner.
   void set_tangent(const mfem::Vector &sol) const
   {
      mfem::Array2D<const mfem::HypreParMatrix *> blocks =
         global_assembly->assemble_tangent_blocks(sol);
      tangent.reset(mfem::HypreParMatrixFromBlocks(blocks));
      if (is_direct())
         return;
      tangent_uu = std::make_unique<mfem::HypreParMatrix>(*blocks(0, 0));
      tangent_pu = std::make_unique<mfem::HypreParMatrix>(*blocks(1, 0));
      const mfem::Array<int> ess_tdof_list = global_assembly->get_dirichlet().get_ess_tdof_list();
      tangent_uu->EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
      delete tangent_pu->EliminateCols(ess_tdof_list);
      amg_u.SetOperator(*tangent_uu);
      preconditioner.SetDiagonalBlock(0, &amg_u);
      preconditioner.SetBlock(1, 0, tangent_pu.get());
   }

   // Consistent predictor: the initial guess of Newton's method is one
   // linear step from the converged state (u,p) with the new load, so that
   // the interior follows the prescribed boundary increment g instead of
   // only the boundary nodes moving,
   //    K_ff d(u,p)_f = -R_f(u,p) - K_fe g,
   // with K_fe g moved to the right-hand side by set_essential_bdr. disp
   // and sol, the values of both fields on the dofs this rank owns, are both
   // updated.
   void initial_guess(double tt, mfem::ParGridFunction &disp, mfem::BlockVector &sol)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();

      // Zero on the fixed faces, the prescribed values at time tt on the
      // displacement-driven ones; g is their difference from disp, and zero
      // for the pressure.
      mfem::ParGridFunction disp_target(disp.ParFESpace());
      disp_target = disp;
      dirichlet.apply_fixed_bc(disp_target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_load_bc(tt, LoadData::disp_loading, disp_target);
      mfem::Vector target_owned(sol.GetBlock(0).Size());
      disp_target.GetTrueDofs(target_owned);
      mfem::BlockVector prescribed_increment(global_assembly->get_offsets());
      prescribed_increment = 0.0;
      prescribed_increment.GetBlock(0) = target_owned;
      prescribed_increment.GetBlock(0) -= sol.GetBlock(0);

      mfem::Vector rhs(global_assembly->get_num_dofs());
      global_assembly->assemble_residual(sol, rhs);
      rhs.Neg();
      set_tangent(sol);
      global_assembly->set_essential_bdr(*tangent, prescribed_increment, rhs);

      mfem::Vector predicted_increment(global_assembly->get_num_dofs());
      predicted_increment = 0.0;
      get_linear_solver().SetOperator(*tangent);
      get_linear_solver().Mult(rhs, predicted_increment);
      sol += predicted_increment;

      // Set the prescribed values exactly, free of round-off; the
      // constrained dofs are in the displacement block, the first.
      for (int dof : dirichlet.get_ess_tdof_list())
         sol(dof) = target_owned(dof);
      disp.SetFromTrueDofs(sol.GetBlock(0));
   }

   // The preconditioner points to amg_u and schur_inverse, which reads
   // pres_jacobi.
   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly;   // R and K
   std::unique_ptr<mfem::HypreParMatrix> pres_mass;               // M_p[1 / mu + 1 / kappa]
   std::unique_ptr<mfem::HypreDiagScale> pres_jacobi;             // diagonal of pres_mass, inverted
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent;         // K of Newton's method
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent_uu;      // A, constrained
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent_pu;      // B, constrained columns removed
   mutable mfem::HypreBoomerAMG amg_u;                            // A^-1
   SchurInverse schur_inverse;                                    // -S^-1
   mutable mfem::BlockLowerTriangularPreconditioner preconditioner;
   mfem::GMRESSolver linear_solver;                               // solver of the tangent
   SystemTools::BlockNewtonMonitor newton_monitor;                // prints the residual norms of u and p
};

#endif

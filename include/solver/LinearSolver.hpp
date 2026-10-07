// ============================================================================
// LinearSolver.hpp
//
// The parallel linear solver of the tangents of Newton's method and of the
// mass matrix, chosen by the linear_solver section of config.yaml:
//    mumps      MUMPS, the parallel direct solver; a 2 x 2 block tangent of
//               the mixed form is first copied into one HypreParMatrix;
//    iterative  Krylov, preconditioned by BoomerAMG, the algebraic multigrid
//               of hypre. A matrix, the displacement tangent or the mass, is
//               solved by CG with one BoomerAMG: it is symmetric, and
//               positive definite away from instabilities. A block tangent
//               [K_uu K_up; K_pu K_pp], symmetric but indefinite, is solved
//               by BiCGSTAB with the block-diagonal preconditioner
//               diag(AMG(K_uu), AMG(S)), with
//                  S = K_pu D^-1 K_up - K_pp,   D = diag(K_uu),
//               which approximates minus the Schur complement of K_uu.
// On 4 ranks the Krylov solvers of the block tangent, MINRES and BiCGSTAB
// alike, now and then stop on a NaN at the start of a solve, at a random
// step; the same run passes when repeated, and never fails on 1 or 2 ranks
// or with MUMPS. The cause is not found. It is an mfem::Solver, so that
// NewtonSolver passes it the tangent of every iteration through SetOperator.
//
// Author: Chongran Zhao
// Date: Oct. 6, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LINEAR_SOLVER_HPP
#define LINEAR_SOLVER_HPP

#include <memory>
#include <string>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

class LinearSolver : public mfem::Solver
{
public:
   // Reads the linear_solver section of config.yaml; comm is that of the
   // matrices to solve.
   LinearSolver(const YAML::Node &paras, MPI_Comm comm)
      : direct_solver(comm), matrix_solver(comm), block_solver(comm)
   {
      const std::string type = paras["type"].as<std::string>();
      MFEM_VERIFY(type == "mumps" || type == "iterative",
                  "Unknown linear_solver type \"" << type << "\", mumps or iterative.");
      is_direct = (type == "mumps");

      direct_solver.SetPrintLevel(0);

      if (!is_direct)
      {
         const auto print_level = paras["print"].as<bool>()
            ? mfem::IterativeSolver::PrintLevel().Warnings().Summary()
            : mfem::IterativeSolver::PrintLevel().Warnings();
         for (mfem::IterativeSolver *krylov :
              {static_cast<mfem::IterativeSolver *>(&matrix_solver),
               static_cast<mfem::IterativeSolver *>(&block_solver)})
         {
            krylov->SetRelTol(paras["rel_tol"].as<double>());
            krylov->SetAbsTol(paras["abs_tol"].as<double>());
            krylov->SetMaxIter(paras["max_iter"].as<int>());
            krylov->SetPrintLevel(print_level);
         }
      }
   }

   // Required by MFEM: overrides mfem::Solver::SetOperator, which
   // NewtonSolver calls at every iteration with the tangent.
   // op is a HypreParMatrix, or a 2 x 2 BlockOperator of HypreParMatrix
   // blocks; it must outlive the solves with it.
   void SetOperator(const mfem::Operator &op) override
   {
      height = op.Height();
      width = op.Width();

      const auto *matrix = dynamic_cast<const mfem::HypreParMatrix *>(&op);
      is_block = !matrix;
      if (matrix)
         set_matrix(*matrix);
      else
         set_block_matrix(dynamic_cast<const mfem::BlockOperator &>(op));
   }

   // Required by MFEM: overrides mfem::Operator::Mult, which NewtonSolver
   // calls at every iteration for the Newton increment.
   // sol = op^-1 rhs.
   void Mult(const mfem::Vector &rhs, mfem::Vector &sol) const override
   {
      if (is_direct)
         direct_solver.Mult(rhs, sol);
      else if (is_block)
         block_solver.Mult(rhs, sol);
      else
         matrix_solver.Mult(rhs, sol);
   }

private:
   // A matrix: MUMPS, or CG with BoomerAMG. The AMG of a displacement
   // matrix treats the three components of a node, consecutive in byVDIM, as
   // one unknown.
   void set_matrix(const mfem::HypreParMatrix &matrix)
   {
      if (is_direct)
      {
         direct_solver.SetOperator(matrix);
         return;
      }

      amg_u = std::make_unique<mfem::HypreBoomerAMG>(matrix);
      amg_u->SetSystemsOptions(3, false);
      amg_u->SetPrintLevel(0);
      matrix_solver.SetPreconditioner(*amg_u);
      matrix_solver.SetOperator(matrix);
   }

   // A block tangent: MUMPS on its copy in one matrix, or BiCGSTAB with the
   // block-diagonal preconditioner.
   void set_block_matrix(const mfem::BlockOperator &tangent)
   {
      const auto &K_uu = dynamic_cast<const mfem::HypreParMatrix &>(tangent.GetBlock(0, 0));
      const auto &K_up = dynamic_cast<const mfem::HypreParMatrix &>(tangent.GetBlock(0, 1));
      const auto &K_pu = dynamic_cast<const mfem::HypreParMatrix &>(tangent.GetBlock(1, 0));
      const auto &K_pp = dynamic_cast<const mfem::HypreParMatrix &>(tangent.GetBlock(1, 1));

      if (is_direct)
      {
         mfem::Array2D<const mfem::HypreParMatrix *> blocks(2, 2);
         blocks(0, 0) = &K_uu;
         blocks(0, 1) = &K_up;
         blocks(1, 0) = &K_pu;
         blocks(1, 1) = &K_pp;
         monolithic.reset(mfem::HypreParMatrixFromBlocks(blocks));
         direct_solver.SetOperator(*monolithic);
         return;
      }

      // S = K_pu D^-1 K_up - K_pp.
      mfem::Vector diag_uu;
      K_uu.GetDiag(diag_uu);
      mfem::HypreParMatrix scaled_up(K_up);
      scaled_up.InvScaleRows(diag_uu);
      const std::unique_ptr<mfem::HypreParMatrix> product(mfem::ParMult(&K_pu, &scaled_up));
      schur.reset(mfem::Add(1.0, *product, -1.0, K_pp));

      amg_u = std::make_unique<mfem::HypreBoomerAMG>(K_uu);
      amg_u->SetSystemsOptions(3, false);
      amg_u->SetPrintLevel(0);
      amg_s = std::make_unique<mfem::HypreBoomerAMG>(*schur);
      amg_s->SetPrintLevel(0);

      block_preconditioner = std::make_unique<mfem::BlockDiagonalPreconditioner>(
         tangent.RowOffsets());
      block_preconditioner->SetDiagonalBlock(0, amg_u.get());
      block_preconditioner->SetDiagonalBlock(1, amg_s.get());
      block_solver.SetPreconditioner(*block_preconditioner);
      block_solver.SetOperator(tangent);
   }

   bool is_direct = true;                                    // MUMPS, or Krylov
   bool is_block = false;                                    // whether the operator is a block tangent
   mfem::MUMPSSolver direct_solver;                          // factorization of the operator
   std::unique_ptr<mfem::HypreParMatrix> monolithic;         // block tangent in one matrix, for MUMPS

   // Of the Krylov solvers, which point to the preconditioners, so they are
   // declared last and go first.
   std::unique_ptr<mfem::HypreParMatrix> schur;              // S
   std::unique_ptr<mfem::HypreBoomerAMG> amg_u;              // AMG of the matrix, or of K_uu
   std::unique_ptr<mfem::HypreBoomerAMG> amg_s;              // AMG of S
   std::unique_ptr<mfem::BlockDiagonalPreconditioner> block_preconditioner;  // diag(AMG(K_uu), AMG(S))
   mfem::CGSolver matrix_solver;                             // of a matrix
   mfem::BiCGSTABSolver block_solver;                        // of a block tangent
};

#endif

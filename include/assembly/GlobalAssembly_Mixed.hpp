// ============================================================================
// GlobalAssembly_Mixed.hpp
//
// Global mixed residual R(u,p) = F_int(u,p) - F_ext and its full tangent.
// One unconstrained block form assembles both Newton and predictor data.
// Boundary elimination is performed on a monolithic copy of the tangent,
// as in GlobalAssembly_Disp.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef GLOBAL_ASSEMBLY_MIXED_HPP
#define GLOBAL_ASSEMBLY_MIXED_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "DirichletBoundary.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "NeumannBoundary.hpp"

class GlobalAssembly_Mixed
{
public:
   // The block form takes ownership of the local integrator, which owns its
   // material. Unlike NonlinearForm, BlockNonlinearForm has no external-
   // integrator ownership option.
   GlobalAssembly_Mixed(mfem::FiniteElementSpace &space_u,
                        mfem::FiniteElementSpace &space_p,
                        std::unique_ptr<LocalAssembly_Mixed> input_local_assembly,
                        std::unique_ptr<DirichletBoundary> input_dirichlet,
                        std::unique_ptr<NeumannBoundary> input_neumann)
      : dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        spaces({&space_u, &space_p}),
        offsets({0, space_u.GetTrueVSize(),
                    space_u.GetTrueVSize() + space_p.GetTrueVSize()}),
        global_assembly(spaces),
        external_force(&space_u), ess_tdof_list(dirichlet->get_ess_tdof_list())
   {
      global_assembly.AddDomainIntegrator(input_local_assembly.get());
      input_local_assembly.release();
      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(external_force);
   }

   int get_num_dofs() const { return offsets.Last(); }
   const mfem::Array<int> &get_offsets() const { return offsets; }
   const DirichletBoundary &get_dirichlet() const { return *dirichlet; }
   const NeumannBoundary &get_neumann() const { return *neumann; }

   // Internal field views of the nonlinear solver's block state.
   void make_solution_views(mfem::BlockVector &sol, mfem::GridFunction &disp,
                            mfem::GridFunction &pres) const
   {
      disp.MakeRef(spaces[0], sol.GetBlock(0), 0);
      pres.MakeRef(spaces[1], sol.GetBlock(1), 0);
   }

   void set_traction_load(double tt)
   {
      neumann->set_time(tt);
      external_force.Assemble();
   }

   // R at every dof, including support reactions. Pressure has no external load.
   void set_residual(const mfem::Vector &sol, mfem::Vector &residual) const
   {
      global_assembly.Mult(sol, residual);
      for (int ii = 0; ii < external_force.Size(); ii++)
         residual(ii) -= external_force[ii];
   }

   // Convert the full block tangent to one owned sparse matrix. The block
   // form keeps its matrices; neither elimination nor UMFPACK modifies them.
   std::unique_ptr<mfem::SparseMatrix> get_tangent(const mfem::Vector &sol) const
   {
      const auto &block_op = dynamic_cast<const mfem::BlockOperator &>(
         global_assembly.GetGradient(sol));
      mfem::BlockMatrix block_mat(block_op.RowOffsets(), block_op.ColOffsets());
      for (int ii = 0; ii < block_op.NumRowBlocks(); ii++)
         for (int jj = 0; jj < block_op.NumColBlocks(); jj++)
            if (!block_op.IsZeroBlock(ii, jj))
               block_mat.SetBlock(ii, jj, const_cast<mfem::SparseMatrix *>(
                  &dynamic_cast<const mfem::SparseMatrix &>(block_op.GetBlock(ii, jj))));
      return std::unique_ptr<mfem::SparseMatrix>(block_mat.CreateMonolithic());
   }

   void set_essential_bdr(mfem::Vector &residual) const
   {
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }
   void set_essential_bdr(mfem::SparseMatrix &tangent) const
   {
      for (int dof : ess_tdof_list)
         tangent.EliminateRowCol(dof, mfem::Operator::DIAG_ONE);
   }
   void set_essential_bdr(mfem::SparseMatrix &tangent,
                          const mfem::Vector &prescribed_increment,
                          mfem::Vector &rhs) const
   {
      for (int dof : ess_tdof_list)
         tangent.EliminateRowCol(dof, prescribed_increment(dof), rhs);
   }


private:
   // Boundary coefficients and spaces outlive the forms that borrow them.
   const std::unique_ptr<DirichletBoundary> dirichlet;
   const std::unique_ptr<NeumannBoundary> neumann;
   mfem::Array<mfem::FiniteElementSpace *> spaces;
   const mfem::Array<int> offsets;
   mfem::BlockNonlinearForm global_assembly;
   mfem::LinearForm external_force;
   const mfem::Array<int> ess_tdof_list;
};

#endif

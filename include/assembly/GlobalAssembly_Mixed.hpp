// ============================================================================
// GlobalAssembly_Mixed.hpp
//
// Global assembly of the mixed displacement-pressure form, with the
// Dirichlet and the Neumann boundary conditions:
//    external force  F_ext, from the tractions, on the displacement only;
//    residual        R(u,p) = F_int(u,p) - F_ext, from one block form over
//                    LocalAssembly_Mixed;
//    tangent         K(u,p) = dR/d(u,p), assembled from the element tangents.
// set_essential_bdr sets R to zero and K to the identity on the constrained
// displacement dofs, and, with an increment of the prescribed displacement,
// also moves the constrained columns of K to the right-hand side.
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

#include "BlockNonlinearForm_External.hpp"
#include "DirichletBoundary.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "NeumannBoundary.hpp"

class GlobalAssembly_Mixed
{
public:
   // Takes the ownership of the local assembly and of the boundary
   // conditions; the global_assembly only borrows the local assembly.
   GlobalAssembly_Mixed(mfem::FiniteElementSpace &space_u,
                        mfem::FiniteElementSpace &space_p,
                        std::unique_ptr<LocalAssembly_Mixed> input_local_assembly,
                        std::unique_ptr<DirichletBoundary> input_dirichlet,
                        std::unique_ptr<NeumannBoundary> input_neumann)
      : local_assembly(std::move(input_local_assembly)),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        external_force(&space_u),
        ess_tdof_list(dirichlet->get_ess_tdof_list())
   {
      mfem::Array<mfem::FiniteElementSpace *> spaces({&space_u, &space_p});
      global_assembly.SetSpaces(spaces);
      global_assembly.AddDomainIntegrator(local_assembly.get());

      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(external_force);
   }

   // Set the tractions to time tt and assemble F_ext.
   void set_traction_load(double tt)
   {
      neumann->set_time(tt);
      external_force.Assemble();
   }

   // Number of unknowns, displacement and pressure.
   int get_num_dofs() const { return global_assembly.Height(); }

   // Offsets of the displacement and the pressure blocks, [0, n_u, n_u + n_p].
   const mfem::Array<int> &get_offsets() const
   {
      return global_assembly.GetBlockTrueOffsets();
   }

   // Consistent mass M_ab = int rho_0 N_a N_b dV of the displacement, without
   // constraints, with rho_0 of the material; the pressure has no mass. It is
   // assembled once, for the dynamics, and the caller owns it.
   std::unique_ptr<mfem::SparseMatrix> assemble_mass()
   {
      mfem::ConstantCoefficient rho(local_assembly->get_rho_0());
      mfem::BilinearForm mass(global_assembly.FESpace(0));
      mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
      mass.Assemble();
      mass.Finalize();
      return std::unique_ptr<mfem::SparseMatrix>(mass.LoseMat());
   }

   // R(u,p) at every dof; the pressure has no external force.
   void assemble_residual(const mfem::Vector &sol, mfem::Vector &residual) const
   {
      global_assembly.Mult(sol, residual);
      mfem::BlockVector residual_blocks(residual, global_assembly.GetBlockTrueOffsets());
      residual_blocks.GetBlock(0) -= external_force;
   }

   // K(u,p) at every dof. The blocks, owned by global_assembly, are copied
   // into one new SparseMatrix, which the caller owns.
   std::unique_ptr<mfem::SparseMatrix> assemble_tangent(const mfem::Vector &sol) const
   {
      const auto &block_op = dynamic_cast<const mfem::BlockOperator &>(
         global_assembly.GetGradient(sol));

      // block_mat only points to the blocks.
      mfem::BlockMatrix block_mat(block_op.RowOffsets(), block_op.ColOffsets());
      for (int ii = 0; ii < block_op.NumRowBlocks(); ii++)
         for (int jj = 0; jj < block_op.NumColBlocks(); jj++)
            if (!block_op.IsZeroBlock(ii, jj))
               block_mat.SetBlock(ii, jj, const_cast<mfem::SparseMatrix *>(
                  &dynamic_cast<const mfem::SparseMatrix &>(block_op.GetBlock(ii, jj))));
      return std::unique_ptr<mfem::SparseMatrix>(block_mat.CreateMonolithic());
   }

   // R zero on the constrained dofs, which carry no equation.
   void set_essential_bdr(mfem::Vector &residual) const
   {
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // K the identity on the constrained dofs.
   void set_essential_bdr(mfem::SparseMatrix &tangent) const
   {
      for (int dof : ess_tdof_list)
         tangent.EliminateRowCol(dof, mfem::Operator::DIAG_ONE);
   }

   // K the identity on the constrained dofs, with the increment g of the
   // prescribed displacement on them moved to the right-hand side:
   //    rhs -= K g on the free dofs,   rhs = g on the constrained dofs.
   void set_essential_bdr(mfem::SparseMatrix &tangent,
                          const mfem::Vector &prescribed_increment,
                          mfem::Vector &rhs) const
   {
      for (int dof : ess_tdof_list)
         tangent.EliminateRowCol(dof, prescribed_increment(dof), rhs);
   }

   // The Dirichlet boundary conditions, whose values the nonlinear solver sets.
   const DirichletBoundary &get_dirichlet() const { return *dirichlet; }

   // The Neumann boundary conditions.
   const NeumannBoundary &get_neumann() const { return *neumann; }

private:
   // Declared before global_assembly, so that global_assembly, which
   // borrows it, goes first.
   const std::unique_ptr<LocalAssembly_Mixed> local_assembly;
   const std::unique_ptr<DirichletBoundary> dirichlet;    // constrained dofs and their values
   const std::unique_ptr<NeumannBoundary> neumann;        // tractions
   BlockNonlinearForm_External global_assembly;           // R + F_ext and K, without constraints
   mfem::LinearForm external_force;                       // F_ext, on the displacement
   const mfem::Array<int> ess_tdof_list;                  // constrained displacement dofs
};

#endif

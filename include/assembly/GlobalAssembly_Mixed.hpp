// ============================================================================
// GlobalAssembly_Mixed.hpp
//
// Parallel global assembly of the mixed displacement-pressure form, with the
// Dirichlet and the Neumann boundary conditions, on the true dofs of each
// rank:
//    external force  F_ext, from the tractions, on the displacement only;
//    residual        R(u,p) = F_int(u,p) - F_ext, from one parallel block
//                    form over LocalAssembly_Mixed;
//    tangent         K(u,p) = dR/d(u,p), a 2 x 2 BlockOperator of
//                    HypreParMatrix blocks [K_uu K_up; K_pu K_pp],
//                    assembled from the element tangents.
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

#include "DirichletBoundary.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "NeumannBoundary.hpp"
#include "ParBlockNonlinearForm_External.hpp"

class GlobalAssembly_Mixed
{
public:
   // Takes the ownership of the local assembly and of the boundary
   // conditions, which are built on space_u; the global_assembly only
   // borrows the local assembly.
   GlobalAssembly_Mixed(mfem::ParFiniteElementSpace &space_u,
                        mfem::ParFiniteElementSpace &space_p,
                        std::unique_ptr<LocalAssembly_Mixed> input_local_assembly,
                        std::unique_ptr<DirichletBoundary> input_dirichlet,
                        std::unique_ptr<NeumannBoundary> input_neumann)
      : local_assembly(std::move(input_local_assembly)),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        traction_form(&space_u),
        external_force(space_u.GetTrueVSize()),
        ess_tdof_list(dirichlet->get_ess_tdof_list())
   {
      mfem::Array<mfem::ParFiniteElementSpace *> spaces({&space_u, &space_p});
      global_assembly.SetParSpaces(spaces);
      global_assembly.AddDomainIntegrator(local_assembly.get());

      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(traction_form);
   }

   // Set the tractions to time tt and assemble F_ext on the true dofs.
   void set_traction_load(double tt)
   {
      neumann->set_time(tt);
      traction_form.Assemble();
      traction_form.ParallelAssemble(external_force);
   }

   // Number of unknowns of this rank, displacement and pressure.
   int get_num_dofs() const { return global_assembly.Height(); }

   // Offsets of the displacement and the pressure blocks of this rank,
   // [0, n_u, n_u + n_p].
   const mfem::Array<int> &get_offsets() const
   {
      return global_assembly.GetBlockTrueOffsets();
   }

   // The ranks of the spaces.
   MPI_Comm get_comm() const { return global_assembly.ParFESpace(0)->GetComm(); }

   // Consistent mass M_ab = int rho_0 N_a N_b dV of the displacement, without
   // constraints, with rho_0 of the material; the pressure has no mass. It is
   // assembled once, for the dynamics, and the caller owns it.
   std::unique_ptr<mfem::HypreParMatrix> assemble_mass()
   {
      mfem::ConstantCoefficient rho(local_assembly->get_rho_0());
      mfem::ParBilinearForm mass(global_assembly.ParFESpace(0));
      mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
      mass.Assemble();
      mass.Finalize();
      return std::unique_ptr<mfem::HypreParMatrix>(mass.ParallelAssemble());
   }

   // R(u,p) at every true dof; the pressure has no external force.
   void assemble_residual(const mfem::Vector &sol, mfem::Vector &residual) const
   {
      global_assembly.Mult(sol, residual);
      mfem::BlockVector residual_blocks(residual, global_assembly.GetBlockTrueOffsets());
      residual_blocks.GetBlock(0) -= external_force;
   }

   // K(u,p) at every true dof. The blocks, owned by global_assembly, are
   // copied into a new BlockOperator, which owns the copies and which the
   // caller owns.
   std::unique_ptr<mfem::BlockOperator> assemble_tangent(const mfem::Vector &sol) const
   {
      const mfem::BlockOperator &block_op = global_assembly.GetGradient(sol);

      auto tangent = std::make_unique<mfem::BlockOperator>(get_offsets());
      tangent->owns_blocks = 1;
      for (int ii = 0; ii < 2; ii++)
         for (int jj = 0; jj < 2; jj++)
            tangent->SetBlock(ii, jj, new mfem::HypreParMatrix(
               dynamic_cast<const mfem::HypreParMatrix &>(block_op.GetBlock(ii, jj))));
      return tangent;
   }

   // R zero on the constrained dofs, which carry no equation.
   void set_essential_bdr(mfem::Vector &residual) const
   {
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // A matrix of the displacement, e.g. the mass, the identity on the
   // constrained dofs.
   void set_essential_bdr(mfem::HypreParMatrix &matrix_u) const
   {
      matrix_u.EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
   }

   // K the identity on the constrained dofs: their rows and columns of K_uu
   // become those of the identity, their rows of K_up and their columns of
   // K_pu zero.
   void set_essential_bdr(mfem::BlockOperator &tangent) const
   {
      auto &K_uu = dynamic_cast<mfem::HypreParMatrix &>(tangent.GetBlock(0, 0));
      auto &K_up = dynamic_cast<mfem::HypreParMatrix &>(tangent.GetBlock(0, 1));
      auto &K_pu = dynamic_cast<mfem::HypreParMatrix &>(tangent.GetBlock(1, 0));
      K_uu.EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
      K_up.EliminateRows(ess_tdof_list);
      delete K_pu.EliminateCols(ess_tdof_list);
   }

   // K the identity on the constrained dofs, with the increment g of the
   // prescribed displacement on them, zero for the pressure, moved to the
   // right-hand side:
   //    rhs -= K g on the free dofs,   rhs = g on the constrained dofs.
   // The eliminated parts of K_uu and K_pu give the first.
   void set_essential_bdr(mfem::BlockOperator &tangent,
                          const mfem::Vector &prescribed_increment,
                          mfem::Vector &rhs) const
   {
      mfem::BlockVector rhs_blocks(rhs, get_offsets());
      const mfem::Vector increment_u(prescribed_increment.GetData(), get_offsets()[1]);

      auto &K_uu = dynamic_cast<mfem::HypreParMatrix &>(tangent.GetBlock(0, 0));
      auto &K_up = dynamic_cast<mfem::HypreParMatrix &>(tangent.GetBlock(0, 1));
      auto &K_pu = dynamic_cast<mfem::HypreParMatrix &>(tangent.GetBlock(1, 0));

      const std::unique_ptr<mfem::HypreParMatrix> eliminated_uu(
         K_uu.EliminateRowsCols(ess_tdof_list));
      K_uu.EliminateBC(*eliminated_uu, ess_tdof_list, increment_u, rhs_blocks.GetBlock(0));

      K_up.EliminateRows(ess_tdof_list);

      const std::unique_ptr<mfem::HypreParMatrix> eliminated_pu(
         K_pu.EliminateCols(ess_tdof_list));
      eliminated_pu->Mult(-1.0, increment_u, 1.0, rhs_blocks.GetBlock(1));
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
   ParBlockNonlinearForm_External global_assembly;        // R + F_ext and K, without constraints
   mfem::ParLinearForm traction_form;                     // F_ext on the local dofs
   mfem::Vector external_force;                           // F_ext on the true dofs
   const mfem::Array<int> ess_tdof_list;                  // constrained displacement true dofs
};

#endif

// ============================================================================
// GlobalAssembly_Mixed.hpp
//
// Parallel global assembly of the mixed displacement-pressure form, with the
// Dirichlet and the Neumann boundary conditions, on the dofs each rank owns:
//    external force  F_ext, from the tractions, on the displacement only;
//    residual        R(u,p) = F_int(u,p) - F_ext, from one parallel block
//                    form over LocalAssembly_Mixed;
//    tangent         K(u,p) = dR/d(u,p), one HypreParMatrix of both fields,
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
#include "IGlobalAssembly.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "NeumannBoundary.hpp"
#include "ParBlockNonlinearForm_External.hpp"

class GlobalAssembly_Mixed : public IGlobalAssembly
{
public:
   // Takes the ownership of the local assembly and of the boundary
   // conditions; the global_assembly only borrows the local assembly.
   GlobalAssembly_Mixed(mfem::ParFiniteElementSpace &space_u,
                        mfem::ParFiniteElementSpace &space_p,
                        std::unique_ptr<LocalAssembly_Mixed> input_local_assembly,
                        std::unique_ptr<DirichletBoundary> input_dirichlet,
                        std::unique_ptr<NeumannBoundary> input_neumann)
      : local_assembly(std::move(input_local_assembly)),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        local_traction(&space_u),
        external_force(space_u.GetTrueVSize()),
        ess_tdof_list(dirichlet->get_ess_tdof_list())
   {
      mfem::Array<mfem::ParFiniteElementSpace *> spaces({&space_u, &space_p});
      global_assembly.SetParSpaces(spaces);
      global_assembly.AddDomainIntegrator(local_assembly.get());

      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(local_traction);
   }

   // Set the tractions to time tt and assemble F_ext: each rank integrates
   // over its boundary elements, and ParallelAssemble sums the parts of the
   // shared dofs onto the rank that owns them.
   void set_traction_load(double tt) override
   {
      neumann->set_time(tt);
      local_traction.Assemble();
      local_traction.ParallelAssemble(external_force);
   }

   // Number of unknowns this rank owns, displacement and pressure.
   int get_num_dofs() const override { return global_assembly.Height(); }

   // Offsets of the displacement and the pressure blocks of the dofs this
   // rank owns, [0, n_u, n_u + n_p].
   const mfem::Array<int> &get_offsets() const
   {
      return global_assembly.GetBlockTrueOffsets();
   }

   // The ranks of the spaces.
   MPI_Comm get_comm() const override { return global_assembly.ParFESpace(0)->GetComm(); }

   // Consistent mass M_ab = int rho_0 N_a N_b dV of the displacement, without
   // constraints, with rho_0 of the material; the pressure has no mass. It is
   // assembled once, for the dynamics, and the caller owns it.
   std::unique_ptr<mfem::HypreParMatrix> assemble_mass() override
   {
      mfem::ConstantCoefficient rho(local_assembly->get_rho_0());
      mfem::ParBilinearForm mass(global_assembly.ParFESpace(0));
      mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
      mass.Assemble();
      mass.Finalize();
      return std::unique_ptr<mfem::HypreParMatrix>(mass.ParallelAssemble());
   }

   // R(u,p) at every dof this rank owns; the pressure has no external force.
   void assemble_residual(const mfem::Vector &sol, mfem::Vector &residual) const override
   {
      global_assembly.Mult(sol, residual);
      mfem::BlockVector residual_blocks(residual, global_assembly.GetBlockTrueOffsets());
      residual_blocks.GetBlock(0) -= external_force;
   }

   // The blocks [K_uu K_up; K_pu K_pp] of K(u,p) on the dofs this rank owns.
   // global_assembly owns them and builds them anew at every call.
   mfem::Array2D<const mfem::HypreParMatrix *> assemble_tangent_blocks(const mfem::Vector &sol) const
   {
      const mfem::BlockOperator &block_op = global_assembly.GetGradient(sol);

      mfem::Array2D<const mfem::HypreParMatrix *> blocks(2, 2);
      for (int ii = 0; ii < 2; ii++)
         for (int jj = 0; jj < 2; jj++)
            blocks(ii, jj) = block_op.IsZeroBlock(ii, jj) ? nullptr :
               &dynamic_cast<const mfem::HypreParMatrix &>(block_op.GetBlock(ii, jj));
      return blocks;
   }

   // K(u,p) on the dofs this rank owns, with its blocks merged into one
   // matrix owned by this assembly. The caller may change it until the
   // next assemble_tangent call; constrained dofs keep their block numbers.
   mfem::HypreParMatrix &assemble_tangent(const mfem::Vector &sol) const override
   {
      mfem::Array2D<const mfem::HypreParMatrix *> blocks = assemble_tangent_blocks(sol);
      tangent.reset(mfem::HypreParMatrixFromBlocks(blocks));
      return *tangent;
   }

   // R zero on the constrained dofs, which carry no equation.
   void set_essential_bdr(mfem::Vector &residual) const override
   {
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // K the identity on the constrained dofs; also for a matrix of the
   // displacement only, e.g. the mass.
   void set_essential_bdr(mfem::HypreParMatrix &tangent) const override
   {
      tangent.EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
   }

   // K the identity on the constrained dofs, with the increment g of the
   // prescribed displacement on them moved to the right-hand side:
   //    rhs -= K g on the free dofs,   rhs = g on the constrained dofs.
   // EliminateRowsCols takes the constrained rows and columns out of K into
   // K_e, and EliminateBC moves K_e g to the right-hand side.
   void set_essential_bdr(mfem::HypreParMatrix &tangent,
                          const mfem::Vector &prescribed_increment,
                          mfem::Vector &rhs) const override
   {
      const std::unique_ptr<mfem::HypreParMatrix> eliminated(
         tangent.EliminateRowsCols(ess_tdof_list));
      tangent.EliminateBC(*eliminated, ess_tdof_list, prescribed_increment, rhs);
   }

   // The Dirichlet boundary conditions, whose values the nonlinear solver sets.
   const DirichletBoundary &get_dirichlet() const override { return *dirichlet; }

   // The Neumann boundary conditions.
   const NeumannBoundary &get_neumann() const override { return *neumann; }

private:
   // Declared before global_assembly, so that global_assembly, which
   // borrows it, goes first.
   const std::unique_ptr<LocalAssembly_Mixed> local_assembly;
   const std::unique_ptr<DirichletBoundary> dirichlet;    // constrained dofs and their values
   const std::unique_ptr<NeumannBoundary> neumann;        // tractions
   ParBlockNonlinearForm_External global_assembly;        // R + F_ext and K, without constraints
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent; // merged tangent, owned here
   mfem::ParLinearForm local_traction;                    // F_ext on the elements of this rank
   mfem::Vector external_force;                           // F_ext on the displacement dofs this rank owns
   const mfem::Array<int> ess_tdof_list;                  // constrained displacement dofs this rank owns
};

#endif

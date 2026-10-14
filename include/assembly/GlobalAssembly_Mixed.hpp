// ============================================================================
// GlobalAssembly_Mixed.hpp
//
// Parallel global assembly of the mixed displacement-pressure form, with the
// Dirichlet and the Neumann boundary conditions, on the dofs each rank owns:
//    external force  F_ext, from the tractions, on the displacement only;
//    residual        R(u,p) = F_int(u,p) - F_ext, from one parallel block
//                    form over LocalAssembly_Mixed;
//    tangent         K(u,p) = dR/d(u,p), one HypreParMatrix of both fields,
//                    assembled from the element tangents;
//    mass            M = int rho_0 N_a N_b dV of the displacement only, for
//                    the dynamics.
// set_essential_bdr sets R to zero and K to the identity on the constrained
// displacement dofs, and, with an increment of the prescribed displacement,
// also moves the constrained columns of K to the right-hand side. On each
// rank the displacement rows come first in K, so the displacement dofs keep
// their numbers.
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
   // conditions; the global_assembly only borrows the local assembly.
   GlobalAssembly_Mixed(mfem::ParFiniteElementSpace &space_u,
                        mfem::ParFiniteElementSpace &space_p,
                        std::unique_ptr<LocalAssembly_Mixed> input_local_assembly,
                        std::unique_ptr<DirichletBoundary> input_dirichlet,
                        std::unique_ptr<NeumannBoundary> input_neumann)
      : space(space_u),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        local_traction(&space_u),
        ess_tdof_list(dirichlet->get_ess_tdof_list()),
        external_force(space_u.GetTrueVSize()),
        local_assembly(std::move(input_local_assembly))
   {
      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(local_traction);

      mfem::Array<mfem::ParFiniteElementSpace *> spaces({&space_u, &space_p});
      global_assembly.SetParSpaces(spaces);
      global_assembly.AddDomainIntegrator(local_assembly.get());
   }

   // Number of unknowns this rank owns, displacement and pressure.
   int get_num_dofs() const { return global_assembly.Height(); }

   // The ranks of the displacement space.
   MPI_Comm get_comm() const { return space.GetComm(); }

   // Offsets of the displacement and the pressure blocks of the dofs this
   // rank owns, [0, n_u, n_u + n_p].
   const mfem::Array<int> &get_offsets() const
   {
      return global_assembly.GetBlockTrueOffsets();
   }

   // Update the traction coefficients and assemble F_ext on the
   // displacement dofs this rank owns; ParallelAssemble sums the shared
   // contributions onto the owning rank.
   void set_traction_load(double tt)
   {
      neumann->set_time(tt);
      local_traction.Assemble();
      local_traction.ParallelAssemble(external_force);
   }

   // R(u,p) at every dof this rank owns; the pressure has no external force.
   void assemble_residual(const mfem::Vector &sol, mfem::Vector &residual) const
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
   // next assemble_tangent call.
   mfem::HypreParMatrix &assemble_tangent(const mfem::Vector &sol) const
   {
      mfem::Array2D<const mfem::HypreParMatrix *> blocks = assemble_tangent_blocks(sol);
      tangent.reset(mfem::HypreParMatrixFromBlocks(blocks));
      return *tangent;
   }

   // M of the displacement on the dofs this rank owns, without
   // constraints; its size is n_u, not get_num_dofs(). The caller owns it.
   std::unique_ptr<mfem::HypreParMatrix> assemble_mass() const
   {
      mfem::ConstantCoefficient rho(local_assembly->get_rho_0());
      mfem::ParBilinearForm mass(&space);
      mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
      mass.Assemble();
      mass.Finalize();
      return std::unique_ptr<mfem::HypreParMatrix>(mass.ParallelAssemble());
   }

   // Zero the constrained residual entries, which carry no equation.
   void set_essential_bdr(mfem::Vector &residual) const
   {
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // Identity on the constrained dofs of a tangent or displacement mass
   // matrix.
   void set_essential_bdr(mfem::HypreParMatrix &tangent) const
   {
      tangent.EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
   }

   // Also move the prescribed increment through the eliminated columns to
   // the right-hand side: rhs -= K g on the free dofs, rhs = g on the
   // constrained ones. EliminateRowsCols returns the removed entries in K_e.
   void set_essential_bdr(mfem::HypreParMatrix &tangent,
                          const mfem::Vector &prescribed_increment,
                          mfem::Vector &rhs) const
   {
      const std::unique_ptr<mfem::HypreParMatrix> eliminated(
         tangent.EliminateRowsCols(ess_tdof_list));
      tangent.EliminateBC(*eliminated, ess_tdof_list, prescribed_increment, rhs);
   }

   const DirichletBoundary &get_dirichlet() const { return *dirichlet; }
   const NeumannBoundary &get_neumann() const { return *neumann; }

private:
   mfem::ParFiniteElementSpace &space;                    // the displacement space, outlives this
   const std::unique_ptr<DirichletBoundary> dirichlet;    // constrained displacement dofs and their values
   const std::unique_ptr<NeumannBoundary> neumann;        // tractions
   mfem::ParLinearForm local_traction;                    // destroyed before the tractions
   const mfem::Array<int> ess_tdof_list;                  // constrained displacement dofs this rank owns
   mfem::Vector external_force;                           // F_ext on the displacement dofs this rank owns

   // Declared before global_assembly, so that global_assembly, which
   // borrows it, goes first.
   const std::unique_ptr<LocalAssembly_Mixed> local_assembly;
   ParBlockNonlinearForm_External global_assembly;        // R + F_ext and K, without constraints
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent; // merged tangent, owned here
};

#endif

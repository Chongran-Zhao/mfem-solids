// ============================================================================
// GlobalAssembly.hpp
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
#ifndef GLOBAL_ASSEMBLY_HPP
#define GLOBAL_ASSEMBLY_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "DirichletBoundary.hpp"
#include "IGlobalAssembly.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "NeumannBoundary.hpp"
#include "ParBlockNonlinearForm_External.hpp"

class GlobalAssembly : public IGlobalAssembly
{
public:
   // Takes the ownership of the local assembly and of the boundary
   // conditions; the global_assembly only borrows the local assembly.
   GlobalAssembly(mfem::ParFiniteElementSpace &space_u,
                  mfem::ParFiniteElementSpace &space_p,
                  std::unique_ptr<LocalAssembly_Mixed> input_local_assembly,
                  std::unique_ptr<DirichletBoundary> input_dirichlet,
                  std::unique_ptr<NeumannBoundary> input_neumann)
      : IGlobalAssembly(space_u, std::move(input_dirichlet), std::move(input_neumann)),
        local_assembly(std::move(input_local_assembly))
   {
      mfem::Array<mfem::ParFiniteElementSpace *> spaces({&space_u, &space_p});
      global_assembly.SetParSpaces(spaces);
      global_assembly.AddDomainIntegrator(local_assembly.get());
   }

   // Number of unknowns this rank owns, displacement and pressure.
   int get_num_dofs() const override { return global_assembly.Height(); }

   // Offsets of the displacement and the pressure blocks of the dofs this
   // rank owns, [0, n_u, n_u + n_p].
   const mfem::Array<int> &get_offsets() const
   {
      return global_assembly.GetBlockTrueOffsets();
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

protected:
   double get_rho_0() const override { return local_assembly->get_rho_0(); }

private:
   // Declared before global_assembly, so that global_assembly, which
   // borrows it, goes first.
   const std::unique_ptr<LocalAssembly_Mixed> local_assembly;
   ParBlockNonlinearForm_External global_assembly;        // R + F_ext and K, without constraints
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent; // merged tangent, owned here
};

#endif

// ============================================================================
// GlobalAssembly_Disp.hpp
//
// Parallel global assembly of the displacement form, with the Dirichlet and
// the Neumann boundary conditions, on the dofs each rank owns:
//    external force  F_ext, from the tractions;
//    residual        R(d) = int N_a,J P_kJ dV - F_ext, from one
//                    ParNonlinearForm over LocalAssembly_Disp;
//    tangent         K(d) = dR/dd, a HypreParMatrix assembled from the
//                    element tangents, whose rows on each rank are the dofs
//                    it owns.
// set_essential_bdr sets R to zero and K to the identity on the constrained
// dofs, and, with an increment of the prescribed displacement, also moves
// the constrained columns of K to the right-hand side.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef GLOBAL_ASSEMBLY_DISP_HPP
#define GLOBAL_ASSEMBLY_DISP_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "DirichletBoundary.hpp"
#include "IGlobalAssembly.hpp"
#include "LocalAssembly_Disp.hpp"
#include "NeumannBoundary.hpp"

class GlobalAssembly_Disp : public IGlobalAssembly
{
public:
   // Takes the ownership of the local assembly and of the boundary
   // conditions; the global_assembly only borrows the local assembly.
   GlobalAssembly_Disp(mfem::ParFiniteElementSpace &space,
                       std::unique_ptr<LocalAssembly_Disp> input_local_assembly,
                       std::unique_ptr<DirichletBoundary> input_dirichlet,
                       std::unique_ptr<NeumannBoundary> input_neumann)
      : IGlobalAssembly(space, std::move(input_dirichlet), std::move(input_neumann)),
        local_assembly(std::move(input_local_assembly)),
        global_assembly(&space)
   {
      global_assembly.UseExternalIntegrators();
      global_assembly.AddDomainIntegrator(local_assembly.get());
   }

   // Number of unknowns this rank owns.
   int get_num_dofs() const override { return global_assembly.Height(); }

   // R(d) at every dof this rank owns.
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const override
   {
      global_assembly.Mult(disp, residual);
      residual -= external_force;
   }

   // K(d) on the dofs this rank owns. global_assembly owns it and builds it
   // anew at every call, so the caller may change it until the next one.
   mfem::HypreParMatrix &assemble_tangent(const mfem::Vector &disp) const override
   {
      return dynamic_cast<mfem::HypreParMatrix &>(global_assembly.GetGradient(disp));
   }

protected:
   double get_rho_0() const override { return local_assembly->get_rho_0(); }

private:
   // Declared before global_assembly, so that global_assembly, which
   // borrows it, goes first.
   const std::unique_ptr<LocalAssembly_Disp> local_assembly;
   mfem::ParNonlinearForm global_assembly;                // R + F_ext and K, without constraints
};

#endif

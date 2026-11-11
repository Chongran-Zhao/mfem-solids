// ============================================================================
// GlobalAssembly_Fracture.hpp
//
// Parallel global assembly of the phase-field problem of fracture at a fixed
// displacement, on the dofs each rank owns:
//    energy     E(alpha), the elastic energy degraded by g(alpha) plus the
//               surface energy, for the check of the alternate
//               minimization;
//    residual   R(alpha) = dE/dalpha, from one ParNonlinearForm over
//               the local assembly, LocalAssembly_Fracture or
//               LocalAssembly_Fracture_Regions;
//    tangent    K(alpha) = dR/dalpha, a HypreParMatrix assembled from the
//               element tangents, whose rows on each rank are the dofs it
//               owns.
// The constrained dofs, those where alpha = 0 (the crack set K_n of the
// irreversibility and the initial crack), change from one load step to the
// next and are set by set_ess_tdof_list; set_essential_bdr sets R to zero
// and K to the identity on them.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef GLOBAL_ASSEMBLY_FRACTURE_HPP
#define GLOBAL_ASSEMBLY_FRACTURE_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

class GlobalAssembly_Fracture
{
public:
   // Takes the ownership of the local assembly; the global_assembly only
   // borrows it.
   GlobalAssembly_Fracture(mfem::ParFiniteElementSpace &input_space,
                           std::unique_ptr<mfem::NonlinearFormIntegrator> input_local_assembly)
      : space(input_space),
        local_assembly(std::move(input_local_assembly)),
        global_assembly(&input_space)
   {
      global_assembly.UseExternalIntegrators();
      global_assembly.AddDomainIntegrator(local_assembly.get());
   }

   // Number of unknowns this rank owns.
   int get_num_dofs() const { return global_assembly.Height(); }

   // The ranks of the phase-field space.
   MPI_Comm get_comm() const { return space.GetComm(); }

   // The constrained dofs this rank owns, where alpha = 0.
   void set_ess_tdof_list(const mfem::Array<int> &input_ess_tdof_list)
   {
      ess_tdof_list = input_ess_tdof_list;
   }

   const mfem::Array<int> &get_ess_tdof_list() const { return ess_tdof_list; }

   // E(alpha) over all ranks, from the values of alpha on the dofs this rank
   // owns.
   double get_energy(const mfem::Vector &alpha) const { return global_assembly.GetEnergy(alpha); }

   // R(alpha) at every dof this rank owns.
   void assemble_residual(const mfem::Vector &alpha, mfem::Vector &residual) const
   {
      global_assembly.Mult(alpha, residual);
   }

   // K(alpha) on the dofs this rank owns. global_assembly owns it and builds
   // it anew at every call, so the caller may change it until the next one.
   mfem::HypreParMatrix &assemble_tangent(const mfem::Vector &alpha) const
   {
      return dynamic_cast<mfem::HypreParMatrix &>(global_assembly.GetGradient(alpha));
   }

   // Zero the constrained residual entries, which carry no equation.
   void set_essential_bdr(mfem::Vector &residual) const
   {
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // Identity on the constrained dofs of a tangent.
   void set_essential_bdr(mfem::HypreParMatrix &tangent) const
   {
      tangent.EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
   }

private:
   mfem::ParFiniteElementSpace &space;   // the phase-field space, outlives this
   mfem::Array<int> ess_tdof_list;       // constrained dofs this rank owns

   // Declared before global_assembly, so that global_assembly, which
   // borrows it, goes first.
   const std::unique_ptr<mfem::NonlinearFormIntegrator> local_assembly;
   mfem::ParNonlinearForm global_assembly;   // E, R and K, without constraints
};

#endif

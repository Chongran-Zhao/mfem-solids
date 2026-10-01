// ============================================================================
// GlobalAssembly_Disp.hpp
//
// Global assembly of the displacement form: the internal force R_int(d) and
// the stiffness K(d) of the whole mesh, from one NonlinearForm over
// LocalAssembly_Disp. Two views of them are given:
//    full        R_int and K as assembled, for the predictor and the
//                reactions;
//    eliminated  R_int zero and K the identity on the constrained dofs, for
//                NewtonSolver, through Mult and GetGradient.
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
#include "LocalAssembly_Disp.hpp"
#include "MaterialModel.hpp"

class GlobalAssembly_Disp : public mfem::Operator
{
public:
   // Takes the ownership of the local assembly; the form only borrows it.
   GlobalAssembly_Disp(mfem::FiniteElementSpace &space,
                       std::unique_ptr<LocalAssembly_Disp> input_local_assembly,
                       const mfem::Array<int> &input_ess_tdof_list)
      : mfem::Operator(space.GetTrueVSize()),
        local_assembly(std::move(input_local_assembly)),
        form(&space),
        ess_tdof_list(input_ess_tdof_list)
   {
      form.UseExternalIntegrators();
      form.AddDomainIntegrator(local_assembly.get());
   }

   // R_int(d), zero on the constrained dofs.
   void Mult(const mfem::Vector &disp, mfem::Vector &residual) const override
   {
      form.Mult(disp, residual);
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // K(d), the identity on the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &disp) const override
   {
      tangent = std::make_unique<mfem::SparseMatrix>(get_stiffness(disp));
      for (int dof : ess_tdof_list)
         tangent->EliminateRowCol(dof, mfem::Operator::DIAG_ONE);
      return *tangent;
   }

   // R_int(d) at every dof: the reaction on the constrained dofs, the load on
   // the others.
   void get_internal_force(const mfem::Vector &disp, mfem::Vector &force) const
   {
      form.Mult(disp, force);
   }

   // K(d) at every dof.
   const mfem::SparseMatrix &get_stiffness(const mfem::Vector &disp) const
   {
      return dynamic_cast<const mfem::SparseMatrix &>(form.GetGradient(disp));
   }

   // The material of the local assembly.
   const MaterialModel &get_material() const { return local_assembly->get_material(); }

   // The constrained dofs.
   const mfem::Array<int> &get_ess_tdof_list() const { return ess_tdof_list; }

private:
   // Declared before form, so that form, which borrows it, goes first.
   const std::unique_ptr<LocalAssembly_Disp> local_assembly;
   mfem::NonlinearForm form;                              // R_int and K, without constraints
   const mfem::Array<int> ess_tdof_list;                  // constrained dofs
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;   // K with the constraints eliminated
};

#endif

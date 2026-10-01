// ============================================================================
// GlobalAssembly_Disp.hpp
//
// Global assembly of the displacement form, with the boundary conditions of
// BoundaryManager:
//    external force  F_ext, from the tractions;
//    residual        R(d) = int N_a,J P_kJ dV - F_ext, from one NonlinearForm
//                    over LocalAssembly_Disp;
//    stiffness       K(d) = dR/dd.
// NewtonSolver gets R zero and K the identity on the constrained dofs, the
// tangent, through Mult and GetGradient; the consistent predictor gets the
// tangent with the constrained columns moved to the right-hand side, through
// get_predictor_system.
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
#include "BoundaryManager.hpp"
#include "LocalAssembly_Disp.hpp"
#include "MaterialModel.hpp"

class GlobalAssembly_Disp : public mfem::Operator
{
public:
   // Takes the ownership of the local assembly and of the boundary
   // conditions; the form only borrows the local assembly.
   GlobalAssembly_Disp(mfem::FiniteElementSpace &space,
                       std::unique_ptr<LocalAssembly_Disp> input_local_assembly,
                       std::unique_ptr<BoundaryManager> input_boundaries)
      : mfem::Operator(space.GetTrueVSize()),
        local_assembly(std::move(input_local_assembly)),
        boundaries(std::move(input_boundaries)),
        form(&space),
        external_force(&space),
        ess_tdof_list(boundaries->get_ess_tdof_list())
   {
      form.UseExternalIntegrators();
      form.AddDomainIntegrator(local_assembly.get());

      external_force = 0.0;
      if (boundaries->is_traction_load())
         boundaries->add_traction_integrators(external_force);
   }

   // Set the tractions to the given load step and assemble F_ext.
   void set_load_step(int step)
   {
      if (!boundaries->is_traction_load())
         return;
      boundaries->update_traction(step);
      external_force.Assemble();
   }

   // R(d), zero on the constrained dofs.
   void Mult(const mfem::Vector &disp, mfem::Vector &residual) const override
   {
      get_residual(disp, residual);
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // The tangent: K(d), the identity on the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &disp) const override
   {
      tangent = std::make_unique<mfem::SparseMatrix>(get_stiffness(disp));
      for (int dof : ess_tdof_list)
         tangent->EliminateRowCol(dof, mfem::Operator::DIAG_ONE);
      return *tangent;
   }

   // The linear system of the consistent predictor at d, with the increment
   // g of the prescribed displacement on the constrained dofs:
   //    tangent * du = rhs,   rhs = -R(d) - K(d) g on the free dofs,
   //                          rhs = g on the constrained dofs,
   // the constrained columns of K moved to the right-hand side.
   mfem::Operator &get_predictor_system(const mfem::Vector &disp,
                                        const mfem::Vector &prescribed_increment,
                                        mfem::Vector &rhs) const
   {
      get_residual(disp, rhs);
      rhs.Neg();
      tangent = std::make_unique<mfem::SparseMatrix>(get_stiffness(disp));
      for (int dof : ess_tdof_list)
         tangent->EliminateRowCol(dof, prescribed_increment(dof), rhs);
      return *tangent;
   }

   // Print the load of the current step.
   void print_load(int step, const mfem::GridFunction &disp) const
   {
      boundaries->print_load_by_step(step, disp, external_force);
   }

   // The material of the local assembly.
   const MaterialModel &get_material() const { return local_assembly->get_material(); }

   // The boundary conditions.
   BoundaryManager &get_boundaries() { return *boundaries; }

private:
   // R(d) at every dof.
   void get_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      form.Mult(disp, residual);
      residual -= external_force;
   }

   // K(d) at every dof.
   const mfem::SparseMatrix &get_stiffness(const mfem::Vector &disp) const
   {
      return dynamic_cast<const mfem::SparseMatrix &>(form.GetGradient(disp));
   }

   // Declared before form, so that form, which borrows it, goes first.
   const std::unique_ptr<LocalAssembly_Disp> local_assembly;
   const std::unique_ptr<BoundaryManager> boundaries;     // Dirichlet and Neumann
   mfem::NonlinearForm form;                              // R + F_ext and K, without constraints
   mfem::LinearForm external_force;                       // F_ext
   const mfem::Array<int> ess_tdof_list;                  // constrained dofs
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;   // K with the constraints eliminated
};

#endif

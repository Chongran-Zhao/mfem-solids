// ============================================================================
// GlobalAssembly_Disp.hpp
//
// Global assembly of the displacement form, with the Dirichlet and the
// Neumann boundary conditions:
//    external force  F_ext, from the tractions;
//    residual        R(d) = int N_a,J P_kJ dV - F_ext, from one NonlinearForm
//                    over LocalAssembly_Disp;
//    tangent         K(d) = dR/dd, assembled from the element tangents.
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
#include "LocalAssembly_Disp.hpp"
#include "NeumannBoundary.hpp"

class GlobalAssembly_Disp
{
public:
   // Takes the ownership of the local assembly and of the boundary
   // conditions; the global_assembly only borrows the local assembly.
   GlobalAssembly_Disp(mfem::FiniteElementSpace &space,
                       std::unique_ptr<LocalAssembly_Disp> input_local_assembly,
                       std::unique_ptr<DirichletBoundary> input_dirichlet,
                       std::unique_ptr<NeumannBoundary> input_neumann)
      : local_assembly(std::move(input_local_assembly)),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        global_assembly(&space),
        external_force(&space),
        ess_tdof_list(dirichlet->get_ess_tdof_list())
   {
      global_assembly.UseExternalIntegrators();
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

   // Number of unknowns.
   int get_num_dofs() const { return global_assembly.Height(); }

   // Consistent mass M_ab = int rho_0 N_a N_b dV, without constraints. It is
   // assembled once, for the dynamics, and the caller owns it.
   std::unique_ptr<mfem::SparseMatrix> assemble_mass(double density)
   {
      mfem::ConstantCoefficient rho(density);
      mfem::BilinearForm mass(global_assembly.FESpace());
      mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
      mass.Assemble();
      mass.Finalize();
      return std::unique_ptr<mfem::SparseMatrix>(mass.LoseMat());
   }

   // R(d) at every dof.
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      global_assembly.Mult(disp, residual);
      residual -= external_force;
   }

   // K(d) at every dof.
   const mfem::SparseMatrix &assemble_tangent(const mfem::Vector &disp) const
   {
      return dynamic_cast<const mfem::SparseMatrix &>(global_assembly.GetGradient(disp));
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
   const std::unique_ptr<LocalAssembly_Disp> local_assembly;
   const std::unique_ptr<DirichletBoundary> dirichlet;    // constrained dofs and their values
   const std::unique_ptr<NeumannBoundary> neumann;        // tractions
   mfem::NonlinearForm global_assembly;                   // R + F_ext and K, without constraints
   mfem::LinearForm external_force;                       // F_ext
   const mfem::Array<int> ess_tdof_list;                  // constrained dofs
};

#endif

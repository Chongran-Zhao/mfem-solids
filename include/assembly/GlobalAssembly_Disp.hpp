// ============================================================================
// GlobalAssembly_Disp.hpp
//
// Parallel global assembly of the displacement form, with the Dirichlet and
// the Neumann boundary conditions, on the true dofs of each rank:
//    external force  F_ext, from the tractions;
//    residual        R(d) = int N_a,J P_kJ dV - F_ext, from one
//                    ParNonlinearForm over LocalAssembly_Disp;
//    tangent         K(d) = dR/dd, a HypreParMatrix assembled from the
//                    element tangents.
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
   GlobalAssembly_Disp(mfem::ParFiniteElementSpace &space,
                       std::unique_ptr<LocalAssembly_Disp> input_local_assembly,
                       std::unique_ptr<DirichletBoundary> input_dirichlet,
                       std::unique_ptr<NeumannBoundary> input_neumann)
      : local_assembly(std::move(input_local_assembly)),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        global_assembly(&space),
        traction_form(&space),
        external_force(space.GetTrueVSize()),
        ess_tdof_list(dirichlet->get_ess_tdof_list())
   {
      global_assembly.UseExternalIntegrators();
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

   // Number of unknowns of this rank.
   int get_num_dofs() const { return global_assembly.Height(); }

   // The ranks of the space.
   MPI_Comm get_comm() const { return global_assembly.ParFESpace()->GetComm(); }

   // Consistent mass M_ab = int rho_0 N_a N_b dV, without constraints, with
   // rho_0 of the material. It is assembled once, for the dynamics, and the
   // caller owns it.
   std::unique_ptr<mfem::HypreParMatrix> assemble_mass()
   {
      mfem::ConstantCoefficient rho(local_assembly->get_rho_0());
      mfem::ParBilinearForm mass(global_assembly.ParFESpace());
      mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
      mass.Assemble();
      mass.Finalize();
      return std::unique_ptr<mfem::HypreParMatrix>(mass.ParallelAssemble());
   }

   // R(d) at every true dof.
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      global_assembly.Mult(disp, residual);
      residual -= external_force;
   }

   // K(d) at every true dof, copied from global_assembly, which owns its
   // gradient, into a new HypreParMatrix, which the caller owns.
   std::unique_ptr<mfem::HypreParMatrix> assemble_tangent(const mfem::Vector &disp) const
   {
      return std::make_unique<mfem::HypreParMatrix>(
         dynamic_cast<const mfem::HypreParMatrix &>(global_assembly.GetGradient(disp)));
   }

   // R zero on the constrained dofs, which carry no equation.
   void set_essential_bdr(mfem::Vector &residual) const
   {
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // K the identity on the constrained dofs.
   void set_essential_bdr(mfem::HypreParMatrix &tangent) const
   {
      tangent.EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
   }

   // K the identity on the constrained dofs, with the increment g of the
   // prescribed displacement on them moved to the right-hand side:
   //    rhs -= K g on the free dofs,   rhs = g on the constrained dofs.
   // The eliminated part K_e, K = K_e + the identity-bordered K, gives the
   // first.
   void set_essential_bdr(mfem::HypreParMatrix &tangent,
                          const mfem::Vector &prescribed_increment,
                          mfem::Vector &rhs) const
   {
      const std::unique_ptr<mfem::HypreParMatrix> eliminated(
         tangent.EliminateRowsCols(ess_tdof_list));
      tangent.EliminateBC(*eliminated, ess_tdof_list, prescribed_increment, rhs);
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
   mfem::ParNonlinearForm global_assembly;                // R + F_ext and K, without constraints
   mfem::ParLinearForm traction_form;                     // F_ext on the local dofs
   mfem::Vector external_force;                           // F_ext on the true dofs
   const mfem::Array<int> ess_tdof_list;                  // constrained true dofs
};

#endif

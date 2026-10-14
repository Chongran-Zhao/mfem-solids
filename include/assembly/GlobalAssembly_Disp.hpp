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
//                    it owns;
//    mass            M = int rho_0 N_a N_b dV, for the dynamics.
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
   GlobalAssembly_Disp(mfem::ParFiniteElementSpace &input_space,
                       std::unique_ptr<LocalAssembly_Disp> input_local_assembly,
                       std::unique_ptr<DirichletBoundary> input_dirichlet,
                       std::unique_ptr<NeumannBoundary> input_neumann)
      : space(input_space),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        local_traction(&input_space),
        ess_tdof_list(dirichlet->get_ess_tdof_list()),
        external_force(input_space.GetTrueVSize()),
        local_assembly(std::move(input_local_assembly)),
        global_assembly(&input_space)
   {
      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(local_traction);

      global_assembly.UseExternalIntegrators();
      global_assembly.AddDomainIntegrator(local_assembly.get());
   }

   // Number of unknowns this rank owns.
   int get_num_dofs() const { return global_assembly.Height(); }

   // The ranks of the displacement space.
   MPI_Comm get_comm() const { return space.GetComm(); }

   // Update the traction coefficients and assemble F_ext on the dofs this
   // rank owns; ParallelAssemble sums the shared contributions onto the
   // owning rank.
   void set_traction_load(double tt)
   {
      neumann->set_time(tt);
      local_traction.Assemble();
      local_traction.ParallelAssemble(external_force);
   }

   // R(d) at every dof this rank owns.
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      global_assembly.Mult(disp, residual);
      residual -= external_force;
   }

   // K(d) on the dofs this rank owns. global_assembly owns it and builds it
   // anew at every call, so the caller may change it until the next one.
   mfem::HypreParMatrix &assemble_tangent(const mfem::Vector &disp) const
   {
      return dynamic_cast<mfem::HypreParMatrix &>(global_assembly.GetGradient(disp));
   }

   // M on the dofs this rank owns, without constraints; the caller owns it.
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

   // Identity on the constrained dofs of a tangent or mass matrix.
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
   const std::unique_ptr<DirichletBoundary> dirichlet;    // constrained dofs and their values
   const std::unique_ptr<NeumannBoundary> neumann;        // tractions
   mfem::ParLinearForm local_traction;                    // destroyed before the tractions
   const mfem::Array<int> ess_tdof_list;                  // constrained dofs this rank owns
   mfem::Vector external_force;                           // F_ext on the dofs this rank owns

   // Declared before global_assembly, so that global_assembly, which
   // borrows it, goes first.
   const std::unique_ptr<LocalAssembly_Disp> local_assembly;
   mfem::ParNonlinearForm global_assembly;                // R + F_ext and K, without constraints
};

#endif

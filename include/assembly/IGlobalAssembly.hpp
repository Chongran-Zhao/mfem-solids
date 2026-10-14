// ============================================================================
// IGlobalAssembly.hpp
//
// Interface of the parallel global assemblies, GlobalAssembly_Disp and
// GlobalAssembly_Mixed, on the dofs each rank owns. Each gives its number
// of unknowns, its residual and its tangent, from its own MFEM form over
// its local assembly; this holds what they share, on the displacement
// space:
//    external force  F_ext, from the tractions of the Neumann boundary;
//    mass            M = int rho_0 N_a N_b dV, for the dynamics;
//    set_essential_bdr, which sets R to zero and K to the identity on the
//    constrained displacement dofs, and, with an increment of the
//    prescribed displacement, also moves the constrained columns of K to
//    the right-hand side; in the mixed form the displacement rows come first
//    in K on each rank, so the displacement dofs keep their numbers.
//
// Author: Chongran Zhao
// Date: Oct. 14, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef IGLOBAL_ASSEMBLY_HPP
#define IGLOBAL_ASSEMBLY_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "DirichletBoundary.hpp"
#include "ILocalAssembly.hpp"
#include "NeumannBoundary.hpp"

class IGlobalAssembly
{
public:
   virtual ~IGlobalAssembly() = default;

   // Number of unknowns this rank owns, of all fields.
   virtual int get_num_dofs() const = 0;

   // R at every dof this rank owns, F_int - F_ext.
   virtual void assemble_residual(const mfem::Vector &sol, mfem::Vector &residual) const = 0;

   // K = dR/dsol on the dofs this rank owns, owned by the assembly and
   // built anew at every call, so the caller may change it until the next
   // one.
   virtual mfem::HypreParMatrix &assemble_tangent(const mfem::Vector &sol) const = 0;

   // The ranks of the displacement space.
   MPI_Comm get_comm() const { return space.GetComm(); }

   // Update the traction coefficients and assemble F_ext on the
   // displacement dofs this rank owns; ParallelAssemble sums the shared
   // contributions onto the owning rank.
   void set_traction_load(double tt)
   {
      neumann->set_time(tt);
      local_traction.Assemble();
      local_traction.ParallelAssemble(external_force);
   }

   // M of the displacement on the dofs this rank owns, without
   // constraints; its size is that of the displacement, which in the mixed
   // form is less than get_num_dofs(). The caller owns it.
   std::unique_ptr<mfem::HypreParMatrix> assemble_mass() const
   {
      mfem::ConstantCoefficient rho(get_local_assembly().get_rho_0());
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

protected:
   // Takes the ownership of the boundary conditions; space is the
   // displacement space, which outlives the assembly.
   IGlobalAssembly(mfem::ParFiniteElementSpace &input_space,
                   std::unique_ptr<DirichletBoundary> input_dirichlet,
                   std::unique_ptr<NeumannBoundary> input_neumann)
      : space(input_space),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        local_traction(&input_space),
        ess_tdof_list(dirichlet->get_ess_tdof_list()),
        external_force(input_space.GetTrueVSize())
   {
      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(local_traction);
   }

   // The local assembly the derived class owns, for rho_0.
   virtual const ILocalAssembly &get_local_assembly() const = 0;

private:
   mfem::ParFiniteElementSpace &space;                    // the displacement space
   const std::unique_ptr<DirichletBoundary> dirichlet;    // constrained displacement dofs and their values
   const std::unique_ptr<NeumannBoundary> neumann;        // tractions
   mfem::ParLinearForm local_traction;                    // destroyed before the tractions
   const mfem::Array<int> ess_tdof_list;                  // constrained displacement dofs this rank owns

protected:
   mfem::Vector external_force;                           // F_ext on the displacement dofs this rank owns
};

#endif

// ============================================================================
// IGlobalAssembly.hpp
//
// Common parallel assembly operations on the true dofs this rank owns:
// reference mass and external force on the displacement space, boundary
// elimination, and access to the boundary conditions. Derived classes
// assemble their residual and tangent with their own MFEM nonlinear form.
// ============================================================================
#ifndef IGLOBAL_ASSEMBLY_HPP
#define IGLOBAL_ASSEMBLY_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "DirichletBoundary.hpp"
#include "NeumannBoundary.hpp"

class IGlobalAssembly
{
public:
   virtual ~IGlobalAssembly() = default;

   // Number of all unknowns this rank owns, including any pressure field.
   virtual int get_num_dofs() const = 0;

   // The ranks of the displacement space and of the assembly.
   virtual MPI_Comm get_comm() const { return displacement_space.GetComm(); }

   // Update the traction coefficients and assemble F_ext on displacement
   // dofs. ParallelAssemble sums shared contributions onto the owning rank.
   virtual void set_traction_load(double tt)
   {
      neumann->set_time(tt);
      local_traction.Assemble();
      local_traction.ParallelAssemble(external_force);
   }

   // Unconstrained residual F_int - F_ext on all unknowns.
   virtual void assemble_residual(const mfem::Vector &sol,
                                  mfem::Vector &residual) const = 0;

   // Unconstrained tangent on all unknowns. The assembly owns the matrix;
   // the caller may modify it until the next assemble_tangent call or
   // destruction of the assembly.
   virtual mfem::HypreParMatrix &assemble_tangent(const mfem::Vector &sol) const = 0;

   // Unconstrained reference mass of the displacement field only;
   // its size may be smaller than get_num_dofs(). The caller owns it.
   virtual std::unique_ptr<mfem::HypreParMatrix> assemble_mass()
   {
      mfem::ConstantCoefficient rho(get_rho_0());
      mfem::ParBilinearForm mass(&displacement_space);
      mass.AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
      mass.Assemble();
      mass.Finalize();
      return std::unique_ptr<mfem::HypreParMatrix>(mass.ParallelAssemble());
   }

   // Zero the constrained residual entries, which carry no equation.
   virtual void set_essential_bdr(mfem::Vector &residual) const
   {
      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // Identity on constrained dofs of a tangent or displacement mass matrix.
   virtual void set_essential_bdr(mfem::HypreParMatrix &tangent) const
   {
      tangent.EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
   }

   // Also move the prescribed increment through the eliminated columns
   // to the right-hand side: rhs -= K g on free dofs, rhs = g on constrained
   // dofs. EliminateRowsCols returns the removed entries in K_e.
   virtual void set_essential_bdr(mfem::HypreParMatrix &tangent,
                                  const mfem::Vector &prescribed_increment,
                                  mfem::Vector &rhs) const
   {
      const std::unique_ptr<mfem::HypreParMatrix> eliminated(
         tangent.EliminateRowsCols(ess_tdof_list));
      tangent.EliminateBC(*eliminated, ess_tdof_list, prescribed_increment, rhs);
   }

   virtual const DirichletBoundary &get_dirichlet() const { return *dirichlet; }
   virtual const NeumannBoundary &get_neumann() const { return *neumann; }

protected:
   // Takes ownership of the boundaries. The displacement space outlives
   // the assembly; the traction form is destroyed before its coefficients.
   IGlobalAssembly(mfem::ParFiniteElementSpace &space,
                   std::unique_ptr<DirichletBoundary> input_dirichlet,
                   std::unique_ptr<NeumannBoundary> input_neumann)
      : displacement_space(space),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        local_traction(&space),
        ess_tdof_list(dirichlet->get_ess_tdof_list()),
        external_force(space.GetTrueVSize())
   {
      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(local_traction);
   }

   // Read the density from the concrete local assembly without copying it.
   virtual double get_rho_0() const = 0;

private:
   mfem::ParFiniteElementSpace &displacement_space;
   const std::unique_ptr<DirichletBoundary> dirichlet;
   const std::unique_ptr<NeumannBoundary> neumann;
   mfem::ParLinearForm local_traction;
   const mfem::Array<int> ess_tdof_list;

protected:
   // Derived residual assembly subtracts it from the displacement entries.
   mfem::Vector external_force;
};

#endif

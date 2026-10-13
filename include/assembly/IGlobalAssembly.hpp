// ============================================================================
// IGlobalAssembly.hpp
//
// Interface of parallel global assembly on the true dofs this rank owns.
// Displacement and mixed forms provide the same assembly and boundary
// operations; block-specific operations belong to the mixed implementation.
// ============================================================================
#ifndef IGLOBAL_ASSEMBLY_HPP
#define IGLOBAL_ASSEMBLY_HPP

#include <memory>

#include <mfem.hpp>

class DirichletBoundary;
class NeumannBoundary;

class IGlobalAssembly
{
public:
   virtual ~IGlobalAssembly() = default;

   // Number of all unknowns this rank owns, and the ranks of the spaces.
   virtual int get_num_dofs() const = 0;
   virtual MPI_Comm get_comm() const = 0;

   // Update the traction coefficients and assemble the external force.
   virtual void set_traction_load(double tt) = 0;

   // Unconstrained residual F_int - F_ext on all unknowns.
   virtual void assemble_residual(const mfem::Vector &sol,
                                  mfem::Vector &residual) const = 0;

   // Unconstrained tangent on all unknowns. The assembly owns the matrix;
   // the caller may modify it until the next assemble_tangent call or
   // destruction of the assembly.
   virtual mfem::HypreParMatrix &assemble_tangent(const mfem::Vector &sol) const = 0;

   // Unconstrained reference mass of the displacement field only;
   // its size may be smaller than get_num_dofs(). The caller owns it.
   virtual std::unique_ptr<mfem::HypreParMatrix> assemble_mass() = 0;

   // Zero the constrained residual entries, or eliminate constrained rows
   // and columns of a tangent or displacement mass matrix.
   virtual void set_essential_bdr(mfem::Vector &residual) const = 0;
   virtual void set_essential_bdr(mfem::HypreParMatrix &tangent) const = 0;

   // Also move the prescribed increment through the eliminated columns
   // to the right-hand side.
   virtual void set_essential_bdr(mfem::HypreParMatrix &tangent,
                                  const mfem::Vector &prescribed_increment,
                                  mfem::Vector &rhs) const = 0;

   virtual const DirichletBoundary &get_dirichlet() const = 0;
   virtual const NeumannBoundary &get_neumann() const = 0;
};

#endif

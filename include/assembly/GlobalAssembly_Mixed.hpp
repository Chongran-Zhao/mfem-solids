// ============================================================================
// GlobalAssembly_Mixed.hpp
//
// Global residual and tangent of the mixed displacement-pressure form.
// Owns the material, boundary conditions and two block nonlinear forms: one
// with essential displacement dofs for Newton, one unconstrained for the
// consistent predictor. The forms own their local integrators, which borrow
// the material; the finite element spaces are borrowed from the caller.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef GLOBAL_ASSEMBLY_MIXED_HPP
#define GLOBAL_ASSEMBLY_MIXED_HPP

#include <memory>
#include <utility>
#include <mfem.hpp>
#include "DirichletBoundary.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "MaterialModel.hpp"
#include "NeumannBoundary.hpp"

class GlobalAssembly_Mixed
{
public:
   GlobalAssembly_Mixed(mfem::FiniteElementSpace &space_u,
                        mfem::FiniteElementSpace &space_p,
                        std::unique_ptr<const MaterialModel> input_material,
                        std::unique_ptr<DirichletBoundary> input_dirichlet,
                        std::unique_ptr<NeumannBoundary> input_neumann)
      : material(std::move(input_material)),
        dirichlet(std::move(input_dirichlet)),
        neumann(std::move(input_neumann)),
        spaces({&space_u, &space_p}),
        offsets({0, space_u.GetTrueVSize(),
                    space_u.GetTrueVSize() + space_p.GetTrueVSize()}),
        global_assembly(spaces), internal_force_form(spaces),
        external_force(&space_u), ess_u(dirichlet->get_ess_tdof_list())
   {
      global_assembly.AddDomainIntegrator(new LocalAssembly_Mixed(*material));
      internal_force_form.AddDomainIntegrator(new LocalAssembly_Mixed(*material));
      mfem::Array<int> ess_p;
      mfem::Array<mfem::Array<int> *> ess({&ess_u, &ess_p});
      mfem::Array<mfem::Vector *> ess_rhs({nullptr, nullptr});
      global_assembly.SetEssentialTrueDofs(ess, ess_rhs);

      external_force = 0.0;
      if (neumann->is_traction_load())
         neumann->add_traction_integrators(external_force);
   }

   int get_num_dofs() const { return offsets.Last(); }
   const mfem::Array<int> &get_offsets() const { return offsets; }
   mfem::FiniteElementSpace &get_disp_space() const { return *spaces[0]; }
   const MaterialModel &get_material() const { return *material; }
   const DirichletBoundary &get_dirichlet() const { return *dirichlet; }
   const NeumannBoundary &get_neumann() const { return *neumann; }

   void set_traction_load(double tt)
   {
      neumann->set_time(tt);
      external_force.Assemble();
   }

   // Full internal residual and tangent, including the constrained columns.
   void set_internal_force(const mfem::Vector &sol, mfem::Vector &residual) const
   {
      internal_force_form.Mult(sol, residual);
   }
   mfem::Operator &get_internal_tangent(const mfem::Vector &sol) const
   {
      return internal_force_form.GetGradient(sol);
   }

   // External force acts on displacement only. Newton uses zero at essential
   // dofs; the predictor needs the full force before imposing its increment.
   void set_external_force(mfem::BlockVector &rhs, bool constrained) const
   {
      rhs = 0.0;
      rhs.GetBlock(0) = external_force;
      if (constrained)
         rhs.GetBlock(0).SetSubVector(ess_u, 0.0);
   }

   void set_residual(const mfem::Vector &sol, mfem::Vector &residual) const
   {
      global_assembly.Mult(sol, residual);
   }
   mfem::Operator &get_tangent(const mfem::Vector &sol) const
   {
      return global_assembly.GetGradient(sol);
   }

   // The constrained tangent has an identity diagonal on essential dofs.
   void set_essential_bdr(const mfem::BlockVector &increment,
                          mfem::BlockVector &rhs) const
   {
      for (int dof : ess_u)
         rhs.GetBlock(0)(dof) = increment.GetBlock(0)(dof);
   }

private:
   // Material and boundary data outlive the forms and traction integrators.
   const std::unique_ptr<const MaterialModel> material;
   const std::unique_ptr<DirichletBoundary> dirichlet;
   const std::unique_ptr<NeumannBoundary> neumann;
   mfem::Array<mfem::FiniteElementSpace *> spaces;
   const mfem::Array<int> offsets;
   mfem::BlockNonlinearForm global_assembly;
   mfem::BlockNonlinearForm internal_force_form;
   mfem::LinearForm external_force;
   mfem::Array<int> ess_u;
};

#endif

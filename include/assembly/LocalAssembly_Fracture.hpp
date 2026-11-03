// ============================================================================
// LocalAssembly_Fracture.hpp
//
// Local (element) assembly of phase-field fracture in the displacement form,
// Total Lagrangian, for alternate minimization of
//    E(u, alpha) = int a(alpha) W(F) + C1 w(alpha) + C2/2 |Grad alpha|^2 dV,
// with C1 = G_c / (4 c_w ell) and C2 = G_c ell / (2 c_w). It joins a material,
// which gives W, P and AA, and a damage model, which gives a, w and c_w, and
// assembles two problems without coupling, as the operators and bilinear
// forms of the displacement and of the damage in mef90:
//    at fixed alpha, R^a_k = int a(alpha) N_a,J P_kJ dV,
//                    K^ab_kl = int a(alpha) N_a,J AA_kJlL N_b,L dV;
//    at fixed u,     R^a = int (a'(alpha) W + C1 w'(alpha)) N_a
//                              + C2 Grad alpha . N_a,J dV,
//                    K^ab = int (a''(alpha) W + C1 w''(alpha)) N_a N_b
//                               + C2 N_a,J N_b,J dV.
// Each problem reads the other field from its grid function, which the caller
// owns and must keep up to date. Both take the quadrature rule of the
// displacement element, as mef90. MFEM's NonlinearForm calls fixed function
// names, so two small integrators forward to the functions of each problem;
// this class owns them, and each NonlinearForm borrows its own through
// UseExternalIntegrators.
//
// Author: Chongran Zhao
// Date: Nov. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_FRACTURE_HPP
#define LOCAL_ASSEMBLY_FRACTURE_HPP

#include <memory>
#include <utility>
#include <vector>

#include <mfem.hpp>

#include "IDamageModel.hpp"
#include "LocalAssemblyTools.hpp"
#include "MaterialModel.hpp"
#include "ShapeCache.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class LocalAssembly_Fracture
{
public:
   // Takes the ownership of the material and of the damage model; disp and
   // alpha stay the caller's.
   LocalAssembly_Fracture(std::unique_ptr<const MaterialModel> input_material,
                          std::unique_ptr<const IDamageModel> input_damage_model,
                          const mfem::GridFunction &input_disp,
                          const mfem::GridFunction &input_alpha)
      : material(std::move(input_material)),
        damage_model(std::move(input_damage_model)),
        C1(damage_model->get_G_c() / (4.0 * damage_model->get_c_w() * damage_model->get_ell())),
        C2(damage_model->get_G_c() * damage_model->get_ell() / (2.0 * damage_model->get_c_w())),
        disp(input_disp),
        alpha(input_alpha),
        disp_integrator(*this),
        damage_integrator(*this) {}

   // The integrators to give to the NonlinearForm of each problem.
   mfem::NonlinearFormIntegrator &get_disp_integrator() { return disp_integrator; }
   mfem::NonlinearFormIntegrator &get_damage_integrator() { return damage_integrator; }

   // Quadrature order of the elements; 0, the default, keeps the rule of
   // LocalAssemblyTools::get_quad_rule on the displacement element.
   void set_quad_order(int input_quad_order) { quad_order = input_quad_order; }

   // Reference density rho_0 of the material, for the mass matrix.
   double get_rho_0() const { return material->get_rho_0(); }

   // Elastic energy int a(alpha) W(F) dV of the element, with disp_elem its
   // displacement.
   double get_elastic_energy(const mfem::FiniteElement &elem_u,
                             mfem::ElementTransformation &elem_map,
                             const mfem::Vector &disp_elem)
   {
      const mfem::Vector alpha_elem = get_elem_values(alpha, elem_map.ElementNo);
      const mfem::FiniteElement &elem_alpha = *alpha.FESpace()->GetFE(elem_map.ElementNo);

      double out = 0.0;
      for (const ShapeCache::PointShapes &point :
           disp_shapes.get(elem_u, &elem_alpha, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp_elem, point.dN_dX);
         out += point.dV * damage_model->get_a(point.M * alpha_elem)
                * material->get_strain_energy(F);
      }
      return out;
   }

   // Surface energy int C1 w(alpha) + C2/2 |Grad alpha|^2 dV of the element,
   // with alpha_elem its damage.
   double get_surface_energy(const mfem::FiniteElement &elem_alpha,
                             mfem::ElementTransformation &elem_map,
                             const mfem::Vector &alpha_elem)
   {
      const mfem::FiniteElement &elem_u = *disp.FESpace()->GetFE(elem_map.ElementNo);

      double out = 0.0;
      mfem::Vector grad_alpha(3);
      for (const ShapeCache::PointShapes &point :
           damage_shapes.get(elem_alpha, &elem_u, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         point.dN_dX.MultTranspose(alpha_elem, grad_alpha);
         out += point.dV * (C1 * damage_model->get_w(point.N * alpha_elem)
                            + 0.5 * C2 * (grad_alpha * grad_alpha));
      }
      return out;
   }

   // R^a_k = int a(alpha) N_a,J P_kJ dV, stored at aa + k * num_nodes.
   void assemble_disp_residual(const mfem::FiniteElement &elem_u,
                               mfem::ElementTransformation &elem_map,
                               const mfem::Vector &disp_elem,
                               mfem::Vector &residual)
   {
      const int num_nodes = elem_u.GetDof();
      residual.SetSize(3 * num_nodes);
      residual = 0.0;

      const mfem::Vector alpha_elem = get_elem_values(alpha, elem_map.ElementNo);
      const mfem::FiniteElement &elem_alpha = *alpha.FESpace()->GetFE(elem_map.ElementNo);

      for (const ShapeCache::PointShapes &point :
           disp_shapes.get(elem_u, &elem_alpha, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         const mfem::DenseMatrix &dN_dX = point.dN_dX;
         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp_elem, dN_dX);
         const Tensor2_3D PK1 = material->get_1st_PK_stress(F);
         const double a_dV = damage_model->get_a(point.M * alpha_elem) * point.dV;

         for (int aa = 0; aa < num_nodes; aa++)
            for (int kk = 0; kk < 3; kk++)
               residual(aa+kk*num_nodes) += a_dV * (PK1(kk, 0) * dN_dX(aa, 0)
                                                    + PK1(kk, 1) * dN_dX(aa, 1)
                                                    + PK1(kk, 2) * dN_dX(aa, 2));
      }
   }

   // K^ab_kl = int a(alpha) N_a,J AA_kJlL N_b,L dV, stored at
   // tangent(aa+k*num_nodes, bb+l*num_nodes).
   void assemble_disp_tangent(const mfem::FiniteElement &elem_u,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Vector &disp_elem,
                              mfem::DenseMatrix &tangent)
   {
      tangent.SetSize(3 * elem_u.GetDof());
      tangent = 0.0;

      const mfem::Vector alpha_elem = get_elem_values(alpha, elem_map.ElementNo);
      const mfem::FiniteElement &elem_alpha = *alpha.FESpace()->GetFE(elem_map.ElementNo);

      for (const ShapeCache::PointShapes &point :
           disp_shapes.get(elem_u, &elem_alpha, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp_elem, point.dN_dX);
         const Tensor4_3D AA = material->get_1st_elasticity_tensor(F);
         const double a_dV = damage_model->get_a(point.M * alpha_elem) * point.dV;

         LocalAssemblyTools::add_momentum_tangent_uu(point.dN_dX, AA, a_dV, tangent);
      }
   }

   // R^a = int (a'(alpha) W + C1 w'(alpha)) N_a + C2 Grad alpha . N_a,J dV.
   void assemble_damage_residual(const mfem::FiniteElement &elem_alpha,
                                 mfem::ElementTransformation &elem_map,
                                 const mfem::Vector &alpha_elem,
                                 mfem::Vector &residual)
   {
      const int num_nodes = elem_alpha.GetDof();
      residual.SetSize(num_nodes);
      residual = 0.0;

      const mfem::Vector disp_elem = get_elem_values(disp, elem_map.ElementNo);
      const mfem::FiniteElement &elem_u = *disp.FESpace()->GetFE(elem_map.ElementNo);

      mfem::Vector grad_alpha(3);
      for (const ShapeCache::PointShapes &point :
           damage_shapes.get(elem_alpha, &elem_u, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         const mfem::DenseMatrix &dN_dX = point.dN_dX;
         const double alpha_pt = point.N * alpha_elem;
         dN_dX.MultTranspose(alpha_elem, grad_alpha);

         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp_elem, point.dM_dX);
         const double C3 = damage_model->get_da(alpha_pt) * material->get_strain_energy(F)
                           + C1 * damage_model->get_dw(alpha_pt);

         for (int aa = 0; aa < num_nodes; aa++)
            residual(aa) += point.dV * (C3 * point.N(aa)
                                        + C2 * (dN_dX(aa, 0) * grad_alpha(0)
                                                + dN_dX(aa, 1) * grad_alpha(1)
                                                + dN_dX(aa, 2) * grad_alpha(2)));
      }
   }

   // K^ab = int (a''(alpha) W + C1 w''(alpha)) N_a N_b + C2 N_a,J N_b,J dV.
   void assemble_damage_tangent(const mfem::FiniteElement &elem_alpha,
                                mfem::ElementTransformation &elem_map,
                                const mfem::Vector &alpha_elem,
                                mfem::DenseMatrix &tangent)
   {
      const int num_nodes = elem_alpha.GetDof();
      tangent.SetSize(num_nodes);
      tangent = 0.0;

      const mfem::Vector disp_elem = get_elem_values(disp, elem_map.ElementNo);
      const mfem::FiniteElement &elem_u = *disp.FESpace()->GetFE(elem_map.ElementNo);

      for (const ShapeCache::PointShapes &point :
           damage_shapes.get(elem_alpha, &elem_u, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         const mfem::DenseMatrix &dN_dX = point.dN_dX;
         const double alpha_pt = point.N * alpha_elem;

         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp_elem, point.dM_dX);
         const double C3 = damage_model->get_d2a(alpha_pt) * material->get_strain_energy(F)
                           + C1 * damage_model->get_d2w(alpha_pt);

         for (int bb = 0; bb < num_nodes; bb++)
            for (int aa = 0; aa < num_nodes; aa++)
               tangent(aa, bb) += point.dV * (C3 * point.N(aa) * point.N(bb)
                                              + C2 * (dN_dX(aa, 0) * dN_dX(bb, 0)
                                                      + dN_dX(aa, 1) * dN_dX(bb, 1)
                                                      + dN_dX(aa, 2) * dN_dX(bb, 2)));
      }
   }

private:
   // The integrator of the displacement problem: energy, residual and tangent
   // at fixed alpha.
   class DispIntegrator : public mfem::NonlinearFormIntegrator
   {
   public:
      explicit DispIntegrator(LocalAssembly_Fracture &input_owner) : owner(input_owner) {}

      double GetElementEnergy(const mfem::FiniteElement &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Vector &elem_sol) override
      {
         return owner.get_elastic_energy(elem, elem_map, elem_sol);
      }

      void AssembleElementVector(const mfem::FiniteElement &elem,
                                 mfem::ElementTransformation &elem_map,
                                 const mfem::Vector &elem_sol,
                                 mfem::Vector &residual) override
      {
         owner.assemble_disp_residual(elem, elem_map, elem_sol, residual);
      }

      void AssembleElementGrad(const mfem::FiniteElement &elem,
                               mfem::ElementTransformation &elem_map,
                               const mfem::Vector &elem_sol,
                               mfem::DenseMatrix &tangent) override
      {
         owner.assemble_disp_tangent(elem, elem_map, elem_sol, tangent);
      }

   private:
      LocalAssembly_Fracture &owner;
   };

   // The integrator of the damage problem: energy, residual and tangent at
   // fixed u.
   class DamageIntegrator : public mfem::NonlinearFormIntegrator
   {
   public:
      explicit DamageIntegrator(LocalAssembly_Fracture &input_owner) : owner(input_owner) {}

      double GetElementEnergy(const mfem::FiniteElement &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Vector &elem_sol) override
      {
         return owner.get_damage_energy(elem, elem_map, elem_sol);
      }

      void AssembleElementVector(const mfem::FiniteElement &elem,
                                 mfem::ElementTransformation &elem_map,
                                 const mfem::Vector &elem_sol,
                                 mfem::Vector &residual) override
      {
         owner.assemble_damage_residual(elem, elem_map, elem_sol, residual);
      }

      void AssembleElementGrad(const mfem::FiniteElement &elem,
                               mfem::ElementTransformation &elem_map,
                               const mfem::Vector &elem_sol,
                               mfem::DenseMatrix &tangent) override
      {
         owner.assemble_damage_tangent(elem, elem_map, elem_sol, tangent);
      }

   private:
      LocalAssembly_Fracture &owner;
   };

   // Energy of the damage problem at fixed u, elastic and surface energies
   // together, with alpha_elem the damage of the element.
   double get_damage_energy(const mfem::FiniteElement &elem_alpha,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Vector &alpha_elem)
   {
      const mfem::Vector disp_elem = get_elem_values(disp, elem_map.ElementNo);
      const mfem::FiniteElement &elem_u = *disp.FESpace()->GetFE(elem_map.ElementNo);

      double out = get_surface_energy(elem_alpha, elem_map, alpha_elem);
      for (const ShapeCache::PointShapes &point :
           damage_shapes.get(elem_alpha, &elem_u, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp_elem, point.dM_dX);
         out += point.dV * damage_model->get_a(point.N * alpha_elem)
                * material->get_strain_energy(F);
      }
      return out;
   }

   // The values of a field on an element, in the order of its element dofs.
   static mfem::Vector get_elem_values(const mfem::GridFunction &field, int element)
   {
      mfem::Array<int> vdofs;
      field.FESpace()->GetElementVDofs(element, vdofs);
      mfem::Vector out;
      field.GetSubVector(vdofs, out);
      return out;
   }

   // The rule of order quad_order, or that of LocalAssemblyTools on the
   // displacement element.
   const mfem::IntegrationRule &get_quad_rule(const mfem::FiniteElement &elem_u,
                                              mfem::ElementTransformation &elem_map) const
   {
      return (quad_order > 0) ? mfem::IntRules.Get(elem_u.GetGeomType(), quad_order)
                              : LocalAssemblyTools::get_quad_rule(elem_u, elem_map);
   }

   const std::unique_ptr<const MaterialModel> material;     // W, P and AA
   const std::unique_ptr<const IDamageModel> damage_model;  // a, w and c_w
   const double C1;                                         // G_c / (4 c_w ell)
   const double C2;                                         // G_c ell / (2 c_w)
   const mfem::GridFunction &disp;                          // u, of the damage problem
   const mfem::GridFunction &alpha;                         // alpha, of the displacement problem
   int quad_order = 0;                                      // 0 for that of LocalAssemblyTools
   ShapeCache disp_shapes{false};                           // N of u, and M = N of alpha
   ShapeCache damage_shapes{false};                         // N of alpha, and M = N of u
   DispIntegrator disp_integrator;                          // given to the form of u
   DamageIntegrator damage_integrator;                      // given to the form of alpha
};

#endif

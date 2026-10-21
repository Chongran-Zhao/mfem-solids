// ============================================================================
// LocalAssembly_Disp.hpp
//
// Local (element) assembly of compressible hyperelasticity in the Total
// Lagrangian form: the element residual and tangent. The unknown is the
// displacement. It is an MFEM NonlinearFormIntegrator, called by
// NonlinearForm, which assembles the global residual and tangent.
//
// Author: Chongran Zhao
// Date: Sep. 25, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_DISP_HPP
#define LOCAL_ASSEMBLY_DISP_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "LocalAssemblyTools.hpp"
#include "MaterialModel.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class LocalAssembly_Disp : public mfem::NonlinearFormIntegrator
{
public:
   // Takes the ownership of the material.
   LocalAssembly_Disp(std::unique_ptr<const MaterialModel> input_material)
      : material(std::move(input_material)) {}

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // AssembleElementVector, which NonlinearForm calls on every element.
   // a is the node index and k the direction (x, y, z).
   // R^a_k = int N_a,J P_kJ dV over the element in the reference configuration.
   // disp and residual store component k of node aa at aa + k * num_nodes.
   void AssembleElementVector(const mfem::FiniteElement &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Vector &disp,
                              mfem::Vector &residual) override
   {
      const int num_nodes = elem.GetDof();
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);

      residual.SetSize(3 * num_nodes);
      residual = 0.0;

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);

         // dN/dxi
         elem.CalcDShape(quad_pt, dN_dxi);
         // N_a,J = dN/dX = dN/dxi (dX/dxi)^-1.
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);

         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp, dN_dX);

         const Tensor2_3D PK1 = material->get_1st_PK_stress(F);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         for (int aa = 0; aa < num_nodes; aa++)
         {
            // x-component
            residual(aa)             += dV * (PK1(0, 0) * dN_dX(aa, 0)
                                            + PK1(0, 1) * dN_dX(aa, 1)
                                            + PK1(0, 2) * dN_dX(aa, 2));
            // y-component
            residual(aa+num_nodes)   += dV * (PK1(1, 0) * dN_dX(aa, 0)
                                            + PK1(1, 1) * dN_dX(aa, 1)
                                            + PK1(1, 2) * dN_dX(aa, 2));
            // z-component
            residual(aa+2*num_nodes) += dV * (PK1(2, 0) * dN_dX(aa, 0)
                                            + PK1(2, 1) * dN_dX(aa, 1)
                                            + PK1(2, 2) * dN_dX(aa, 2));
         }
      }
   }

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // AssembleElementGrad, which NonlinearForm calls on every element.
   // K^ab_kl = int N_a,J AA_kJlL N_b,L dV, with AA = dP/dF,
   // stored at tangent(aa+k*num_nodes, bb+l*num_nodes).
   void AssembleElementGrad(const mfem::FiniteElement &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Vector &disp,
                            mfem::DenseMatrix &tangent) override
   {
      const int num_nodes = elem.GetDof();
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);

      tangent.SetSize(3*num_nodes);
      tangent = 0.0;

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);

         // dN/dxi
         elem.CalcDShape(quad_pt, dN_dxi);
         // N_a,J = dN/dX = dN/dxi (dX/dxi)^-1.
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);

         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp, dN_dX);

         const Tensor4_3D AA = material->get_1st_elasticity_tensor(F);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         LocalAssemblyTools::add_momentum_tangent_uu(dN_dX, AA, dV, tangent);
      }
   }

   // Reference density rho_0 of the material, for the mass matrix.
   double get_rho_0() const { return material->get_rho_0(); }

   // Quadrature order of the elements; 0, the default, keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   void set_quad_order(int input_quad_order) { quad_order = input_quad_order; }

private:
   // The rule of order quad_order, or that of LocalAssemblyTools.
   const mfem::IntegrationRule &get_quad_rule(const mfem::FiniteElement &elem,
                                              mfem::ElementTransformation &elem_map) const
   {
      return (quad_order > 0) ? mfem::IntRules.Get(elem.GetGeomType(), quad_order)
                              : LocalAssemblyTools::get_quad_rule(elem, elem_map);
   }

   const std::unique_ptr<const MaterialModel> material;
   int quad_order = 0;   // quadrature order, 0 for that of LocalAssemblyTools
};

#endif

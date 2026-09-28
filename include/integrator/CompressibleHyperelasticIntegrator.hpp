// ============================================================================
// CompressibleHyperelasticIntegrator.hpp
//
// Element residual and tangent of compressible hyperelasticity in the
// Total Lagrangian form. The unknown is the displacement.
//
// Author: Chongran Zhao
// Date: Sep. 25, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef COMPRESSIBLE_HYPERELASTIC_INTEGRATOR_HPP
#define COMPRESSIBLE_HYPERELASTIC_INTEGRATOR_HPP

#include "HyperelasticMaterialModel.hpp"
#include "Tensor2_3D.hpp"
#include "mfem.hpp"

class CompressibleHyperelasticIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   CompressibleHyperelasticIntegrator(const HyperelasticMaterialModel &input_material)
      : material(input_material) {}

   // a is the nodal point index, k is the dimension index (x, y, z)
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

         const Tensor2_3D F = get_deformation_gradient(disp, dN_dX);

         const Tensor2_3D PK1 = material.get_1st_PK_stress(F);

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

         const Tensor2_3D F = get_deformation_gradient(disp, dN_dX);

         const Tensor4_3D AA = material.get_1st_elasticity_tensor(F);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         for (int aa = 0; aa < num_nodes; aa++)
         {
            const Vector_3D dNa_dX(dN_dX(aa, 0), dN_dX(aa, 1), dN_dX(aa, 2));

            for (int bb = 0; bb < num_nodes; bb++)
            {
               const Vector_3D dNb_dX(dN_dX(bb, 0), dN_dX(bb, 1), dN_dX(bb, 2));

               // N_a,J AA_kJlL N_b,L
               for (int kk = 0; kk < 3; kk++)
                  for (int ll = 0; ll < 3; ll++)
                     tangent(aa+kk*num_nodes, bb+ll*num_nodes) += dV *
                        ( dNa_dX(0) * (AA(kk, 0, ll, 0) * dNb_dX(0)
                                    +  AA(kk, 0, ll, 1) * dNb_dX(1)
                                    +  AA(kk, 0, ll, 2) * dNb_dX(2))
                        + dNa_dX(1) * (AA(kk, 1, ll, 0) * dNb_dX(0)
                                    +  AA(kk, 1, ll, 1) * dNb_dX(1)
                                    +  AA(kk, 1, ll, 2) * dNb_dX(2))
                        + dNa_dX(2) * (AA(kk, 2, ll, 0) * dNb_dX(0)
                                    +  AA(kk, 2, ll, 1) * dNb_dX(1)
                                    +  AA(kk, 2, ll, 2) * dNb_dX(2)));
            }
         }
      }
   }

private:
   const HyperelasticMaterialModel &material;

   // Shared by the residual and the tangent, so that both use the same points.
   static const mfem::IntegrationRule &get_quad_rule(
      const mfem::FiniteElement &elem, mfem::ElementTransformation &elem_map)
   {
      return mfem::IntRules.Get(elem.GetGeomType(), 2 * elem_map.OrderGrad(&elem));
   }

   // F_kJ = delta_kJ + d_ak N_a,J.
   static Tensor2_3D get_deformation_gradient(const mfem::Vector &disp,
                                              const mfem::DenseMatrix &dN_dX)
   {
      const int num_nodes = dN_dX.Height();
      Tensor2_3D F = Tensor2_3D::identity();
      for (int aa = 0; aa < num_nodes; aa++)
      {
         const double disp_x = disp(aa);
         const double disp_y = disp(aa+num_nodes);
         const double disp_z = disp(aa+2*num_nodes);

         F(0, 0) += disp_x * dN_dX(aa, 0);
         F(0, 1) += disp_x * dN_dX(aa, 1);
         F(0, 2) += disp_x * dN_dX(aa, 2);
         F(1, 0) += disp_y * dN_dX(aa, 0);
         F(1, 1) += disp_y * dN_dX(aa, 1);
         F(1, 2) += disp_y * dN_dX(aa, 2);
         F(2, 0) += disp_z * dN_dX(aa, 0);
         F(2, 1) += disp_z * dN_dX(aa, 1);
         F(2, 2) += disp_z * dN_dX(aa, 2);
      }
      return F;
   }
};

#endif

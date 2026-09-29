// ============================================================================
// HyperelasticIntegrator_Mixed.hpp
//
// Element residual and tangent of hyperelasticity in the mixed
// displacement-pressure (u/p) form, Total Lagrangian. The unknowns are the
// displacement and the pressure; the volumetric model enters only through
// J(p), so the same integrator covers compressible and fully incompressible
// materials.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef HYPERELASTIC_INTEGRATOR_MIXED_HPP
#define HYPERELASTIC_INTEGRATOR_MIXED_HPP

#include <mfem.hpp>
#include "IntegratorTools.hpp"
#include "MaterialModel_Hyperelasticity.hpp"
#include "Tensor2_3D.hpp"

class HyperelasticIntegrator_Mixed : public mfem::BlockNonlinearFormIntegrator
{
public:
   HyperelasticIntegrator_Mixed(const MaterialModel_Hyperelasticity &input_material)
      : material(input_material) {}

   // Block 0 is the displacement, block 1 the pressure.
   // R^a_k = int N_a,J (P_ich_kJ - p J F^-1_Jk) dV,
   // R^c   = -int M_c (J - J(p)) dV,
   void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Array<const mfem::Vector *> &elem_sol,
                              const mfem::Array<mfem::Vector *> &residual) override
   {
      const mfem::FiniteElement &elem_u = *elem[0];
      const mfem::FiniteElement &elem_p = *elem[1];
      const mfem::Vector &disp = *elem_sol[0];
      const mfem::Vector &pres = *elem_sol[1];
      mfem::Vector &residual_u = *residual[0];
      mfem::Vector &residual_p = *residual[1];

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();
      mfem::DenseMatrix dN_dxi(num_nodes_u, 3), dN_dX(num_nodes_u, 3);
      mfem::Vector M(num_nodes_p);

      residual_u.SetSize(3 * num_nodes_u);
      residual_u = 0.0;
      residual_p.SetSize(num_nodes_p);
      residual_p = 0.0;

      const mfem::IntegrationRule &quad_rule = IntegratorTools::get_quad_rule(elem_u, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);

         // dN/dxi
         elem_u.CalcDShape(quad_pt, dN_dxi);
         // N_a,J = dN/dX = dN/dxi (dX/dxi)^-1.
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
         // M_c, the pressure shape functions.
         elem_p.CalcShape(quad_pt, M);

         // Interpolate the deformation gradient F.
         const Tensor2_3D F = IntegratorTools::get_deformation_gradient(disp, dN_dX);
         const double J = F.det();

         // Interpolate the pressure p = M_c p_c;
         // Vector * Vector is the inner product in MFEM.
         const double p = M * pres;

         // P = P_ich - p J F^-T
         const Tensor2_3D PK1 = material.get_1st_PK_stress_ich(F)
                                - p * J * F.inverse().transpose();

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         for (int aa = 0; aa < num_nodes_u; aa++)
         {
            // x-component
            residual_u(aa)               += dV * (PK1(0, 0) * dN_dX(aa, 0)
                                                + PK1(0, 1) * dN_dX(aa, 1)
                                                + PK1(0, 2) * dN_dX(aa, 2));
            // y-component
            residual_u(aa+num_nodes_u)   += dV * (PK1(1, 0) * dN_dX(aa, 0)
                                                + PK1(1, 1) * dN_dX(aa, 1)
                                                + PK1(1, 2) * dN_dX(aa, 2));
            // z-component
            residual_u(aa+2*num_nodes_u) += dV * (PK1(2, 0) * dN_dX(aa, 0)
                                                + PK1(2, 1) * dN_dX(aa, 1)
                                                + PK1(2, 2) * dN_dX(aa, 2));
         }

         // J - J(p), zero where the volume ratio matches the pressure.
         const double vol_residual = J - material.get_J(p);
         for (int cc = 0; cc < num_nodes_p; cc++)
            residual_p(cc) -= dV * M(cc) * vol_residual;
      }
   }

private:
   const MaterialModel_Hyperelasticity &material;
};

#endif

// ============================================================================
// LocalAssembly_Mixed.hpp
//
// Local (element) assembly of hyperelasticity in the mixed
// displacement-pressure (u/p) form, Total Lagrangian: the element residual and
// tangent. The unknowns are the displacement and the pressure; the volumetric
// model enters only through J(p), so the same local assembly covers
// compressible and fully incompressible materials. It is an MFEM
// BlockNonlinearFormIntegrator, called by BlockNonlinearForm, which assembles
// the global residual and tangent.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_MIXED_HPP
#define LOCAL_ASSEMBLY_MIXED_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "LocalAssemblyTools.hpp"
#include "MaterialModel.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"
#include "Vector_3D.hpp"

class LocalAssembly_Mixed : public mfem::BlockNonlinearFormIntegrator
{
public:
   // Takes ownership of the material, as in LocalAssembly_Disp.
   LocalAssembly_Mixed(std::unique_ptr<const MaterialModel> input_material)
      : material(std::move(input_material)) {}

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementVector, which BlockNonlinearForm calls on every element.
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

      const mfem::IntegrationRule &quad_rule = LocalAssemblyTools::get_quad_rule(elem_u, elem_map);

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
         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp, dN_dX);
         const double J = F.det();

         // Interpolate the pressure p = M_c p_c;
         // Vector * Vector is the inner product in MFEM.
         const double p = M * pres;

         // P = P_ich - p J F^-T
         const Tensor2_3D PK1 = material->get_1st_PK_stress_ich(F)
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
         const double vol_residual = J - material->get_J(p);
         for (int cc = 0; cc < num_nodes_p; cc++)
            residual_p(cc) -= dV * M(cc) * vol_residual;
      }
   }

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementGrad, which BlockNonlinearForm calls on every element.
   // The four blocks of the tangent, the derivatives of R^a_k and R^c:
   // K_uu(a k, b l) = int N_a,J AA_kJlL N_b,L dV,
   //    AA_kJlL = AA_ich_kJlL - p J (F^-1_Jk F^-1_Ll - F^-1_Jl F^-1_Lk),
   // K_up(a k, d)   = -int J N_a,J F^-1_Jk M_d dV,
   // K_pu(c, b l)   = -int M_c J N_b,L F^-1_Ll dV, the transpose of K_up,
   // K_pp(c, d)     = int dJ/dp M_c M_d dV.
   void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Array<const mfem::Vector *> &elem_sol,
                            const mfem::Array2D<mfem::DenseMatrix *> &tangent) override
   {
      const mfem::FiniteElement &elem_u = *elem[0];
      const mfem::FiniteElement &elem_p = *elem[1];
      const mfem::Vector &disp = *elem_sol[0];
      const mfem::Vector &pres = *elem_sol[1];
      mfem::DenseMatrix &K_uu = *tangent(0, 0);
      mfem::DenseMatrix &K_up = *tangent(0, 1);
      mfem::DenseMatrix &K_pu = *tangent(1, 0);
      mfem::DenseMatrix &K_pp = *tangent(1, 1);

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();
      mfem::DenseMatrix dN_dxi(num_nodes_u, 3), dN_dX(num_nodes_u, 3);
      mfem::Vector M(num_nodes_p);

      K_uu.SetSize(3 * num_nodes_u, 3 * num_nodes_u);
      K_uu = 0.0;
      K_up.SetSize(3 * num_nodes_u, num_nodes_p);
      K_up = 0.0;
      K_pu.SetSize(num_nodes_p, 3 * num_nodes_u);
      K_pu = 0.0;
      K_pp.SetSize(num_nodes_p, num_nodes_p);
      K_pp = 0.0;

      const mfem::IntegrationRule &quad_rule = LocalAssemblyTools::get_quad_rule(elem_u, elem_map);

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
         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp, dN_dX);
         const double J = F.det();
         const Tensor2_3D F_inv = F.inverse();

         // Interpolate the pressure p = M_c p_c;
         // Vector * Vector is the inner product in MFEM.
         const double p = M * pres;

         // AA = AA_ich - p d(J F^-T)/dF
         Tensor4_3D AA = material->get_1st_elasticity_tensor_ich(F);
         for (int kk = 0; kk < 3; kk++)
            for (int JJ = 0; JJ < 3; JJ++)
               for (int ll = 0; ll < 3; ll++)
                  for (int LL = 0; LL < 3; LL++)
                     AA(kk, JJ, ll, LL) -= p * J * (F_inv(JJ, kk) * F_inv(LL, ll)
                                                  - F_inv(JJ, ll) * F_inv(LL, kk));

         const double dJ_dp = material->get_dJ_dp(p);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // K_uu
         for (int aa = 0; aa < num_nodes_u; aa++)
         {
            const Vector_3D dNa_dX(dN_dX(aa, 0), dN_dX(aa, 1), dN_dX(aa, 2));

            for (int bb = 0; bb < num_nodes_u; bb++)
            {
               const Vector_3D dNb_dX(dN_dX(bb, 0), dN_dX(bb, 1), dN_dX(bb, 2));

               // N_a,J AA_kJlL N_b,L
               for (int kk = 0; kk < 3; kk++)
                  for (int ll = 0; ll < 3; ll++)
                     K_uu(aa+kk*num_nodes_u, bb+ll*num_nodes_u) += dV *
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

         // K_up and K_pu, filled together since one is the transpose of the
         // other.
         for (int aa = 0; aa < num_nodes_u; aa++)
            for (int kk = 0; kk < 3; kk++)
            {
               // J N_a,J F^-1_Jk
               const double J_dNa_dx = J * (dN_dX(aa, 0) * F_inv(0, kk)
                                          + dN_dX(aa, 1) * F_inv(1, kk)
                                          + dN_dX(aa, 2) * F_inv(2, kk));
               for (int dd = 0; dd < num_nodes_p; dd++)
               {
                  K_up(aa+kk*num_nodes_u, dd) -= dV * J_dNa_dx * M(dd);
                  K_pu(dd, aa+kk*num_nodes_u) -= dV * J_dNa_dx * M(dd);
               }
            }

         // K_pp
         for (int cc = 0; cc < num_nodes_p; cc++)
            for (int dd = 0; dd < num_nodes_p; dd++)
               K_pp(cc, dd) += dV * dJ_dp * M(cc) * M(dd);
      }
   }

   // Reference density rho_0 of the material, for the mass matrix.
   double get_rho_0() const { return material->get_rho_0(); }

private:
   const std::unique_ptr<const MaterialModel> material;
};

#endif

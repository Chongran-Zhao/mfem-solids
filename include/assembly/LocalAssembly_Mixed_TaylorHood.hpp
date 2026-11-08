// ============================================================================
// LocalAssembly_Mixed_TaylorHood.hpp
//
// Local (element) assembly of hyperelasticity in the mixed displacement-
// pressure (u/p) form with Taylor-Hood elements, the pressure one order below
// the displacement, Total Lagrangian: the element residual and tangent. The
// momentum equation is that of LocalAssembly_Mixed; the mass equation holds
// J = J(p) weakly. The volumetric model enters only through J(p), so the same
// local assembly covers compressible and fully incompressible materials. The
// tangent is symmetric.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_MIXED_TAYLOR_HOOD_HPP
#define LOCAL_ASSEMBLY_MIXED_TAYLOR_HOOD_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "LocalAssemblyTools.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "MaterialModel.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class LocalAssembly_Mixed_TaylorHood : public LocalAssembly_Mixed
{
public:
   // Takes ownership of the material, as in LocalAssembly_Disp.
   LocalAssembly_Mixed_TaylorHood(std::unique_ptr<const MaterialModel> input_material)
      : LocalAssembly_Mixed(std::move(input_material)) {}

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

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem_u, elem_map);

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

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         add_momentum_residual(dN_dX, get_1st_PK_stress(F, p), dV, residual_u);

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

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem_u, elem_map);

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

         const double dJ_dp = material->get_dJ_dp(p);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // K_uu
         LocalAssemblyTools::add_momentum_tangent_uu(dN_dX, get_1st_elasticity_tensor(F, p), dV, K_uu);

         // K_up and K_pu, filled together since one is the transpose of the
         // other.
         for (int aa = 0; aa < num_nodes_u; aa++)
            for (int kk = 0; kk < 3; kk++)
            {
               // J N_a,J F^-1_Jk
               const double J_dNa_dx = get_J_dNa_dx(dN_dX, aa, kk, J, F_inv);
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
};

#endif

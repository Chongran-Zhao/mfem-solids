// ============================================================================
// LocalAssembly_Mixed_VMS.hpp
//
// LocalAssembly_Mixed with the variational multiscale (VMS) stabilization of
// the mass equation, so that the displacement and the pressure may have the
// same order (Q1/Q1, Q2/Q2). The residual of the momentum equation, per
// reference volume,
//    R_mom = rho_0 a + J F^-T Grad p - Div P_ich,
// weighted by the spatial gradient of the pressure test function, is added to
// the mass equation of LocalAssembly_Mixed:
//    R^c = -int M_c (J - J(p)) dV - int tau grad M_c . R_mom dV,
//    grad M_c = F^-T Grad M_c,   tau = tau_factor h^2 / (2 mu),
// with h the size of the element in the reference configuration. Div P_ich
// is kept in R_mom, so that the stabilization vanishes for the exact
// solution; its tangent drops the derivative of AA_ich, the only part of
// the tangent that is not exact. The momentum equation is that of
// LocalAssembly_Mixed, so the tangent is not symmetric.
//
// The acceleration enters as a_alpha = acce_offset + acce_factor u_alpha, an
// affine function of the displacement that the local assembly receives, set
// by set_acce at every time step; without it a = 0, as in statics.
//
// Author: Chongran Zhao
// Date: Oct. 14, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_MIXED_VMS_HPP
#define LOCAL_ASSEMBLY_MIXED_VMS_HPP

#include <array>
#include <cmath>
#include <memory>
#include <utility>

#include <mfem.hpp>

#include "LocalAssemblyTools.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "MaterialModel.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"
#include "Vector_3D.hpp"

class LocalAssembly_Mixed_VMS : public LocalAssembly_Mixed
{
public:
   // Takes ownership of the material; mu and tau_factor give tau.
   LocalAssembly_Mixed_VMS(std::unique_ptr<const MaterialModel> input_material,
                           double input_shear_modulus, double input_tau_factor)
      : LocalAssembly_Mixed(std::move(input_material)),
        shear_modulus(input_shear_modulus),
        tau_factor(input_tau_factor) {}

   // a_alpha = acce_offset + acce_factor u_alpha, with acce_offset on the
   // displacement space, its local dofs shared ones included; it must
   // outlive the next assembly.
   void set_acce(const mfem::GridFunction &input_acce_offset, double input_acce_factor)
   {
      acce_offset = &input_acce_offset;
      acce_factor = input_acce_factor;
   }

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementVector. R^a_k and the Galerkin part of R^c come from
   // LocalAssembly_Mixed; this adds
   // R^c -= int tau grad M_c . R_mom dV.
   void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Array<const mfem::Vector *> &elem_sol,
                              const mfem::Array<mfem::Vector *> &residual) override
   {
      LocalAssembly_Mixed::AssembleElementVector(elem, elem_map, elem_sol, residual);

      const mfem::FiniteElement &elem_u = *elem[0];
      const mfem::FiniteElement &elem_p = *elem[1];
      const mfem::Vector &disp = *elem_sol[0];
      const mfem::Vector &pres = *elem_sol[1];
      mfem::Vector &residual_p = *residual[1];

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();
      mfem::Vector N(num_nodes_u);
      mfem::DenseMatrix dN_dxi(num_nodes_u, 3), dN_dX(num_nodes_u, 3), d2N_dX2(num_nodes_u, 6);
      mfem::DenseMatrix dM_dxi(num_nodes_p, 3), dM_dX(num_nodes_p, 3);

      const mfem::Vector elem_acce_offset = get_elem_acce_offset(elem_map, num_nodes_u);
      const double tau = get_tau(elem_u, elem_map);

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem_u, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);

         // N_a, N_a,J and N_a,JL of the displacement, M_c,J of the pressure.
         elem_u.CalcShape(quad_pt, N);
         elem_u.CalcDShape(quad_pt, dN_dxi);
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
         elem_u.CalcPhysHessian(elem_map, d2N_dX2);
         elem_p.CalcDShape(quad_pt, dM_dxi);
         mfem::Mult(dM_dxi, elem_map.InverseJacobian(), dM_dX);

         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp, dN_dX);
         const Tensor2_3D F_inv = F.inverse();
         const Tensor4_3D AA_ich = material->get_1st_elasticity_tensor_ich(F);
         const Vector_3D R_mom = get_momentum_residual(disp, pres, elem_acce_offset, N,
                                                      d2N_dX2, dM_dX, F, AA_ich);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         for (int cc = 0; cc < num_nodes_p; cc++)
         {
            const Vector_3D dMc_dx = get_spatial_gradient(dM_dX, cc, F_inv);
            residual_p(cc) -= dV * tau * (dMc_dx(0) * R_mom(0) + dMc_dx(1) * R_mom(1)
                                        + dMc_dx(2) * R_mom(2));
         }
      }
   }

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementGrad. The blocks of LocalAssembly_Mixed, and the
   // derivatives of the VMS term of R^c, with grad N_b = F^-T Grad N_b:
   // K_pu(c, b l) += int tau ( - rho_0 acce_factor N_b grad_l M_c
   //                           + grad_l M_c (grad N_b . R_mom)
   //                           - J grad_l N_b (grad M_c . grad p)
   //                           + J grad_l p (grad M_c . grad N_b)
   //                           + grad_k M_c AA_ich_kJlL N_b,LJ ) dV,
   // K_pp(c, d)   -= int tau J grad M_c . grad M_d dV.
   void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Array<const mfem::Vector *> &elem_sol,
                            const mfem::Array2D<mfem::DenseMatrix *> &tangent) override
   {
      LocalAssembly_Mixed::AssembleElementGrad(elem, elem_map, elem_sol, tangent);

      const mfem::FiniteElement &elem_u = *elem[0];
      const mfem::FiniteElement &elem_p = *elem[1];
      const mfem::Vector &disp = *elem_sol[0];
      const mfem::Vector &pres = *elem_sol[1];
      mfem::DenseMatrix &K_pu = *tangent(1, 0);
      mfem::DenseMatrix &K_pp = *tangent(1, 1);

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();
      mfem::Vector N(num_nodes_u);
      mfem::DenseMatrix dN_dxi(num_nodes_u, 3), dN_dX(num_nodes_u, 3), d2N_dX2(num_nodes_u, 6);
      mfem::DenseMatrix dM_dxi(num_nodes_p, 3), dM_dX(num_nodes_p, 3);

      const mfem::Vector elem_acce_offset = get_elem_acce_offset(elem_map, num_nodes_u);
      const double tau = get_tau(elem_u, elem_map);
      const double rho_0 = material->get_rho_0();

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem_u, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);

         // N_a, N_a,J and N_a,JL of the displacement, M_c,J of the pressure.
         elem_u.CalcShape(quad_pt, N);
         elem_u.CalcDShape(quad_pt, dN_dxi);
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
         elem_u.CalcPhysHessian(elem_map, d2N_dX2);
         elem_p.CalcDShape(quad_pt, dM_dxi);
         mfem::Mult(dM_dxi, elem_map.InverseJacobian(), dM_dX);

         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp, dN_dX);
         const Tensor2_3D F_inv = F.inverse();
         const double J = F.det();
         const Tensor4_3D AA_ich = material->get_1st_elasticity_tensor_ich(F);
         const Vector_3D R_mom = get_momentum_residual(disp, pres, elem_acce_offset, N,
                                                      d2N_dX2, dM_dX, F, AA_ich);

         // grad p = F^-T Grad p, Grad p = M_c,J p_c.
         Vector_3D dp_dX;
         for (int cc = 0; cc < num_nodes_p; cc++)
            for (int JJ = 0; JJ < 3; JJ++)
               dp_dX(JJ) += dM_dX(cc, JJ) * pres(cc);
         const Vector_3D dp_dx = get_spatial_gradient(dp_dX, F_inv);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         for (int bb = 0; bb < num_nodes_u; bb++)
         {
            const Vector_3D dNb_dx = get_spatial_gradient(dN_dX, bb, F_inv);
            const double dNb_dx_R_mom = dNb_dx(0) * R_mom(0) + dNb_dx(1) * R_mom(1)
                                      + dNb_dx(2) * R_mom(2);
            const Tensor2_3D d2Nb_dX2 = get_hessian(d2N_dX2, bb);

            // AA_ich_kJlL N_b,LJ for every k and l.
            Tensor2_3D AA_d2Nb;
            for (int kk = 0; kk < 3; kk++)
               for (int ll = 0; ll < 3; ll++)
                  for (int JJ = 0; JJ < 3; JJ++)
                     for (int LL = 0; LL < 3; LL++)
                        AA_d2Nb(kk, ll) += AA_ich(kk, JJ, ll, LL) * d2Nb_dX2(LL, JJ);

            for (int cc = 0; cc < num_nodes_p; cc++)
            {
               const Vector_3D dMc_dx = get_spatial_gradient(dM_dX, cc, F_inv);
               const double dMc_dx_dp_dx = dMc_dx(0) * dp_dx(0) + dMc_dx(1) * dp_dx(1)
                                         + dMc_dx(2) * dp_dx(2);
               const double dMc_dx_dNb_dx = dMc_dx(0) * dNb_dx(0) + dMc_dx(1) * dNb_dx(1)
                                          + dMc_dx(2) * dNb_dx(2);

               for (int ll = 0; ll < 3; ll++)
                  K_pu(cc, bb+ll*num_nodes_u) += dV * tau *
                     ( - rho_0 * acce_factor * N(bb) * dMc_dx(ll)
                       + dMc_dx(ll) * dNb_dx_R_mom
                       - J * dNb_dx(ll) * dMc_dx_dp_dx
                       + J * dp_dx(ll) * dMc_dx_dNb_dx
                       + dMc_dx(0) * AA_d2Nb(0, ll) + dMc_dx(1) * AA_d2Nb(1, ll)
                       + dMc_dx(2) * AA_d2Nb(2, ll) );
            }
         }

         for (int cc = 0; cc < num_nodes_p; cc++)
         {
            const Vector_3D dMc_dx = get_spatial_gradient(dM_dX, cc, F_inv);
            for (int dd = 0; dd < num_nodes_p; dd++)
            {
               const Vector_3D dMd_dx = get_spatial_gradient(dM_dX, dd, F_inv);
               K_pp(cc, dd) -= dV * tau * J * (dMc_dx(0) * dMd_dx(0) + dMc_dx(1) * dMd_dx(1)
                                             + dMc_dx(2) * dMd_dx(2));
            }
         }
      }
   }

private:
   // R_mom_k = rho_0 a_k + J F^-1_Jk p,J - P_ich_kJ,J at the current point,
   // with a = acce_offset + acce_factor u and
   // P_ich_kJ,J = AA_ich_kJlL u_l,LJ.
   Vector_3D get_momentum_residual(const mfem::Vector &disp, const mfem::Vector &pres,
                                   const mfem::Vector &elem_acce_offset, const mfem::Vector &N,
                                   const mfem::DenseMatrix &d2N_dX2,
                                   const mfem::DenseMatrix &dM_dX, const Tensor2_3D &F,
                                   const Tensor4_3D &AA_ich) const
   {
      const int num_nodes_u = N.Size();
      const int num_nodes_p = dM_dX.Height();
      const double rho_0 = material->get_rho_0();
      const double J = F.det();
      const Tensor2_3D F_inv = F.inverse();

      // a_k = N_a (acce_offset_ak + acce_factor d_ak)
      Vector_3D acce;
      for (int aa = 0; aa < num_nodes_u; aa++)
         for (int kk = 0; kk < 3; kk++)
            acce(kk) += N(aa) * (elem_acce_offset(aa+kk*num_nodes_u)
                                 + acce_factor * disp(aa+kk*num_nodes_u));

      // Grad p = M_c,J p_c
      Vector_3D dp_dX;
      for (int cc = 0; cc < num_nodes_p; cc++)
         for (int JJ = 0; JJ < 3; JJ++)
            dp_dX(JJ) += dM_dX(cc, JJ) * pres(cc);

      // u_l,LJ = d_al N_a,LJ
      std::array<Tensor2_3D, 3> d2u_dX2;
      for (int aa = 0; aa < num_nodes_u; aa++)
      {
         const Tensor2_3D d2Na_dX2 = get_hessian(d2N_dX2, aa);
         for (int ll = 0; ll < 3; ll++)
            d2u_dX2[ll] += disp(aa+ll*num_nodes_u) * d2Na_dX2;
      }

      Vector_3D out;
      for (int kk = 0; kk < 3; kk++)
      {
         out(kk) = rho_0 * acce(kk)
                   + J * (F_inv(0, kk) * dp_dX(0) + F_inv(1, kk) * dp_dX(1)
                        + F_inv(2, kk) * dp_dX(2));
         for (int JJ = 0; JJ < 3; JJ++)
            for (int ll = 0; ll < 3; ll++)
               for (int LL = 0; LL < 3; LL++)
                  out(kk) -= AA_ich(kk, JJ, ll, LL) * d2u_dX2[ll](LL, JJ);
      }
      return out;
   }

   // grad_i N_a = F^-1_Ji N_a,J for row aa of dN_dX.
   static Vector_3D get_spatial_gradient(const mfem::DenseMatrix &dN_dX, int aa,
                                         const Tensor2_3D &F_inv)
   {
      return get_spatial_gradient(Vector_3D(dN_dX(aa, 0), dN_dX(aa, 1), dN_dX(aa, 2)), F_inv);
   }

   // grad_i = F^-1_Ji Grad_J
   static Vector_3D get_spatial_gradient(const Vector_3D &dX, const Tensor2_3D &F_inv)
   {
      return Vector_3D(F_inv(0, 0) * dX(0) + F_inv(1, 0) * dX(1) + F_inv(2, 0) * dX(2),
                       F_inv(0, 1) * dX(0) + F_inv(1, 1) * dX(1) + F_inv(2, 1) * dX(2),
                       F_inv(0, 2) * dX(0) + F_inv(1, 2) * dX(1) + F_inv(2, 2) * dX(2));
   }

   // N_a,JL of row aa of MFEM's Hessian, stored as {xx, xy, xz, yy, yz, zz}.
   static Tensor2_3D get_hessian(const mfem::DenseMatrix &d2N_dX2, int aa)
   {
      return Tensor2_3D(d2N_dX2(aa, 0), d2N_dX2(aa, 1), d2N_dX2(aa, 2),
                        d2N_dX2(aa, 1), d2N_dX2(aa, 3), d2N_dX2(aa, 4),
                        d2N_dX2(aa, 2), d2N_dX2(aa, 4), d2N_dX2(aa, 5));
   }

   // acce_offset on the dofs of the element, ordered as the element
   // displacement; zero without set_acce.
   mfem::Vector get_elem_acce_offset(mfem::ElementTransformation &elem_map,
                                     int num_nodes_u) const
   {
      mfem::Vector out(3 * num_nodes_u);
      out = 0.0;
      if (acce_offset == nullptr)
         return out;
      mfem::Array<int> vdofs;
      acce_offset->FESpace()->GetElementVDofs(elem_map.ElementNo, vdofs);
      acce_offset->GetSubVector(vdofs, out);
      return out;
   }

   // tau = tau_factor h^2 / (2 mu), h the cube root of the reference volume
   // of the element, taken at its center.
   double get_tau(const mfem::FiniteElement &elem, mfem::ElementTransformation &elem_map) const
   {
      const mfem::Geometry::Type geom = elem.GetGeomType();
      elem_map.SetIntPoint(&mfem::Geometries.GetCenter(geom));
      const double elem_size = std::cbrt(elem_map.Weight() * mfem::Geometry::Volume[geom]);
      return tau_factor * elem_size * elem_size / (2.0 * shear_modulus);
   }

   const double shear_modulus;                        // mu of tau
   const double tau_factor;                           // tau = tau_factor h^2 / (2 mu)
   const mfem::GridFunction *acce_offset = nullptr;   // a = acce_offset + acce_factor u
   double acce_factor = 0.0;
};

#endif

// ============================================================================
// LocalAssembly_Mixed_VMS.hpp
//
// Local (element) assembly of the rate form of the mixed displacement-
// pressure hyperelastodynamics, with the variational multiscale (VMS)
// stabilization of the mass equation, so that the velocity and the pressure
// may have the same order (Q1/Q1, Q2/Q2), as in PERIGEE solids. The unknowns
// of the element are the velocity v and the pressure p at the intermediate
// state of TimeMethod_GenAlpha_1st; the displacement and the rates are
// affine functions of them,
//    u = disp_offset + disp_factor v,
//    dv/dt = dot_velo_offset + dot_factor v,   dp/dt = dot_pres_offset + dot_factor p,
// set by set_rates at every time step. The residual is
//    R^a_k = int N_a,J (P_ich_kJ - p J F^-1_Jk) dV, that of LocalAssembly_Mixed
//            at u, without the inertia M dv/dt, added by the nonlinear solver;
//    R^c   = -int M_c (J F^-1_Jk v_k,J - J'(p) dp/dt) dV
//            - int tau grad M_c . R_mom dV,
// with the residual of the momentum equation, per reference volume,
//    R_mom = rho_0 dv/dt + J F^-T Grad p - Div P_ich,
// grad M_c = F^-T Grad M_c and tau = tau_factor h / (c rho_0), c = sqrt(mu /
// rho_0), h the size of the element in the reference configuration. Div P_ich
// is kept in R_mom, so that the stabilization vanishes for the exact solution.
// The tangent is that with respect to v and p, exact for a J(p) linear in p,
// as are those of the volumetric models of the mixed form; it is not
// symmetric.
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

   // u = disp_offset + disp_factor v, dv/dt = dot_velo_offset + dot_factor v
   // and dp/dt = dot_pres_offset + dot_factor p, the offsets on the local
   // dofs of the velocity and the pressure spaces, shared ones included;
   // they must outlive the next assembly.
   void set_rates(const mfem::GridFunction &input_disp_offset, double input_disp_factor,
                  const mfem::GridFunction &input_dot_velo_offset,
                  const mfem::GridFunction &input_dot_pres_offset, double input_dot_factor)
   {
      disp_offset = &input_disp_offset;
      disp_factor = input_disp_factor;
      dot_velo_offset = &input_dot_velo_offset;
      dot_pres_offset = &input_dot_pres_offset;
      dot_factor = input_dot_factor;
   }

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementVector. Block 0 is the velocity, block 1 the pressure.
   // R^a_k comes from LocalAssembly_Mixed at u; R^c is that of the rate form
   // with the VMS term.
   void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Array<const mfem::Vector *> &elem_sol,
                              const mfem::Array<mfem::Vector *> &residual) override
   {
      const mfem::FiniteElement &elem_u = *elem[0];
      const mfem::FiniteElement &elem_p = *elem[1];
      const mfem::Vector &velo = *elem_sol[0];
      const mfem::Vector &pres = *elem_sol[1];
      mfem::Vector &residual_p = *residual[1];

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();
      const ElementRates rates = get_element_rates(elem_map, velo, pres);

      // R^a_k at u; the mass equation of LocalAssembly_Mixed is replaced.
      const mfem::Array<const mfem::Vector *> disp_sol({&rates.disp, &pres});
      LocalAssembly_Mixed::AssembleElementVector(elem, elem_map, disp_sol, residual);
      residual_p = 0.0;

      mfem::Vector N(num_nodes_u), M(num_nodes_p);
      mfem::DenseMatrix dN_dxi(num_nodes_u, 3), dN_dX(num_nodes_u, 3), d2N_dX2(num_nodes_u, 6);
      mfem::DenseMatrix dM_dxi(num_nodes_p, 3), dM_dX(num_nodes_p, 3);

      const double tau = get_tau(elem_u, elem_map);

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem_u, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);
         calc_shapes(elem_u, elem_p, elem_map, quad_pt, N, dN_dxi, dN_dX, d2N_dX2,
                     M, dM_dxi, dM_dX);

         const QuadPointState state = get_quad_point_state(velo, pres, rates, N, dN_dX,
                                                           d2N_dX2, M, dM_dX);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // J div v - J'(p) dp/dt, zero where the volume rate matches the
         // pressure rate.
         const double vol_rate = state.J * state.div_velo - state.dJ_dp * state.dot_pres;

         for (int cc = 0; cc < num_nodes_p; cc++)
         {
            const Vector_3D dMc_dx = get_spatial_gradient(dM_dX, cc, state.F_inv);
            residual_p(cc) -= dV * (M(cc) * vol_rate
                                    + tau * (dMc_dx(0) * state.R_mom(0) + dMc_dx(1) * state.R_mom(1)
                                           + dMc_dx(2) * state.R_mom(2)));
         }
      }
   }

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementGrad. The derivatives with respect to v and p, with
   // grad N_b = F^-T Grad N_b, grad v the spatial velocity gradient and
   // the factors c_u = disp_factor, c_r = dot_factor:
   // K_uu(a k, b l) = c_u K_uu of LocalAssembly_Mixed at u,
   // K_up(a k, d)   = K_up of LocalAssembly_Mixed at u,
   // K_pu(c, b l)   = -int M_c J ( grad_l N_b + c_u ( grad_l N_b div v
   //                                               - grad_k N_b grad_l v_k ) ) dV
   //                  + int tau ( - rho_0 c_r N_b grad_l M_c
   //                              + c_u ( grad_l M_c (grad N_b . R_mom)
   //                                      - J grad_l N_b (grad M_c . grad p)
   //                                      + J grad_l p (grad M_c . grad N_b)
   //                                      + grad_k M_c dDivP_kl ) ) dV,
   //    dDivP_kl = AA_ich_kJlL N_b,LJ + dAA_ich_kJlL/dF_mM u_m,MJ N_b,L,
   // the derivative of P_ich_kJ,J = AA_ich_kJmM u_m,MJ with respect to u,
   // whose second part is the derivative of AA_ich along the slices
   // G^J_mM = u_m,MJ, dAA_ich_kJlL/dF_mM G^J_mM = dAA_ich[G^J]_kJlL, since
   // dAA_ich_kJlL/dF_mM is symmetric in (l L) and (m M);
   // K_pp(c, d)     = int ( c_r J'(p) M_c M_d - tau J grad M_c . grad M_d ) dV.
   void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Array<const mfem::Vector *> &elem_sol,
                            const mfem::Array2D<mfem::DenseMatrix *> &tangent) override
   {
      const mfem::FiniteElement &elem_u = *elem[0];
      const mfem::FiniteElement &elem_p = *elem[1];
      const mfem::Vector &velo = *elem_sol[0];
      const mfem::Vector &pres = *elem_sol[1];
      mfem::DenseMatrix &K_uu = *tangent(0, 0);
      mfem::DenseMatrix &K_pu = *tangent(1, 0);
      mfem::DenseMatrix &K_pp = *tangent(1, 1);

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();
      const ElementRates rates = get_element_rates(elem_map, velo, pres);

      // K_uu and K_up at u; the mass equation is replaced.
      const mfem::Array<const mfem::Vector *> disp_sol({&rates.disp, &pres});
      LocalAssembly_Mixed::AssembleElementGrad(elem, elem_map, disp_sol, tangent);
      K_uu *= disp_factor;
      K_pu = 0.0;
      K_pp = 0.0;

      mfem::Vector N(num_nodes_u), M(num_nodes_p);
      mfem::DenseMatrix dN_dxi(num_nodes_u, 3), dN_dX(num_nodes_u, 3), d2N_dX2(num_nodes_u, 6);
      mfem::DenseMatrix dM_dxi(num_nodes_p, 3), dM_dX(num_nodes_p, 3);

      const double tau = get_tau(elem_u, elem_map);
      const double rho_0 = material->get_rho_0();

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem_u, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);
         calc_shapes(elem_u, elem_p, elem_map, quad_pt, N, dN_dxi, dN_dX, d2N_dX2,
                     M, dM_dxi, dM_dX);

         const QuadPointState state = get_quad_point_state(velo, pres, rates, N, dN_dX,
                                                           d2N_dX2, M, dM_dX);
         const Tensor2_3D &F_inv = state.F_inv;
         const double J = state.J;

         // sum_J dAA_ich[G^J]_kJlL, G^J_mM = u_m,MJ, stored as [l](k, L).
         std::array<Tensor2_3D, 3> dAA_d2u;
         for (int JJ = 0; JJ < 3; JJ++)
         {
            const std::array<Tensor2_3D, 3> &d2u = state.d2u_dX2;
            const Tensor2_3D slice_J(d2u[0](0, JJ), d2u[0](1, JJ), d2u[0](2, JJ),
                                     d2u[1](0, JJ), d2u[1](1, JJ), d2u[1](2, JJ),
                                     d2u[2](0, JJ), d2u[2](1, JJ), d2u[2](2, JJ));
            const Tensor4_3D dAA_ich = material->get_d_1st_elasticity_tensor_ich(state.F, slice_J);
            for (int ll = 0; ll < 3; ll++)
               for (int kk = 0; kk < 3; kk++)
                  for (int LL = 0; LL < 3; LL++)
                     dAA_d2u[ll](kk, LL) += dAA_ich(kk, JJ, ll, LL);
         }

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         for (int bb = 0; bb < num_nodes_u; bb++)
         {
            const Vector_3D dNb_dx = get_spatial_gradient(dN_dX, bb, F_inv);
            const double dNb_dx_R_mom = dNb_dx(0) * state.R_mom(0) + dNb_dx(1) * state.R_mom(1)
                                      + dNb_dx(2) * state.R_mom(2);
            const Tensor2_3D d2Nb_dX2 = get_hessian(d2N_dX2, bb);

            // grad_k N_b grad_l v_k for every l.
            const Vector_3D dNb_dx_dv_dx(
               dNb_dx(0) * state.dv_dx(0, 0) + dNb_dx(1) * state.dv_dx(1, 0) + dNb_dx(2) * state.dv_dx(2, 0),
               dNb_dx(0) * state.dv_dx(0, 1) + dNb_dx(1) * state.dv_dx(1, 1) + dNb_dx(2) * state.dv_dx(2, 1),
               dNb_dx(0) * state.dv_dx(0, 2) + dNb_dx(1) * state.dv_dx(1, 2) + dNb_dx(2) * state.dv_dx(2, 2));

            // dDivP_kl for every k and l.
            Tensor2_3D dDivP;
            for (int kk = 0; kk < 3; kk++)
               for (int ll = 0; ll < 3; ll++)
               {
                  for (int JJ = 0; JJ < 3; JJ++)
                     for (int LL = 0; LL < 3; LL++)
                        dDivP(kk, ll) += state.AA_ich(kk, JJ, ll, LL) * d2Nb_dX2(LL, JJ);
                  for (int LL = 0; LL < 3; LL++)
                     dDivP(kk, ll) += dAA_d2u[ll](kk, LL) * dN_dX(bb, LL);
               }

            for (int cc = 0; cc < num_nodes_p; cc++)
            {
               const Vector_3D dMc_dx = get_spatial_gradient(dM_dX, cc, F_inv);
               const double dMc_dx_dp_dx = dMc_dx(0) * state.dp_dx(0) + dMc_dx(1) * state.dp_dx(1)
                                         + dMc_dx(2) * state.dp_dx(2);
               const double dMc_dx_dNb_dx = dMc_dx(0) * dNb_dx(0) + dMc_dx(1) * dNb_dx(1)
                                          + dMc_dx(2) * dNb_dx(2);

               for (int ll = 0; ll < 3; ll++)
                  K_pu(cc, bb+ll*num_nodes_u) += dV *
                     ( - M(cc) * J * ( dNb_dx(ll)
                                       + disp_factor * ( dNb_dx(ll) * state.div_velo
                                                         - dNb_dx_dv_dx(ll) ) )
                       + tau * ( - rho_0 * dot_factor * N(bb) * dMc_dx(ll)
                                 + disp_factor * ( dMc_dx(ll) * dNb_dx_R_mom
                                                   - J * dNb_dx(ll) * dMc_dx_dp_dx
                                                   + J * state.dp_dx(ll) * dMc_dx_dNb_dx
                                                   + dMc_dx(0) * dDivP(0, ll)
                                                   + dMc_dx(1) * dDivP(1, ll)
                                                   + dMc_dx(2) * dDivP(2, ll) ) ) );
            }
         }

         for (int cc = 0; cc < num_nodes_p; cc++)
         {
            const Vector_3D dMc_dx = get_spatial_gradient(dM_dX, cc, F_inv);
            for (int dd = 0; dd < num_nodes_p; dd++)
            {
               const Vector_3D dMd_dx = get_spatial_gradient(dM_dX, dd, F_inv);
               K_pp(cc, dd) += dV * (dot_factor * state.dJ_dp * M(cc) * M(dd)
                                     - tau * J * (dMc_dx(0) * dMd_dx(0) + dMc_dx(1) * dMd_dx(1)
                                                + dMc_dx(2) * dMd_dx(2)));
            }
         }
      }
   }

private:
   // The element values of u, dv/dt and dp/dt from v and p.
   struct ElementRates
   {
      mfem::Vector disp;       // u = disp_offset + disp_factor v
      mfem::Vector dot_velo;   // dv/dt = dot_velo_offset + dot_factor v
      mfem::Vector dot_pres;   // dp/dt = dot_pres_offset + dot_factor p
   };

   // What the residual and the tangent need at a quadrature point.
   struct QuadPointState
   {
      Tensor2_3D F;                        // deformation gradient
      Tensor2_3D F_inv;                    // F^-1
      double J;                            // det F
      Tensor2_3D dv_dx;                    // grad_l v_k, stored as (k, l)
      double div_velo;                     // div v = grad_k v_k
      Vector_3D dp_dx;                     // grad p
      double dot_pres;                     // dp/dt
      double dJ_dp;                        // J'(p)
      Tensor4_3D AA_ich;                   // dP_ich/dF
      std::array<Tensor2_3D, 3> d2u_dX2;   // u_l,LJ, stored as [l](L, J)
      Vector_3D R_mom;                     // residual of the momentum equation
   };

   ElementRates get_element_rates(mfem::ElementTransformation &elem_map,
                                  const mfem::Vector &velo, const mfem::Vector &pres) const
   {
      MFEM_VERIFY(disp_offset != nullptr, "LocalAssembly_Mixed_VMS needs set_rates.");
      ElementRates out;
      out.disp = get_elem_values(*disp_offset, elem_map);
      out.disp.Add(disp_factor, velo);
      out.dot_velo = get_elem_values(*dot_velo_offset, elem_map);
      out.dot_velo.Add(dot_factor, velo);
      out.dot_pres = get_elem_values(*dot_pres_offset, elem_map);
      out.dot_pres.Add(dot_factor, pres);
      return out;
   }

   // N, N_a,J and N_a,JL of the velocity, M and M_c,J of the pressure.
   static void calc_shapes(const mfem::FiniteElement &elem_u, const mfem::FiniteElement &elem_p,
                           mfem::ElementTransformation &elem_map,
                           const mfem::IntegrationPoint &quad_pt, mfem::Vector &N,
                           mfem::DenseMatrix &dN_dxi, mfem::DenseMatrix &dN_dX,
                           mfem::DenseMatrix &d2N_dX2, mfem::Vector &M,
                           mfem::DenseMatrix &dM_dxi, mfem::DenseMatrix &dM_dX)
   {
      elem_u.CalcShape(quad_pt, N);
      elem_u.CalcDShape(quad_pt, dN_dxi);
      mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
      elem_u.CalcPhysHessian(elem_map, d2N_dX2);
      elem_p.CalcShape(quad_pt, M);
      elem_p.CalcDShape(quad_pt, dM_dxi);
      mfem::Mult(dM_dxi, elem_map.InverseJacobian(), dM_dX);
   }

   // The state at the current point, with
   // R_mom_k = rho_0 dv_k/dt + J F^-1_Jk p,J - P_ich_kJ,J and
   // P_ich_kJ,J = AA_ich_kJlL u_l,LJ.
   QuadPointState get_quad_point_state(const mfem::Vector &velo, const mfem::Vector &pres,
                                       const ElementRates &rates, const mfem::Vector &N,
                                       const mfem::DenseMatrix &dN_dX,
                                       const mfem::DenseMatrix &d2N_dX2,
                                       const mfem::Vector &M,
                                       const mfem::DenseMatrix &dM_dX) const
   {
      const int num_nodes_u = N.Size();
      const int num_nodes_p = M.Size();

      QuadPointState out;
      out.F = LocalAssemblyTools::get_deformation_gradient(rates.disp, dN_dX);
      out.F_inv = out.F.inverse();
      out.J = out.F.det();

      // grad_l v_k = v_k,J F^-1_Jl
      const Tensor2_3D dv_dX = LocalAssemblyTools::get_deformation_gradient(velo, dN_dX)
                               - Tensor2_3D::identity();
      out.dv_dx = dv_dX * out.F_inv;
      out.div_velo = out.dv_dx.trace();

      // Grad p = M_c,J p_c, grad p = F^-T Grad p, p and dp/dt.
      Vector_3D dp_dX;
      double p = 0.0;
      out.dot_pres = 0.0;
      for (int cc = 0; cc < num_nodes_p; cc++)
      {
         for (int JJ = 0; JJ < 3; JJ++)
            dp_dX(JJ) += dM_dX(cc, JJ) * pres(cc);
         p += M(cc) * pres(cc);
         out.dot_pres += M(cc) * rates.dot_pres(cc);
      }
      out.dp_dx = get_spatial_gradient(dp_dX, out.F_inv);
      out.dJ_dp = material->get_dJ_dp(p);

      // dv_k/dt = N_a dv_ak/dt
      Vector_3D dot_velo;
      for (int aa = 0; aa < num_nodes_u; aa++)
         for (int kk = 0; kk < 3; kk++)
            dot_velo(kk) += N(aa) * rates.dot_velo(aa+kk*num_nodes_u);

      out.AA_ich = material->get_1st_elasticity_tensor_ich(out.F);
      out.d2u_dX2 = get_disp_hessian(rates.disp, d2N_dX2);

      const double rho_0 = material->get_rho_0();
      for (int kk = 0; kk < 3; kk++)
      {
         out.R_mom(kk) = rho_0 * dot_velo(kk) + out.J * out.dp_dx(kk);
         for (int JJ = 0; JJ < 3; JJ++)
            for (int ll = 0; ll < 3; ll++)
               for (int LL = 0; LL < 3; LL++)
                  out.R_mom(kk) -= out.AA_ich(kk, JJ, ll, LL) * out.d2u_dX2[ll](LL, JJ);
      }
      return out;
   }

   // u_l,LJ = d_al N_a,LJ, stored as [l](L, J).
   static std::array<Tensor2_3D, 3> get_disp_hessian(const mfem::Vector &disp,
                                                     const mfem::DenseMatrix &d2N_dX2)
   {
      const int num_nodes_u = d2N_dX2.Height();
      std::array<Tensor2_3D, 3> out;
      for (int aa = 0; aa < num_nodes_u; aa++)
      {
         const Tensor2_3D d2Na_dX2 = get_hessian(d2N_dX2, aa);
         for (int ll = 0; ll < 3; ll++)
            out[ll] += disp(aa+ll*num_nodes_u) * d2Na_dX2;
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

   // The values of a grid function on the dofs of the element, ordered as
   // the element vectors of BlockNonlinearForm.
   static mfem::Vector get_elem_values(const mfem::GridFunction &field,
                                       mfem::ElementTransformation &elem_map)
   {
      mfem::Array<int> vdofs;
      field.FESpace()->GetElementVDofs(elem_map.ElementNo, vdofs);
      mfem::Vector out;
      field.GetSubVector(vdofs, out);
      return out;
   }

   // tau = tau_factor h / (c rho_0), c = sqrt(mu / rho_0), h the cube root
   // of the reference volume of the element, taken at its center.
   double get_tau(const mfem::FiniteElement &elem, mfem::ElementTransformation &elem_map) const
   {
      const mfem::Geometry::Type geom = elem.GetGeomType();
      elem_map.SetIntPoint(&mfem::Geometries.GetCenter(geom));
      const double elem_size = std::cbrt(elem_map.Weight() * mfem::Geometry::Volume[geom]);
      const double rho_0 = material->get_rho_0();
      return tau_factor * elem_size / (std::sqrt(shear_modulus / rho_0) * rho_0);
   }

   const double shear_modulus;                            // mu of tau
   const double tau_factor;                               // tau = tau_factor h / (c rho_0)
   const mfem::GridFunction *disp_offset = nullptr;       // u = disp_offset + disp_factor v
   double disp_factor = 0.0;
   const mfem::GridFunction *dot_velo_offset = nullptr;   // dv/dt = dot_velo_offset + dot_factor v
   const mfem::GridFunction *dot_pres_offset = nullptr;   // dp/dt = dot_pres_offset + dot_factor p
   double dot_factor = 0.0;
};

#endif

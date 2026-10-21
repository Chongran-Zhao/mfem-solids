// ============================================================================
// LocalAssembly_Mixed_VMS.hpp
//
// Local (element) assembly of the rate form of the mixed displacement-
// pressure hyperelastodynamics, with the variational multiscale (VMS)
// stabilization of the mass equation, so that the velocity and the pressure
// may have the same order (Q1/Q1, Q2/Q2), as in PERIGEE solids. The unknowns
// of the element are the velocity v and the pressure p at the intermediate
// state of TimeMethod_GenAlpha_1st_Order; the displacement and the rates are
// affine functions of them,
//    u = disp_offset + disp_factor v,
//    dv/dt = dot_velo_offset + dot_factor v,   dp/dt = dot_pres_offset + dot_factor p,
// set by set_rates at every time step. The residual is
//    R^a_k = int N_a,J (P_ich_kJ - p J F^-1_Jk) dV, the momentum equation of
//            LocalAssembly_Mixed at u, without the inertia M dv/dt, added by
//            the nonlinear solver;
//    R^c   = -int M_c (J F^-1_Jk v_k,J - J'(p) dp/dt) dV
//            - int tau grad M_c . R_mom dV,
// with the residual of the momentum equation, per reference volume,
//    R_mom = rho_0 dv/dt + J F^-T Grad p - Div P_ich,
// grad M_c = F^-T Grad M_c and tau = tau_factor dx J / (c rho_0 k^2): as in
// PERIGEE and Liu & Marsden (CMAME 337, 2018), dx the longest diagonal of a
// hexahedron or the diameter of the circumscribed sphere of a tetrahedron, in
// the reference configuration, and c the largest wave speed, sqrt(mu / rho_0)
// for an incompressible material and sqrt((kappa + 4/3 mu) / rho_0), kappa =
// -1 / J'(0), for a compressible one; k the order of the elements. Div P_ich
// is kept in R_mom, so that the stabilization vanishes for the exact solution.
// With it, tau has an upper bound, as for the Laplacian in the residual of
// the stabilized Stokes problem, smaller for higher orders: with the tau of
// PERIGEE, Q2/Q2 grows a spurious pressure mode at the vertices of the free
// edges, which the factor 1 / k^2 removes; for linear elements tau is that of
// PERIGEE.
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

#include <algorithm>
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
      mfem::Vector &residual_u = *residual[0];
      mfem::Vector &residual_p = *residual[1];

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();
      const ElementRates rates = get_element_rates(elem_map, velo, pres);

      residual_u.SetSize(3 * num_nodes_u);
      residual_u = 0.0;
      residual_p.SetSize(num_nodes_p);
      residual_p = 0.0;

      mfem::Vector N(num_nodes_u), M(num_nodes_p);
      mfem::DenseMatrix dN_dxi(num_nodes_u, 3), dN_dX(num_nodes_u, 3), d2N_dX2(num_nodes_u, 6);
      mfem::DenseMatrix dM_dxi(num_nodes_p, 3), dM_dX(num_nodes_p, 3);
      mfem::DenseMatrix d2N_dxi2, d2G_dxi2;   // work matrices of calc_shapes

      const double tau_J = get_tau_per_J(elem_u, elem_map);

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem_u, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);
         calc_shapes(elem_u, elem_p, elem_map, quad_pt, N, dN_dxi, dN_dX, d2N_dX2,
                     M, dM_dxi, dM_dX, d2N_dxi2, d2G_dxi2);

         const QuadPointState state = get_quad_point_state(velo, pres, rates, N, dN_dX,
                                                           d2N_dX2, M, dM_dX);
         const double tau = tau_J * state.J;

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // R^a_k at u.
         add_momentum_residual(dN_dX, get_1st_PK_stress(state.F, state.p), dV, residual_u);

         // J div v - J'(p) dp/dt, zero where the volume rate matches the
         // pressure rate.
         const double vol_rate = state.J * state.div_velo - state.dJ_dp * state.dot_pres;

         for (int cc = 0; cc < num_nodes_p; cc++)
         {
            const Vector_3D dMc_dx = get_spatial_gradient(dM_dX, cc, state.F_inv);
            const double dMc_dx_R_mom = dMc_dx(0) * state.R_mom(0) + dMc_dx(1) * state.R_mom(1)
                                      + dMc_dx(2) * state.R_mom(2);
            residual_p(cc) -= dV * (M(cc) * vol_rate + tau * dMc_dx_R_mom);
         }
      }
   }

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementGrad. The derivatives with respect to v and p, with
   // grad N_b = F^-T Grad N_b, grad v the spatial velocity gradient and
   // the factors c_u = disp_factor, c_r = dot_factor:
   // K_uu(a k, b l) = c_u int N_a,J AA_kJlL N_b,L dV, AA of LocalAssembly_Mixed at u,
   // K_up(a k, d)   = -int J N_a,J F^-1_Jk M_d dV,
   // K_pu(c, b l)   = -int M_c J ( grad_l N_b + c_u ( grad_l N_b div v
   //                                               - grad_k N_b grad_l v_k ) ) dV
   //                  + int tau ( - rho_0 c_r N_b grad_l M_c
   //                              + c_u ( grad_l M_c (grad N_b . R_mom)
   //                                      - grad_l N_b (grad M_c . R_mom)
   //                                      - J grad_l N_b (grad M_c . grad p)
   //                                      + J grad_l p (grad M_c . grad N_b)
   //                                      + grad_k M_c dDivP_kl ) ) dV,
   //    dDivP_kl = AA_ich_kJlL N_b,LJ + dAA_ich_kJlL/dF_mM u_m,MJ N_b,L,
   // the derivative of P_ich_kJ,J = AA_ich_kJmM u_m,MJ with respect to u,
   // whose second part is the derivative of AA_ich along the slices
   // G^J_mM = u_m,MJ, dAA_ich_kJlL/dF_mM G^J_mM = dAA_ich[G^J]_kJlL, since
   // dAA_ich_kJlL/dF_mM is symmetric in (l L) and (m M); the term
   // - grad_l N_b (grad M_c . R_mom) is the derivative of the J in tau;
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
      mfem::DenseMatrix &K_up = *tangent(0, 1);
      mfem::DenseMatrix &K_pu = *tangent(1, 0);
      mfem::DenseMatrix &K_pp = *tangent(1, 1);

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();
      const ElementRates rates = get_element_rates(elem_map, velo, pres);

      K_uu.SetSize(3 * num_nodes_u, 3 * num_nodes_u);
      K_uu = 0.0;
      K_up.SetSize(3 * num_nodes_u, num_nodes_p);
      K_up = 0.0;
      K_pu.SetSize(num_nodes_p, 3 * num_nodes_u);
      K_pu = 0.0;
      K_pp.SetSize(num_nodes_p, num_nodes_p);
      K_pp = 0.0;

      mfem::Vector N(num_nodes_u), M(num_nodes_p);
      mfem::DenseMatrix dN_dxi(num_nodes_u, 3), dN_dX(num_nodes_u, 3), d2N_dX2(num_nodes_u, 6);
      mfem::DenseMatrix dM_dxi(num_nodes_p, 3), dM_dX(num_nodes_p, 3);
      mfem::DenseMatrix d2N_dxi2, d2G_dxi2;   // work matrices of calc_shapes

      const double tau_J = get_tau_per_J(elem_u, elem_map);
      const double rho_0 = material->get_rho_0();

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem_u, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);
         calc_shapes(elem_u, elem_p, elem_map, quad_pt, N, dN_dxi, dN_dX, d2N_dX2,
                     M, dM_dxi, dM_dX, d2N_dxi2, d2G_dxi2);

         const QuadPointState state = get_quad_point_state(velo, pres, rates, N, dN_dX,
                                                           d2N_dX2, M, dM_dX);
         const Tensor2_3D &F_inv = state.F_inv;
         const double J = state.J;
         const double tau = tau_J * J;

         // CC_ich and AA_ich, once for the point.
         const Tensor4_3D CC_ich = material->get_2nd_elasticity_tensor_ich(state.F);
         const Tensor4_3D AA_ich = material->get_1st_elasticity_tensor_ich(state.F, CC_ich);

         // sum_J dAA_ich[G^J]_kJlL, G^J_mM = u_m,MJ, stored as [l](k, L).
         const std::array<Tensor2_3D, 3> dAA_d2u =
            material->get_d_1st_elasticity_tensor_ich_sum(state.F, CC_ich,
                                                          get_hessian_slices(state.d2u_dX2));

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // K_uu and K_up of the momentum equation at u.
         LocalAssemblyTools::add_momentum_tangent_uu(dN_dX, get_1st_elasticity_tensor(state.F, state.p, AA_ich),
                                 disp_factor * dV, K_uu);
         for (int aa = 0; aa < num_nodes_u; aa++)
            for (int kk = 0; kk < 3; kk++)
            {
               const double J_dNa_dx = get_J_dNa_dx(dN_dX, aa, kk, J, F_inv);
               for (int dd = 0; dd < num_nodes_p; dd++)
                  K_up(aa+kk*num_nodes_u, dd) -= dV * J_dNa_dx * M(dd);
            }

         for (int bb = 0; bb < num_nodes_u; bb++)
         {
            const Vector_3D dNb_dx = get_spatial_gradient(dN_dX, bb, F_inv);
            const double dNb_dx_R_mom = dNb_dx(0) * state.R_mom(0) + dNb_dx(1) * state.R_mom(1)
                                      + dNb_dx(2) * state.R_mom(2);
            const Tensor2_3D d2Nb_dX2 = get_hessian(d2N_dX2, bb);

            // grad_k N_b grad_l v_k for every l.
            const Tensor2_3D &dv_dx = state.dv_dx;
            const Vector_3D dNb_dx_dv_dx(
               dNb_dx(0) * dv_dx(0, 0) + dNb_dx(1) * dv_dx(1, 0) + dNb_dx(2) * dv_dx(2, 0),
               dNb_dx(0) * dv_dx(0, 1) + dNb_dx(1) * dv_dx(1, 1) + dNb_dx(2) * dv_dx(2, 1),
               dNb_dx(0) * dv_dx(0, 2) + dNb_dx(1) * dv_dx(1, 2) + dNb_dx(2) * dv_dx(2, 2));

            // dDivP_kl for every k and l.
            Tensor2_3D dDivP;
            for (int kk = 0; kk < 3; kk++)
               for (int ll = 0; ll < 3; ll++)
               {
                  for (int JJ = 0; JJ < 3; JJ++)
                     for (int LL = 0; LL < 3; LL++)
                        dDivP(kk, ll) += AA_ich(kk, JJ, ll, LL) * d2Nb_dX2(LL, JJ);
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
               const double dMc_dx_R_mom = dMc_dx(0) * state.R_mom(0) + dMc_dx(1) * state.R_mom(1)
                                         + dMc_dx(2) * state.R_mom(2);

               for (int ll = 0; ll < 3; ll++)
                  K_pu(cc, bb+ll*num_nodes_u) += dV *
                     ( - M(cc) * J * ( dNb_dx(ll)
                                       + disp_factor * ( dNb_dx(ll) * state.div_velo
                                                         - dNb_dx_dv_dx(ll) ) )
                       + tau * ( - rho_0 * dot_factor * N(bb) * dMc_dx(ll)
                                 + disp_factor * ( dMc_dx(ll) * dNb_dx_R_mom
                                                   - dNb_dx(ll) * dMc_dx_R_mom
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
      double p;                            // pressure
      Vector_3D dp_dx;                     // grad p
      double dot_pres;                     // dp/dt
      double dJ_dp;                        // J'(p)
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

   // N, N_a,J and N_a,JL of the velocity, M and M_c,J of the pressure;
   // d2N_dxi2 and d2G_dxi2 are work matrices.
   static void calc_shapes(const mfem::FiniteElement &elem_u, const mfem::FiniteElement &elem_p,
                           mfem::ElementTransformation &elem_map,
                           const mfem::IntegrationPoint &quad_pt, mfem::Vector &N,
                           mfem::DenseMatrix &dN_dxi, mfem::DenseMatrix &dN_dX,
                           mfem::DenseMatrix &d2N_dX2, mfem::Vector &M,
                           mfem::DenseMatrix &dM_dxi, mfem::DenseMatrix &dM_dX,
                           mfem::DenseMatrix &d2N_dxi2, mfem::DenseMatrix &d2G_dxi2)
   {
      elem_u.CalcShape(quad_pt, N);
      elem_u.CalcDShape(quad_pt, dN_dxi);
      mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
      LocalAssemblyTools::calc_phys_hessian(elem_u, elem_map, dN_dX, d2N_dxi2, d2G_dxi2, d2N_dX2);
      elem_p.CalcShape(quad_pt, M);
      elem_p.CalcDShape(quad_pt, dM_dxi);
      mfem::Mult(dM_dxi, elem_map.InverseJacobian(), dM_dX);
   }

   // The state at the current point, with
   // R_mom_k = rho_0 dv_k/dt + J F^-1_Jk p,J - P_ich_kJ,J and
   // P_ich_kJ,J = AA_ich_kJlL u_l,LJ = sum_J dP_ich[G^J]_kJ, the derivatives of
   // P_ich along G^J_lL = u_l,LJ, without AA_ich.
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
      out.p = 0.0;
      out.dot_pres = 0.0;
      for (int cc = 0; cc < num_nodes_p; cc++)
      {
         for (int JJ = 0; JJ < 3; JJ++)
            dp_dX(JJ) += dM_dX(cc, JJ) * pres(cc);
         out.p += M(cc) * pres(cc);
         out.dot_pres += M(cc) * rates.dot_pres(cc);
      }
      out.dp_dx = get_spatial_gradient(dp_dX, out.F_inv);
      out.dJ_dp = material->get_dJ_dp(out.p);

      // dv_k/dt = N_a dv_ak/dt
      Vector_3D dot_velo;
      for (int aa = 0; aa < num_nodes_u; aa++)
         for (int kk = 0; kk < 3; kk++)
            dot_velo(kk) += N(aa) * rates.dot_velo(aa+kk*num_nodes_u);

      out.d2u_dX2 = get_disp_hessian(rates.disp, d2N_dX2);

      const double rho_0 = material->get_rho_0();
      for (int kk = 0; kk < 3; kk++)
         out.R_mom(kk) = rho_0 * dot_velo(kk) + out.J * out.dp_dx(kk);
      const std::array<Tensor2_3D, 3> slices = get_hessian_slices(out.d2u_dX2);
      for (int JJ = 0; JJ < 3; JJ++)
      {
         const Tensor2_3D dP_ich = material->get_d_1st_PK_stress_ich(out.F, slices[JJ]);
         for (int kk = 0; kk < 3; kk++)
            out.R_mom(kk) -= dP_ich(kk, JJ);
      }
      return out;
   }

   // The slices G^J_lL = u_l,LJ of d2u_dX2 = u_l,LJ stored as [l](L, J).
   static std::array<Tensor2_3D, 3> get_hessian_slices(const std::array<Tensor2_3D, 3> &d2u_dX2)
   {
      const std::array<Tensor2_3D, 3> &d2u = d2u_dX2;
      std::array<Tensor2_3D, 3> out;
      for (int JJ = 0; JJ < 3; JJ++)
         out[JJ] = Tensor2_3D(d2u[0](0, JJ), d2u[0](1, JJ), d2u[0](2, JJ),
                              d2u[1](0, JJ), d2u[1](1, JJ), d2u[1](2, JJ),
                              d2u[2](0, JJ), d2u[2](1, JJ), d2u[2](2, JJ));
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

   // tau / J = tau_factor dx / (c rho_0 k^2), with dx the size of the
   // element, c the largest wave speed and k the order of the element;
   // constant over the element.
   double get_tau_per_J(const mfem::FiniteElement &elem,
                        mfem::ElementTransformation &elem_map) const
   {
      const double rho_0 = material->get_rho_0();
      const double dJ_dp = material->get_dJ_dp(0.0);
      const double wave_modulus = (dJ_dp == 0.0) ? shear_modulus
                                                 : -1.0 / dJ_dp + 4.0 / 3.0 * shear_modulus;
      const int order = elem.GetOrder();
      return tau_factor * get_elem_size(elem, elem_map)
             / (std::sqrt(wave_modulus / rho_0) * rho_0 * order * order);
   }

   // dx of PERIGEE in the reference configuration: the longest of the four
   // diagonals of a hexahedron, between vertices 0-6, 1-7, 2-4 and 3-5, or
   // the diameter of the sphere through the four vertices of a tetrahedron.
   static double get_elem_size(const mfem::FiniteElement &elem,
                               mfem::ElementTransformation &elem_map)
   {
      const mfem::Geometry::Type geom = elem.GetGeomType();
      const mfem::IntegrationRule &ref_vertices = *mfem::Geometries.GetVertices(geom);
      std::array<Vector_3D, 8> vertex;
      mfem::Vector xyz(3);
      for (int vv = 0; vv < ref_vertices.GetNPoints(); vv++)
      {
         elem_map.Transform(ref_vertices.IntPoint(vv), xyz);
         vertex[vv] = Vector_3D(xyz(0), xyz(1), xyz(2));
      }

      if (geom == mfem::Geometry::CUBE)
      {
         double out = 0.0;
         const int diagonals[4][2] = {{0, 6}, {1, 7}, {2, 4}, {3, 5}};
         for (const auto &diagonal : diagonals)
            out = std::max(out, get_distance(vertex[diagonal[0]], vertex[diagonal[1]]));
         return out;
      }

      MFEM_VERIFY(geom == mfem::Geometry::TETRAHEDRON,
                  "LocalAssembly_Mixed_VMS supports hexahedra and tetrahedra.");
      // The center x of the sphere solves 2 (X_i - X_0) . x = |X_i|^2 - |X_0|^2.
      Tensor2_3D AA;
      Vector_3D bb;
      for (int ii = 0; ii < 3; ii++)
      {
         for (int jj = 0; jj < 3; jj++)
            AA(ii, jj) = 2.0 * (vertex[ii+1](jj) - vertex[0](jj));
         for (int jj = 0; jj < 3; jj++)
            bb(ii) += vertex[ii+1](jj) * vertex[ii+1](jj) - vertex[0](jj) * vertex[0](jj);
      }
      const Tensor2_3D AA_inv = AA.inverse();
      const Vector_3D center(AA_inv(0, 0) * bb(0) + AA_inv(0, 1) * bb(1) + AA_inv(0, 2) * bb(2),
                             AA_inv(1, 0) * bb(0) + AA_inv(1, 1) * bb(1) + AA_inv(1, 2) * bb(2),
                             AA_inv(2, 0) * bb(0) + AA_inv(2, 1) * bb(1) + AA_inv(2, 2) * bb(2));
      return 2.0 * get_distance(center, vertex[0]);
   }

   // |X - Y|
   static double get_distance(const Vector_3D &X, const Vector_3D &Y)
   {
      return std::sqrt((X(0) - Y(0)) * (X(0) - Y(0)) + (X(1) - Y(1)) * (X(1) - Y(1))
                       + (X(2) - Y(2)) * (X(2) - Y(2)));
   }

   const double shear_modulus;                            // mu of tau
   const double tau_factor;                               // tau = tau_factor dx J / (c rho_0 k^2)
   const mfem::GridFunction *disp_offset = nullptr;       // u = disp_offset + disp_factor v
   double disp_factor = 0.0;
   const mfem::GridFunction *dot_velo_offset = nullptr;   // dv/dt = dot_velo_offset + dot_factor v
   const mfem::GridFunction *dot_pres_offset = nullptr;   // dp/dt = dot_pres_offset + dot_factor p
   double dot_factor = 0.0;
};

#endif

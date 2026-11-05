// ============================================================================
// MassEquation_VMS.hpp
//
// The mass equation of the rate form of the mixed displacement-pressure
// hyperelastodynamics, with the variational multiscale (VMS) stabilization, so
// that the velocity and the pressure may have the same order (Q1/Q1, Q2/Q2),
// as in PERIGEE solids. The unknowns of the element are the velocity v and the
// pressure p at the intermediate state of TimeMethod_GenAlpha_1st_Order; the
// displacement and the rates are affine functions of them,
//    u = disp_offset + disp_factor v,
//    dv/dt = dot_velo_offset + dot_factor v,   dp/dt = dot_pres_offset + dot_factor p,
// set by set_rates at every time step. The residual is
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
// With several regions, each has its shear modulus and tau_factor.
//
// Author: Chongran Zhao
// Date: Oct. 14, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MASS_EQUATION_VMS_HPP
#define MASS_EQUATION_VMS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <mfem.hpp>

#include "IMassEquation.hpp"
#include "LocalAssemblyTools.hpp"
#include "Vector_3D.hpp"

class MassEquation_VMS : public IMassEquation
{
public:
   // mu and tau_factor give tau.
   MassEquation_VMS(double input_shear_modulus, double input_tau_factor)
      : MassEquation_VMS(std::vector<double>{input_shear_modulus},
                         std::vector<double>{input_tau_factor}) {}

   // One mu and tau_factor per region: shear_moduli[aa] and tau_factors[aa] are
   // those of element attribute aa + 1.
   MassEquation_VMS(const std::vector<double> &input_shear_moduli,
                    const std::vector<double> &input_tau_factors)
      : shear_moduli(input_shear_moduli),
        tau_factors(input_tau_factors)
   {
      MFEM_VERIFY(shear_moduli.size() == tau_factors.size(),
                  "Every region needs a shear modulus and a tau_factor.");
   }

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

   bool needs_hessian() const override { return true; }

   double get_disp_factor() const override { return disp_factor; }

   // Block 0 is the velocity.
   const mfem::Vector &set_element(const MaterialModel &material,
                                   const mfem::FiniteElement &elem_u,
                                   mfem::ElementTransformation &elem_map,
                                   const mfem::Vector &sol_u,
                                   const mfem::Vector &pres) override
   {
      elem_velo = &sol_u;
      elem_pres = &pres;
      rates = get_element_rates(elem_map, sol_u, pres);
      tau_J = get_tau_per_J_cached(material, elem_u, elem_map);
      return rates.disp;
   }

   void add_residual(const MaterialModel &material, const ShapeCache::PointShapes &point,
                     const Tensor2_3D &F, double p, mfem::Vector &residual_p) override
   {
      const mfem::Vector &M = point.M;
      const mfem::DenseMatrix &dM_dX = point.dM_dX;
      const int num_nodes_p = M.Size();
      const double dV = point.dV;

      const QuadPointState state = get_quad_point_state(material, point, F, p);
      const double tau = tau_J * state.J;

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

   // The derivatives with respect to v and p, with grad N_b = F^-T Grad N_b,
   // grad v the spatial velocity gradient and the factors c_u = disp_factor,
   // c_r = dot_factor:
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
   void add_tangent(const MaterialModel &material, const ShapeCache::PointShapes &point,
                    const Tensor2_3D &F, double p, const Tensor4_3D &CC_ich,
                    const Tensor4_3D &AA_ich, mfem::DenseMatrix &K_pu,
                    mfem::DenseMatrix &K_pp) override
   {
      const mfem::Vector &N = point.N;
      const mfem::DenseMatrix &dN_dX = point.dN_dX;
      const mfem::DenseMatrix &d2N_dX2 = point.d2N_dX2;
      const mfem::Vector &M = point.M;
      const mfem::DenseMatrix &dM_dX = point.dM_dX;
      const int num_nodes_u = N.Size();
      const int num_nodes_p = M.Size();
      const double dV = point.dV;
      const double rho_0 = material.get_rho_0();

      const QuadPointState state = get_quad_point_state(material, point, F, p);
      const Tensor2_3D &F_inv = state.F_inv;
      const double J = state.J;
      const double tau = tau_J * J;

      // sum_J dAA_ich[G^J]_kJlL, G^J_mM = u_m,MJ, stored as [l](k, L).
      const std::array<Tensor2_3D, 3> dAA_d2u =
         material.get_d_1st_elasticity_tensor_ich_sum(F, CC_ich,
                                                       get_hessian_slices(state.d2u_dX2));

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

private:
   // tau / J of the element of elem_map, computed at its first assembly.
   double get_tau_per_J_cached(const MaterialModel &material, const mfem::FiniteElement &elem,
                               mfem::ElementTransformation &elem_map)
   {
      const int element = elem_map.ElementNo;
      if (element >= static_cast<int>(tau_cache.size()))
         tau_cache.resize(element + 1, -1.0);
      if (tau_cache[element] < 0.0)
         tau_cache[element] = get_tau_per_J(material, elem, elem_map);
      return tau_cache[element];
   }

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
      Tensor2_3D F_inv;                    // F^-1
      double J;                            // det F
      Tensor2_3D dv_dx;                    // grad_l v_k, stored as (k, l)
      double div_velo;                     // div v = grad_k v_k
      Vector_3D dp_dx;                     // grad p
      double dot_pres;                     // dp/dt
      double dJ_dp;                        // J'(p)
      std::array<Tensor2_3D, 3> d2u_dX2;   // u_l,LJ, stored as [l](L, J)
      Vector_3D R_mom;                     // residual of the momentum equation
   };

   ElementRates get_element_rates(mfem::ElementTransformation &elem_map,
                                  const mfem::Vector &velo, const mfem::Vector &pres) const
   {
      MFEM_VERIFY(disp_offset != nullptr, "MassEquation_VMS needs set_rates.");
      ElementRates out;
      out.disp = get_elem_values(*disp_offset, elem_map);
      out.disp.Add(disp_factor, velo);
      out.dot_velo = get_elem_values(*dot_velo_offset, elem_map);
      out.dot_velo.Add(dot_factor, velo);
      out.dot_pres = get_elem_values(*dot_pres_offset, elem_map);
      out.dot_pres.Add(dot_factor, pres);
      return out;
   }

   // The state at the current point, with
   // R_mom_k = rho_0 dv_k/dt + J F^-1_Jk p,J - P_ich_kJ,J and
   // P_ich_kJ,J = AA_ich_kJlL u_l,LJ = sum_J dP_ich[G^J]_kJ, the derivatives of
   // P_ich along G^J_lL = u_l,LJ, without AA_ich.
   QuadPointState get_quad_point_state(const MaterialModel &material,
                                       const ShapeCache::PointShapes &point,
                                       const Tensor2_3D &F, double p) const
   {
      const mfem::Vector &N = point.N;
      const mfem::DenseMatrix &dN_dX = point.dN_dX;
      const mfem::DenseMatrix &d2N_dX2 = point.d2N_dX2;
      const mfem::Vector &M = point.M;
      const mfem::DenseMatrix &dM_dX = point.dM_dX;
      const mfem::Vector &velo = *elem_velo;
      const mfem::Vector &pres = *elem_pres;
      const int num_nodes_u = N.Size();
      const int num_nodes_p = M.Size();

      QuadPointState out;
      out.F_inv = F.inverse();
      out.J = F.det();

      // grad_l v_k = v_k,J F^-1_Jl
      const Tensor2_3D dv_dX = LocalAssemblyTools::get_deformation_gradient(velo, dN_dX)
                               - Tensor2_3D::identity();
      out.dv_dx = dv_dX * out.F_inv;
      out.div_velo = out.dv_dx.trace();

      // Grad p = M_c,J p_c, grad p = F^-T Grad p and dp/dt.
      Vector_3D dp_dX;
      out.dot_pres = 0.0;
      for (int cc = 0; cc < num_nodes_p; cc++)
      {
         for (int JJ = 0; JJ < 3; JJ++)
            dp_dX(JJ) += dM_dX(cc, JJ) * pres(cc);
         out.dot_pres += M(cc) * rates.dot_pres(cc);
      }
      out.dp_dx = get_spatial_gradient(dp_dX, out.F_inv);
      out.dJ_dp = material.get_dJ_dp(p);

      // dv_k/dt = N_a dv_ak/dt
      Vector_3D dot_velo;
      for (int aa = 0; aa < num_nodes_u; aa++)
         for (int kk = 0; kk < 3; kk++)
            dot_velo(kk) += N(aa) * rates.dot_velo(aa+kk*num_nodes_u);

      out.d2u_dX2 = get_disp_hessian(rates.disp, d2N_dX2);

      const double rho_0 = material.get_rho_0();
      for (int kk = 0; kk < 3; kk++)
         out.R_mom(kk) = rho_0 * dot_velo(kk) + out.J * out.dp_dx(kk);
      const std::array<Tensor2_3D, 3> dP_ich =
         material.get_d_1st_PK_stress_ich(F, get_hessian_slices(out.d2u_dX2));
      for (int JJ = 0; JJ < 3; JJ++)
         for (int kk = 0; kk < 3; kk++)
            out.R_mom(kk) -= dP_ich[JJ](kk, JJ);
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
   double get_tau_per_J(const MaterialModel &material, const mfem::FiniteElement &elem,
                        mfem::ElementTransformation &elem_map) const
   {
      const int region = (shear_moduli.size() == 1) ? 0 : elem_map.Attribute - 1;
      MFEM_VERIFY(region < static_cast<int>(shear_moduli.size()),
                  "No shear modulus for element attribute " << elem_map.Attribute << ".");
      const double shear_modulus = shear_moduli[region];
      const double rho_0 = material.get_rho_0();
      const double dJ_dp = material.get_dJ_dp(0.0);
      const double wave_modulus = (dJ_dp == 0.0) ? shear_modulus
                                                 : -1.0 / dJ_dp + 4.0 / 3.0 * shear_modulus;
      const int order = elem.GetOrder();
      return tau_factors[region] * get_elem_size(elem, elem_map)
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
                  "MassEquation_VMS supports hexahedra and tetrahedra.");
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

   const std::vector<double> shear_moduli;                // mu of tau, by region
   const std::vector<double> tau_factors;                 // tau = tau_factor dx J / (c rho_0 k^2), by region
   const mfem::GridFunction *disp_offset = nullptr;       // u = disp_offset + disp_factor v
   double disp_factor = 0.0;
   const mfem::GridFunction *dot_velo_offset = nullptr;   // dv/dt = dot_velo_offset + dot_factor v
   const mfem::GridFunction *dot_pres_offset = nullptr;   // dp/dt = dot_pres_offset + dot_factor p
   double dot_factor = 0.0;
   const mfem::Vector *elem_velo = nullptr;               // v of the current element
   const mfem::Vector *elem_pres = nullptr;               // p of the current element
   ElementRates rates;                                    // u, dv/dt and dp/dt of the current element
   double tau_J = 0.0;                                    // tau / J of the current element
   std::vector<double> tau_cache;                         // tau / J by element number, -1 until computed
};

#endif

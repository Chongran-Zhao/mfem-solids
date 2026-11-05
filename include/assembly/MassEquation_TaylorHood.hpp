// ============================================================================
// MassEquation_TaylorHood.hpp
//
// The mass equation of Taylor-Hood elements, the pressure one order below the
// displacement; block 0 is the displacement. It holds J = J(p) weakly,
//    R^c        = -int M_c (J - J(p)) dV,
//    K_pu(c, b l) = -int M_c J N_b,L F^-1_Ll dV, the transpose of K_up,
//    K_pp(c, d) = int dJ/dp M_c M_d dV.
// The volumetric model enters only through J(p), so the same equation covers
// compressible and fully incompressible materials. The tangent is symmetric.
//
// Author: Chongran Zhao
// Date: Nov. 5, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MASS_EQUATION_TAYLOR_HOOD_HPP
#define MASS_EQUATION_TAYLOR_HOOD_HPP

#include <mfem.hpp>

#include "IMassEquation.hpp"
#include "LocalAssemblyTools.hpp"

class MassEquation_TaylorHood : public IMassEquation
{
public:
   bool needs_hessian() const override { return false; }

   double get_disp_factor() const override { return 1.0; }

   const mfem::Vector &set_element(const MaterialModel &material,
                                   const mfem::FiniteElement &elem_u,
                                   mfem::ElementTransformation &elem_map,
                                   const mfem::Vector &sol_u,
                                   const mfem::Vector &pres) override
   {
      return sol_u;
   }

   void add_residual(const MaterialModel &material, const ShapeCache::PointShapes &point,
                     const Tensor2_3D &F, double p, mfem::Vector &residual_p) override
   {
      // J - J(p), zero where the volume ratio matches the pressure.
      const double vol_residual = F.det() - material.get_J(p);
      for (int cc = 0; cc < point.M.Size(); cc++)
         residual_p(cc) -= point.dV * point.M(cc) * vol_residual;
   }

   void add_tangent(const MaterialModel &material, const ShapeCache::PointShapes &point,
                    const Tensor2_3D &F, double p, const Tensor4_3D &CC_ich,
                    const Tensor4_3D &AA_ich, mfem::DenseMatrix &K_pu,
                    mfem::DenseMatrix &K_pp) override
   {
      const mfem::Vector &M = point.M;
      const double dV = point.dV;
      const int num_nodes_u = point.N.Size();
      const int num_nodes_p = M.Size();
      const double J = F.det();
      const Tensor2_3D F_inv = F.inverse();
      const double dJ_dp = material.get_dJ_dp(p);

      // K_pu
      for (int bb = 0; bb < num_nodes_u; bb++)
         for (int ll = 0; ll < 3; ll++)
         {
            // J N_b,L F^-1_Ll
            const double J_dNb_dx = LocalAssemblyTools::get_J_dNa_dx(point.dN_dX, bb, ll, J, F_inv);
            for (int cc = 0; cc < num_nodes_p; cc++)
               K_pu(cc, bb+ll*num_nodes_u) -= dV * J_dNb_dx * M(cc);
         }

      // K_pp
      for (int cc = 0; cc < num_nodes_p; cc++)
         for (int dd = 0; dd < num_nodes_p; dd++)
            K_pp(cc, dd) += dV * dJ_dp * M(cc) * M(dd);
   }
};

#endif

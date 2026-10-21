// ============================================================================
// LocalAssemblyTools.hpp
//
// Tools shared by the local assemblies: the quadrature rule and the
// deformation gradient at a quadrature point.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_TOOLS_HPP
#define LOCAL_ASSEMBLY_TOOLS_HPP

#include <mfem.hpp>

#include "Tensor2_3D.hpp"

class LocalAssemblyTools
{
public:
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

   // N_a,JL, the Hessian of the shape functions of elem with respect to X at
   // the point of elem_map, stored as {xx, xy, xz, yy, yz, zz} like
   // mfem::FiniteElement::CalcPhysHessian, which solves a 6 x 6 system at
   // every point instead. From the chain rule
   //    N_a,ij = N_a,JL X_J,i X_L,j + N_a,J X_J,ij,
   //    N_a,JL = (N_a,ij - N_a,J X_J,ij) xi_i,J xi_j,L,
   // with X_J,ij = X_gJ G_g,ij the Hessian of the geometry, G_g the shape
   // functions of the isoparametric transformation, and dN_dX = N_a,J given.
   // d2N_dxi2 and d2G_dxi2 are work matrices.
   static void calc_phys_hessian(const mfem::FiniteElement &elem,
                                 mfem::ElementTransformation &elem_map,
                                 const mfem::DenseMatrix &dN_dX, mfem::DenseMatrix &d2N_dxi2,
                                 mfem::DenseMatrix &d2G_dxi2, mfem::DenseMatrix &d2N_dX2)
   {
      const mfem::IntegrationPoint &quad_pt = elem_map.GetIntPoint();
      const mfem::IsoparametricTransformation &geometry =
         dynamic_cast<const mfem::IsoparametricTransformation &>(elem_map);
      const mfem::DenseMatrix &X_g = geometry.GetPointMat();
      const mfem::DenseMatrix &dxi_dX = elem_map.InverseJacobian();
      const int num_nodes = dN_dX.Height();
      const int num_geom_nodes = X_g.Width();

      // X_J,ij of the six pairs ij.
      d2G_dxi2.SetSize(num_geom_nodes, 6);
      geometry.GetFE()->CalcHessian(quad_pt, d2G_dxi2);
      double d2X_dxi2[3][6] = {};
      for (int gg = 0; gg < num_geom_nodes; gg++)
         for (int JJ = 0; JJ < 3; JJ++)
            for (int pair = 0; pair < 6; pair++)
               d2X_dxi2[JJ][pair] += X_g(JJ, gg) * d2G_dxi2(gg, pair);

      d2N_dxi2.SetSize(num_nodes, 6);
      elem.CalcHessian(quad_pt, d2N_dxi2);
      d2N_dX2.SetSize(num_nodes, 6);
      const int pair_of[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
      for (int aa = 0; aa < num_nodes; aa++)
      {
         // N_a,ij - N_a,J X_J,ij
         double pair_value[6];
         for (int pair = 0; pair < 6; pair++)
            pair_value[pair] = d2N_dxi2(aa, pair) - dN_dX(aa, 0) * d2X_dxi2[0][pair]
                               - dN_dX(aa, 1) * d2X_dxi2[1][pair]
                               - dN_dX(aa, 2) * d2X_dxi2[2][pair];

         // (...)_ij xi_j,L
         double right[3][3];
         for (int ii = 0; ii < 3; ii++)
            for (int LL = 0; LL < 3; LL++)
               right[ii][LL] = pair_value[pair_of[ii][0]] * dxi_dX(0, LL)
                               + pair_value[pair_of[ii][1]] * dxi_dX(1, LL)
                               + pair_value[pair_of[ii][2]] * dxi_dX(2, LL);

         // xi_i,J (...)_ij xi_j,L for the six pairs JL.
         for (int JJ = 0; JJ < 3; JJ++)
            for (int LL = JJ; LL < 3; LL++)
               d2N_dX2(aa, pair_of[JJ][LL]) = dxi_dX(0, JJ) * right[0][LL]
                                              + dxi_dX(1, JJ) * right[1][LL]
                                              + dxi_dX(2, JJ) * right[2][LL];
      }
   }
};

#endif

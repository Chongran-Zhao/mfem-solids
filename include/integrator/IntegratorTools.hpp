// ============================================================================
// IntegratorTools.hpp
//
// Tools shared by the element integrators: the quadrature rule and the
// deformation gradient at a quadrature point, and at the element center for
// the output of the stress.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef INTEGRATOR_TOOLS_HPP
#define INTEGRATOR_TOOLS_HPP

#include <mfem.hpp>
#include "Tensor2_3D.hpp"

class IntegratorTools
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

   // F at the center of element ee, for the output of the stress.
   static Tensor2_3D get_center_deformation_gradient(const mfem::FiniteElementSpace &fespace,
                                                     const mfem::GridFunction &disp, int ee)
   {
      const mfem::FiniteElement &elem = *fespace.GetFE(ee);
      mfem::ElementTransformation &elem_map = *fespace.GetElementTransformation(ee);
      const mfem::IntegrationPoint &center = mfem::Geometries.GetCenter(elem.GetGeomType());
      elem_map.SetIntPoint(&center);

      const int num_nodes = elem.GetDof();
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);
      elem.CalcDShape(center, dN_dxi);
      mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);

      // Element displacement: x of all nodes, then y, then z.
      mfem::Array<int> vdofs;
      mfem::Vector elem_disp;
      fespace.GetElementVDofs(ee, vdofs);
      disp.GetSubVector(vdofs, elem_disp);

      return get_deformation_gradient(elem_disp, dN_dX);
   }
};

#endif

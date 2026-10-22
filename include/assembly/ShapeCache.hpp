// ============================================================================
// ShapeCache.hpp
//
// The shape functions at the quadrature points of each element, in the
// reference configuration, which the Total Lagrangian form keeps for the
// whole run: computed at the first assembly of an element and kept, by
// element number, for the next residuals and tangents. Each point holds N,
// N_a,J and dV of the displacement (or velocity) space, its Hessian N_a,JL
// if asked, and M, M_c,J of the pressure space if there is one. The shapes
// of an element are computed again when the quadrature rule changes. The
// local assemblies own one each.
//
// Author: Chongran Zhao
// Date: Oct. 22, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef SHAPE_CACHE_HPP
#define SHAPE_CACHE_HPP

#include <vector>

#include <mfem.hpp>

#include "LocalAssemblyTools.hpp"

class ShapeCache
{
public:
   // The shape functions at a quadrature point.
   struct PointShapes
   {
      mfem::Vector N;              // N_a
      mfem::DenseMatrix dN_dX;     // N_a,J
      mfem::DenseMatrix d2N_dX2;   // N_a,JL as {xx, xy, xz, yy, yz, zz}, if asked
      mfem::Vector M;              // M_c of the pressure, if any
      mfem::DenseMatrix dM_dX;     // M_c,J of the pressure, if any
      double dV = 0.0;             // w_q det(dX/dxi)
   };

   // with_hessian: whether to keep N_a,JL.
   explicit ShapeCache(bool input_with_hessian) : with_hessian(input_with_hessian) {}

   // The shapes at the points of quad_rule on the element of elem_map, of
   // elem and, unless null, of the pressure element elem_p; computed once
   // per element and rule, quad_rule being one of mfem::IntRules.
   const std::vector<PointShapes> &get(const mfem::FiniteElement &elem,
                                       const mfem::FiniteElement *elem_p,
                                       mfem::ElementTransformation &elem_map,
                                       const mfem::IntegrationRule &quad_rule)
   {
      const int element = elem_map.ElementNo;
      if (element >= static_cast<int>(cache.size()))
         cache.resize(element + 1);
      ElementShapes &shapes = cache[element];
      std::vector<PointShapes> &points = shapes.points;
      if (shapes.quad_rule == &quad_rule)
         return points;
      shapes.quad_rule = &quad_rule;

      const int num_nodes = elem.GetDof();
      const int num_nodes_p = elem_p ? elem_p->GetDof() : 0;
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dM_dxi(num_nodes_p, 3);
      mfem::DenseMatrix d2N_dxi2, d2G_dxi2;   // work matrices of calc_phys_hessian
      points.resize(quad_rule.GetNPoints());
      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);
         PointShapes &point = points[qq];

         // N_a,J = dN/dX = dN/dxi (dX/dxi)^-1.
         point.N.SetSize(num_nodes);
         elem.CalcShape(quad_pt, point.N);
         elem.CalcDShape(quad_pt, dN_dxi);
         point.dN_dX.SetSize(num_nodes, 3);
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), point.dN_dX);
         if (with_hessian)
            LocalAssemblyTools::calc_phys_hessian(elem, elem_map, point.dN_dX, d2N_dxi2,
                                                  d2G_dxi2, point.d2N_dX2);
         if (elem_p)
         {
            point.M.SetSize(num_nodes_p);
            elem_p->CalcShape(quad_pt, point.M);
            elem_p->CalcDShape(quad_pt, dM_dxi);
            point.dM_dX.SetSize(num_nodes_p, 3);
            mfem::Mult(dM_dxi, elem_map.InverseJacobian(), point.dM_dX);
         }
         // dV = w_q * det( dX/dxi )
         point.dV = quad_pt.weight * elem_map.Weight();
      }
      return points;
   }

private:
   // The shapes at the points of an element and their rule.
   struct ElementShapes
   {
      const mfem::IntegrationRule *quad_rule = nullptr;
      std::vector<PointShapes> points;
   };

   const bool with_hessian;            // keep N_a,JL
   std::vector<ElementShapes> cache;   // by element number
};

#endif

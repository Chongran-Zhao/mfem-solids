// ============================================================================
// LocalAssembly_Fracture.hpp
//
// Local (element) assembly of the phase-field problem of fracture, at a fixed
// displacement: the element energy, residual and tangent of
//    E(alpha) = int a(alpha) W + C1 w(alpha) + C2/2 |Grad alpha|^2 dV,
// with C1 = G_c / (4 c_w ell) and C2 = G_c ell / (2 c_w), the phase field
// alpha, the damage of mef90, the unknown and W the elastic energy density at
// the fixed displacement, as in
// MEF90DefMechOperatorDamage and MEF90DefMechBilinearFormDamage of mef90.
// It is an MFEM NonlinearFormIntegrator, called by NonlinearForm, which
// assembles the global residual and tangent.
//
// Author: Chongran Zhao
// Date: Nov. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_FRACTURE_HPP
#define LOCAL_ASSEMBLY_FRACTURE_HPP

#include <memory>
#include <utility>
#include <vector>

#include <mfem.hpp>

#include "ISurfaceEnergy.hpp"
#include "LocalAssemblyTools.hpp"
#include "ShapeCache.hpp"

class LocalAssembly_Fracture : public mfem::NonlinearFormIntegrator
{
public:
   // Takes the ownership of the surface energy; input_W, the elastic energy
   // density, stays the caller's and is read at every assembly.
   LocalAssembly_Fracture(std::unique_ptr<const ISurfaceEnergy> input_surface_energy,
                          mfem::Coefficient &input_W)
      : surface_energy(std::move(input_surface_energy)),
        C1(surface_energy->get_G_c() / (4.0 * surface_energy->get_c_w() * surface_energy->get_ell())),
        C2(surface_energy->get_G_c() * surface_energy->get_ell() / (2.0 * surface_energy->get_c_w())),
        W(input_W) {}

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // GetElementEnergy.
   // int a(alpha) W + C1 w(alpha) + C2/2 |Grad alpha|^2 dV over the element.
   double GetElementEnergy(const mfem::FiniteElement &elem,
                           mfem::ElementTransformation &elem_map,
                           const mfem::Vector &alpha_elem) override
   {
      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem, elem_map);
      const std::vector<ShapeCache::PointShapes> &points =
         shape_cache.get(elem, nullptr, elem_map, quad_rule);

      double out = 0.0;
      mfem::Vector grad_alpha(3);
      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const ShapeCache::PointShapes &point = points[qq];
         const double alpha = point.N * alpha_elem;
         point.dN_dX.MultTranspose(alpha_elem, grad_alpha);

         out += point.dV * (surface_energy->get_a(alpha) * get_W(elem_map, quad_rule, qq)
                            + C1 * surface_energy->get_w(alpha)
                            + 0.5 * C2 * (grad_alpha * grad_alpha));
      }
      return out;
   }

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // AssembleElementVector, which NonlinearForm calls on every element.
   // a is the node index.
   // R^a = int (a'(alpha) W + C1 w'(alpha)) N_a + C2 Grad alpha . N_a,J dV.
   void AssembleElementVector(const mfem::FiniteElement &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Vector &alpha_elem,
                              mfem::Vector &residual) override
   {
      const int num_nodes = elem.GetDof();
      residual.SetSize(num_nodes);
      residual = 0.0;

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem, elem_map);
      const std::vector<ShapeCache::PointShapes> &points =
         shape_cache.get(elem, nullptr, elem_map, quad_rule);

      mfem::Vector grad_alpha(3);
      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const ShapeCache::PointShapes &point = points[qq];
         const mfem::DenseMatrix &dN_dX = point.dN_dX;
         const double alpha = point.N * alpha_elem;
         dN_dX.MultTranspose(alpha_elem, grad_alpha);

         const double C3 = surface_energy->get_da(alpha) * get_W(elem_map, quad_rule, qq)
                           + C1 * surface_energy->get_dw(alpha);

         for (int aa = 0; aa < num_nodes; aa++)
            residual(aa) += point.dV * (C3 * point.N(aa)
                                        + C2 * (dN_dX(aa, 0) * grad_alpha(0)
                                                + dN_dX(aa, 1) * grad_alpha(1)
                                                + dN_dX(aa, 2) * grad_alpha(2)));
      }
   }

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // AssembleElementGrad, which NonlinearForm calls on every element.
   // K^ab = int (a''(alpha) W + C1 w''(alpha)) N_a N_b + C2 N_a,J N_b,J dV.
   void AssembleElementGrad(const mfem::FiniteElement &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Vector &alpha_elem,
                            mfem::DenseMatrix &tangent) override
   {
      const int num_nodes = elem.GetDof();
      tangent.SetSize(num_nodes);
      tangent = 0.0;

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem, elem_map);
      const std::vector<ShapeCache::PointShapes> &points =
         shape_cache.get(elem, nullptr, elem_map, quad_rule);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const ShapeCache::PointShapes &point = points[qq];
         const mfem::DenseMatrix &dN_dX = point.dN_dX;
         const double alpha = point.N * alpha_elem;

         const double C3 = surface_energy->get_d2a(alpha) * get_W(elem_map, quad_rule, qq)
                           + C1 * surface_energy->get_d2w(alpha);

         for (int bb = 0; bb < num_nodes; bb++)
            for (int aa = 0; aa < num_nodes; aa++)
               tangent(aa, bb) += point.dV * (C3 * point.N(aa) * point.N(bb)
                                              + C2 * (dN_dX(aa, 0) * dN_dX(bb, 0)
                                                      + dN_dX(aa, 1) * dN_dX(bb, 1)
                                                      + dN_dX(aa, 2) * dN_dX(bb, 2)));
      }
   }

   // Quadrature order of the elements; 0, the default, keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   void set_quad_order(int input_quad_order) { quad_order = input_quad_order; }

private:
   // The rule of order quad_order, or that of LocalAssemblyTools.
   const mfem::IntegrationRule &get_quad_rule(const mfem::FiniteElement &elem,
                                              mfem::ElementTransformation &elem_map) const
   {
      return (quad_order > 0) ? mfem::IntRules.Get(elem.GetGeomType(), quad_order)
                              : LocalAssemblyTools::get_quad_rule(elem, elem_map);
   }

   // W at the quadrature point qq of the element.
   double get_W(mfem::ElementTransformation &elem_map, const mfem::IntegrationRule &quad_rule,
                int qq) const
   {
      const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
      elem_map.SetIntPoint(&quad_pt);
      return W.Eval(elem_map, quad_pt);
   }

   const std::unique_ptr<const ISurfaceEnergy> surface_energy;
   const double C1;                 // G_c / (4 c_w ell)
   const double C2;                 // G_c ell / (2 c_w)
   mfem::Coefficient &W;            // elastic energy density at the fixed displacement
   int quad_order = 0;              // quadrature order, 0 for that of LocalAssemblyTools
   ShapeCache shape_cache{false};   // N_a, N_a,J and dV of each element
};

#endif

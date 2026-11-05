// ============================================================================
// LocalAssembly_Mixed.hpp
//
// Local (element) assembly of hyperelasticity in the mixed displacement-
// pressure (u/p) form, Total Lagrangian: the element residual and tangent.
// The momentum equation, the same in all the formulations, is assembled here,
//    R^a_k = int N_a,J (P_ich_kJ - p J F^-1_Jk) dV,
//    K_uu(a k, b l) = c_u int N_a,J AA_kJlL N_b,L dV,
//       AA_kJlL = AA_ich_kJlL - p J (F^-1_Jk F^-1_Ll - F^-1_Jl F^-1_Lk),
//    K_up(a k, d)   = -int J N_a,J F^-1_Jk M_d dV,
// with c_u = du/dsol_u, and the mass equation, R^c, K_pu and K_pp, by the
// IMassEquation it owns, which is what sets the formulation. Inertia is left
// to the nonlinear solvers. It is an MFEM BlockNonlinearFormIntegrator, called
// by BlockNonlinearForm, which assembles the global residual and tangent.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_MIXED_HPP
#define LOCAL_ASSEMBLY_MIXED_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "IMassEquation.hpp"
#include "LocalAssemblyTools.hpp"
#include "MaterialModel.hpp"
#include "ShapeCache.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class LocalAssembly_Mixed : public mfem::BlockNonlinearFormIntegrator
{
public:
   // Takes ownership of the material, as in LocalAssembly_Disp, and of the
   // mass equation.
   LocalAssembly_Mixed(std::unique_ptr<const MaterialModel> input_material,
                       std::unique_ptr<IMassEquation> input_mass_equation)
      : material(std::move(input_material)),
        mass_equation(std::move(input_mass_equation)),
        shape_cache(mass_equation->needs_hessian()) {}

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementVector, which BlockNonlinearForm calls on every element.
   // Block 0 is the displacement or the velocity, as the mass equation sets
   // it, block 1 the pressure.
   void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Array<const mfem::Vector *> &elem_sol,
                              const mfem::Array<mfem::Vector *> &residual) override
   {
      const mfem::FiniteElement &elem_u = *elem[0];
      const mfem::FiniteElement &elem_p = *elem[1];
      const mfem::Vector &pres = *elem_sol[1];
      mfem::Vector &residual_u = *residual[0];
      mfem::Vector &residual_p = *residual[1];

      const MaterialModel &elem_material = get_material(elem_map);
      const mfem::Vector &disp = mass_equation->set_element(elem_material, elem_u, elem_map,
                                                            *elem_sol[0], pres);

      residual_u.SetSize(3 * elem_u.GetDof());
      residual_u = 0.0;
      residual_p.SetSize(elem_p.GetDof());
      residual_p = 0.0;

      // N_a,J, M_c and dV at the points, kept from the first assembly.
      for (const ShapeCache::PointShapes &point :
           shape_cache.get(elem_u, &elem_p, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp, point.dN_dX);

         // Vector * Vector is the inner product in MFEM.
         const double p = point.M * pres;

         add_momentum_residual(point.dN_dX, get_1st_PK_stress(elem_material, F, p), point.dV,
                               residual_u);
         mass_equation->add_residual(elem_material, point, F, p, residual_p);
      }
   }

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementGrad, which BlockNonlinearForm calls on every element.
   void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Array<const mfem::Vector *> &elem_sol,
                            const mfem::Array2D<mfem::DenseMatrix *> &tangent) override
   {
      const mfem::FiniteElement &elem_u = *elem[0];
      const mfem::FiniteElement &elem_p = *elem[1];
      const mfem::Vector &pres = *elem_sol[1];
      mfem::DenseMatrix &K_uu = *tangent(0, 0);
      mfem::DenseMatrix &K_up = *tangent(0, 1);
      mfem::DenseMatrix &K_pu = *tangent(1, 0);
      mfem::DenseMatrix &K_pp = *tangent(1, 1);

      const int num_nodes_u = elem_u.GetDof();
      const int num_nodes_p = elem_p.GetDof();

      const MaterialModel &elem_material = get_material(elem_map);
      const mfem::Vector &disp = mass_equation->set_element(elem_material, elem_u, elem_map,
                                                            *elem_sol[0], pres);
      const double disp_factor = mass_equation->get_disp_factor();

      K_uu.SetSize(3 * num_nodes_u, 3 * num_nodes_u);
      K_uu = 0.0;
      K_up.SetSize(3 * num_nodes_u, num_nodes_p);
      K_up = 0.0;
      K_pu.SetSize(num_nodes_p, 3 * num_nodes_u);
      K_pu = 0.0;
      K_pp.SetSize(num_nodes_p, num_nodes_p);
      K_pp = 0.0;

      // N_a,J, M_c and dV at the points, kept from the first assembly.
      for (const ShapeCache::PointShapes &point :
           shape_cache.get(elem_u, &elem_p, elem_map, get_quad_rule(elem_u, elem_map)))
      {
         const mfem::DenseMatrix &dN_dX = point.dN_dX;
         const mfem::Vector &M = point.M;
         const double dV = point.dV;

         const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(disp, dN_dX);
         const double J = F.det();
         const Tensor2_3D F_inv = F.inverse();

         // Vector * Vector is the inner product in MFEM.
         const double p = M * pres;

         // CC_ich and AA_ich, once for the point.
         const Tensor4_3D CC_ich = elem_material.get_2nd_elasticity_tensor_ich(F);
         const Tensor4_3D AA_ich = elem_material.get_1st_elasticity_tensor_ich(F, CC_ich);

         // K_uu
         LocalAssemblyTools::add_momentum_tangent_uu(
            dN_dX, get_1st_elasticity_tensor(F, p, AA_ich), disp_factor * dV, K_uu);

         // K_up
         for (int aa = 0; aa < num_nodes_u; aa++)
            for (int kk = 0; kk < 3; kk++)
            {
               const double J_dNa_dx = LocalAssemblyTools::get_J_dNa_dx(dN_dX, aa, kk, J, F_inv);
               for (int dd = 0; dd < num_nodes_p; dd++)
                  K_up(aa+kk*num_nodes_u, dd) -= dV * J_dNa_dx * M(dd);
            }

         // K_pu and K_pp
         mass_equation->add_tangent(elem_material, point, F, p, CC_ich, AA_ich, K_pu, K_pp);
      }
   }

   // Reference density rho_0 of the material, for the mass matrix.
   double get_rho_0() const { return material->get_rho_0(); }

   // Quadrature order of the elements; 0, the default, keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   void set_quad_order(int input_quad_order) { quad_order = input_quad_order; }

protected:
   // The material of the element of elem_map.
   virtual const MaterialModel &get_material(mfem::ElementTransformation &elem_map) const
   {
      return *material;
   }

   const std::unique_ptr<const MaterialModel> material;

private:
   // The rule of order quad_order, or that of LocalAssemblyTools.
   const mfem::IntegrationRule &get_quad_rule(const mfem::FiniteElement &elem,
                                              mfem::ElementTransformation &elem_map) const
   {
      return (quad_order > 0) ? mfem::IntRules.Get(elem.GetGeomType(), quad_order)
                              : LocalAssemblyTools::get_quad_rule(elem, elem_map);
   }

   // P = P_ich - p J F^-T
   static Tensor2_3D get_1st_PK_stress(const MaterialModel &elem_material, const Tensor2_3D &F,
                                       double p)
   {
      return elem_material.get_1st_PK_stress_ich(F) - p * F.det() * F.inverse().transpose();
   }

   // AA = AA_ich - p d(J F^-T)/dF,
   // AA_kJlL = AA_ich_kJlL - p J (F^-1_Jk F^-1_Ll - F^-1_Jl F^-1_Lk).
   static Tensor4_3D get_1st_elasticity_tensor(const Tensor2_3D &F, double p, Tensor4_3D AA)
   {
      const double J = F.det();
      const Tensor2_3D F_inv = F.inverse();
      for (int kk = 0; kk < 3; kk++)
         for (int JJ = 0; JJ < 3; JJ++)
            for (int ll = 0; ll < 3; ll++)
               for (int LL = 0; LL < 3; LL++)
                  AA(kk, JJ, ll, LL) -= p * J * (F_inv(JJ, kk) * F_inv(LL, ll)
                                               - F_inv(JJ, ll) * F_inv(LL, kk));
      return AA;
   }

   // R^a_k += dV N_a,J P_kJ
   static void add_momentum_residual(const mfem::DenseMatrix &dN_dX, const Tensor2_3D &PK1,
                                     double dV, mfem::Vector &residual_u)
   {
      const int num_nodes_u = dN_dX.Height();
      for (int aa = 0; aa < num_nodes_u; aa++)
      {
         // x-component
         residual_u(aa)               += dV * (PK1(0, 0) * dN_dX(aa, 0)
                                             + PK1(0, 1) * dN_dX(aa, 1)
                                             + PK1(0, 2) * dN_dX(aa, 2));
         // y-component
         residual_u(aa+num_nodes_u)   += dV * (PK1(1, 0) * dN_dX(aa, 0)
                                             + PK1(1, 1) * dN_dX(aa, 1)
                                             + PK1(1, 2) * dN_dX(aa, 2));
         // z-component
         residual_u(aa+2*num_nodes_u) += dV * (PK1(2, 0) * dN_dX(aa, 0)
                                             + PK1(2, 1) * dN_dX(aa, 1)
                                             + PK1(2, 2) * dN_dX(aa, 2));
      }
   }

   const std::unique_ptr<IMassEquation> mass_equation;
   int quad_order = 0;       // quadrature order, 0 for that of LocalAssemblyTools
   ShapeCache shape_cache;   // N_a,J, M_c and dV of each element
};

#endif

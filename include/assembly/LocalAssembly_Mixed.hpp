// ============================================================================
// LocalAssembly_Mixed.hpp
//
// Base of the local (element) assemblies of hyperelasticity in the mixed
// displacement-pressure (u/p) form, Total Lagrangian: the momentum equation,
//    R^a_k = int N_a,J (P_ich_kJ - p J F^-1_Jk) dV,
// and its tangent, which the formulations share, added at a quadrature point
// by the functions below and LocalAssemblyTools::add_momentum_tangent_uu; the
// material, the quadrature rule and the density.
// The derived classes give the mass equation and assemble the element:
// LocalAssembly_Mixed_TaylorHood, the weak J = J(p) of Taylor-Hood elements,
// and LocalAssembly_Mixed_VMS, its rate form with the VMS stabilization of
// equal-order elements. It is an MFEM BlockNonlinearFormIntegrator, called by
// BlockNonlinearForm, which assembles the global residual and tangent.
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

#include "LocalAssemblyTools.hpp"
#include "MaterialModel.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class LocalAssembly_Mixed : public mfem::BlockNonlinearFormIntegrator
{
public:
   // Takes ownership of the material, as in LocalAssembly_Disp.
   LocalAssembly_Mixed(std::unique_ptr<const MaterialModel> input_material)
      : material(std::move(input_material)) {}

   // Required by MFEM: overrides mfem::BlockNonlinearFormIntegrator::
   // AssembleElementVector and AssembleElementGrad, given by each
   // formulation. Block 0 is the displacement (or the velocity), block 1 the
   // pressure.
   void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Array<const mfem::Vector *> &elem_sol,
                              const mfem::Array<mfem::Vector *> &residual) override = 0;
   void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Array<const mfem::Vector *> &elem_sol,
                            const mfem::Array2D<mfem::DenseMatrix *> &tangent) override = 0;

   // Reference density rho_0 of the material, for the mass matrix.
   double get_rho_0() const { return material->get_rho_0(); }

   // Quadrature order of the elements; 0, the default, keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   void set_quad_order(int input_quad_order) { quad_order = input_quad_order; }

protected:
   // The rule of order quad_order, or that of LocalAssemblyTools.
   const mfem::IntegrationRule &get_quad_rule(const mfem::FiniteElement &elem,
                                              mfem::ElementTransformation &elem_map) const
   {
      return (quad_order > 0) ? mfem::IntRules.Get(elem.GetGeomType(), quad_order)
                              : LocalAssemblyTools::get_quad_rule(elem, elem_map);
   }

   // P = P_ich - p J F^-T
   Tensor2_3D get_1st_PK_stress(const Tensor2_3D &F, double p) const
   {
      return material->get_1st_PK_stress_ich(F) - p * F.det() * F.inverse().transpose();
   }

   // AA = AA_ich - p d(J F^-T)/dF,
   // AA_kJlL = AA_ich_kJlL - p J (F^-1_Jk F^-1_Ll - F^-1_Jl F^-1_Lk).
   Tensor4_3D get_1st_elasticity_tensor(const Tensor2_3D &F, double p) const
   {
      return get_1st_elasticity_tensor(F, p, material->get_1st_elasticity_tensor_ich(F));
   }

   // AA from AA_ich computed elsewhere.
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

   // J N_a,J F^-1_Jk, the factor of K_up(a k, d) = -int J N_a,J F^-1_Jk M_d dV,
   // which is also that of its transpose K_pu of Taylor-Hood.
   static double get_J_dNa_dx(const mfem::DenseMatrix &dN_dX, int aa, int kk, double J,
                              const Tensor2_3D &F_inv)
   {
      return J * (dN_dX(aa, 0) * F_inv(0, kk) + dN_dX(aa, 1) * F_inv(1, kk)
                + dN_dX(aa, 2) * F_inv(2, kk));
   }

   const std::unique_ptr<const MaterialModel> material;
   int quad_order = 0;   // quadrature order, 0 for that of LocalAssemblyTools
};

#endif

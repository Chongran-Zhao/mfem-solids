// ============================================================================
// LocalAssembly_Fracture_Regions.hpp
//
// Local (element) assembly of the phase-field problem of fracture at a fixed
// displacement, as LocalAssembly_Fracture, with one material per region of
// the mesh, each breakable or not: the element energy, residual and tangent.
// In a breakable region the elastic energy psi_e(F) of the material of the
// region is degraded by g(alpha) and drives the phase field. In a region that
// does not break, as the regions with elastic: true in mef90
// (m_MEF90_DefMechAssembly.F90), the elastic energy is not degraded and does
// not drive the phase field, which keeps the surface terms there: alpha stays
// free, governed by the surface energy only, so that a crack may reach the
// boundary of the region. The element energy of such a region counts psi_e,
// which does not depend on alpha, so that the energy is that of the
// displacement problem plus the surface energy. The unknown is the phase
// field alpha; F = I + Grad u comes from the displacement, which it reads at
// every assembly, and g, w and c_grad from the surface energy. It is an MFEM
// NonlinearFormIntegrator, called by NonlinearForm, which assembles the global
// energy, residual and tangent. The tangent is symmetric.
//
// Author: Chongran Zhao
// Date: Nov. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_FRACTURE_REGIONS_HPP
#define LOCAL_ASSEMBLY_FRACTURE_REGIONS_HPP

#include <memory>
#include <utility>
#include <vector>

#include <mfem.hpp>

#include "IMaterialModel.hpp"
#include "ISurfaceEnergy.hpp"
#include "LocalAssemblyTools.hpp"
#include "Tensor2_3D.hpp"

class LocalAssembly_Fracture_Regions : public mfem::NonlinearFormIntegrator
{
public:
   // Takes the ownership of the surface energy and of the materials;
   // materials[aa] is that of element attribute aa + 1, and breakable[aa]
   // whether it breaks; input_disp, the displacement on the same mesh, stays
   // the caller's.
   LocalAssembly_Fracture_Regions(std::unique_ptr<const ISurfaceEnergy> input_surface_energy,
                                  std::vector<std::unique_ptr<const IMaterialModel>> input_materials,
                                  const std::vector<bool> &input_breakable,
                                  const mfem::GridFunction &input_disp)
      : surface_energy(std::move(input_surface_energy)),
        materials(std::move(input_materials)),
        breakable(input_breakable),
        disp(input_disp)
   {
      MFEM_VERIFY(materials.size() == breakable.size(),
                  "One breakable flag per material is needed.");
   }

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // GetElementEnergy, which NonlinearForm::GetEnergy calls on every element.
   // E = int g(alpha) psi_e + G_c ( w(alpha) + c_grad alpha,J alpha,J ) dV,
   // with psi_e in place of g(alpha) psi_e where the region does not break.
   double GetElementEnergy(const mfem::FiniteElement &elem,
                           mfem::ElementTransformation &elem_map,
                           const mfem::Vector &phase) override
   {
      const int num_nodes = elem.GetDof();
      mfem::Vector N(num_nodes), dalpha_dX(3), phase_minus_1(phase);
      phase_minus_1 -= 1.0;
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);

      const double G_c = surface_energy->get_G_c();
      const double c_grad = surface_energy->get_c_grad();
      const bool is_breakable = get_breakable(elem_map.Attribute);

      double energy = 0.0;

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);

         // N_c
         elem.CalcShape(quad_pt, N);
         // dN/dxi
         elem.CalcDShape(quad_pt, dN_dxi);
         // N_c,J = dN/dX = dN/dxi (dX/dxi)^-1.
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);

         // Interpolate the phase field alpha = N_c alpha_c;
         // Vector * Vector is the inner product in MFEM.
         const double alpha = N * phase;
         // alpha,J = N_c,J alpha_c = N_c,J (alpha_c - 1), the shapes summing
         // to 1; from alpha_c - 1 it is exactly 0 where alpha = 1, free of
         // round-off, so that the bound alpha <= 1 of a VI solver does not
         // hold the sound material by the sign of a round-off residual.
         dN_dX.MultTranspose(phase_minus_1, dalpha_dX);

         const double psi_e = get_elastic_energy(elem_map);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // g(alpha) psi_e, or psi_e where the region does not break
         const double elastic = is_breakable ? surface_energy->get_g(alpha) * psi_e : psi_e;

         energy += dV * (elastic
                         + G_c * (surface_energy->get_w(alpha)
                                  + c_grad * (dalpha_dX * dalpha_dX)));
      }
      return energy;
   }

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // AssembleElementVector, which NonlinearForm calls on every element.
   // c is the node index.
   // R^c = int ( g'(alpha) psi_e + G_c w'(alpha) ) N_c
   //           + 2 G_c c_grad alpha,J N_c,J dV,
   // without g'(alpha) psi_e where the region does not break.
   void AssembleElementVector(const mfem::FiniteElement &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Vector &phase,
                              mfem::Vector &residual) override
   {
      const int num_nodes = elem.GetDof();
      mfem::Vector N(num_nodes), dalpha_dX(3), phase_minus_1(phase);
      phase_minus_1 -= 1.0;
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);

      const double G_c = surface_energy->get_G_c();
      const double c_grad = surface_energy->get_c_grad();
      const bool is_breakable = get_breakable(elem_map.Attribute);

      residual.SetSize(num_nodes);
      residual = 0.0;

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);

         // N_c
         elem.CalcShape(quad_pt, N);
         // dN/dxi
         elem.CalcDShape(quad_pt, dN_dxi);
         // N_c,J = dN/dX = dN/dxi (dX/dxi)^-1.
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);

         // Interpolate the phase field alpha = N_c alpha_c;
         // Vector * Vector is the inner product in MFEM.
         const double alpha = N * phase;
         // alpha,J = N_c,J alpha_c = N_c,J (alpha_c - 1), the shapes summing
         // to 1; from alpha_c - 1 it is exactly 0 where alpha = 1, free of
         // round-off, so that the bound alpha <= 1 of a VI solver does not
         // hold the sound material by the sign of a round-off residual.
         dN_dX.MultTranspose(phase_minus_1, dalpha_dX);

         // g'(alpha) psi_e where the region breaks, 0 elsewhere
         const double driving =
            is_breakable ? surface_energy->get_dg(alpha) * get_elastic_energy(elem_map) : 0.0;

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // g'(alpha) psi_e + G_c w'(alpha)
         const double dE_dalpha = driving + G_c * surface_energy->get_dw(alpha);

         for (int cc = 0; cc < num_nodes; cc++)
            residual(cc) += dV * (dE_dalpha * N(cc)
                                  + 2.0 * G_c * c_grad * (dalpha_dX(0) * dN_dX(cc, 0)
                                                          + dalpha_dX(1) * dN_dX(cc, 1)
                                                          + dalpha_dX(2) * dN_dX(cc, 2)));
      }
   }

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // AssembleElementGrad, which NonlinearForm calls on every element.
   // K^cd = int ( g''(alpha) psi_e + G_c w''(alpha) ) N_c N_d
   //            + 2 G_c c_grad N_c,J N_d,J dV,
   // without g''(alpha) psi_e where the region does not break, stored at
   // tangent(cc, dd).
   void AssembleElementGrad(const mfem::FiniteElement &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Vector &phase,
                            mfem::DenseMatrix &tangent) override
   {
      const int num_nodes = elem.GetDof();
      mfem::Vector N(num_nodes);
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);

      const double G_c = surface_energy->get_G_c();
      const double c_grad = surface_energy->get_c_grad();
      const bool is_breakable = get_breakable(elem_map.Attribute);

      tangent.SetSize(num_nodes);
      tangent = 0.0;

      const mfem::IntegrationRule &quad_rule = get_quad_rule(elem, elem_map);

      for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
      {
         const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
         elem_map.SetIntPoint(&quad_pt);

         // N_c
         elem.CalcShape(quad_pt, N);
         // dN/dxi
         elem.CalcDShape(quad_pt, dN_dxi);
         // N_c,J = dN/dX = dN/dxi (dX/dxi)^-1.
         mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);

         // Interpolate the phase field alpha = N_c alpha_c;
         // Vector * Vector is the inner product in MFEM.
         const double alpha = N * phase;

         // g''(alpha) psi_e where the region breaks, 0 elsewhere
         const double driving =
            is_breakable ? surface_energy->get_d2g(alpha) * get_elastic_energy(elem_map) : 0.0;

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // g''(alpha) psi_e + G_c w''(alpha)
         const double d2E_dalpha2 = driving + G_c * surface_energy->get_d2w(alpha);

         for (int cc = 0; cc < num_nodes; cc++)
            for (int dd = 0; dd < num_nodes; dd++)
               tangent(cc, dd) += dV * (d2E_dalpha2 * N(cc) * N(dd)
                                        + 2.0 * G_c * c_grad * (dN_dX(cc, 0) * dN_dX(dd, 0)
                                                                + dN_dX(cc, 1) * dN_dX(dd, 1)
                                                                + dN_dX(cc, 2) * dN_dX(dd, 2)));
      }
   }

   // Quadrature order of the elements; 0, the default, keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   void set_quad_order(int input_quad_order) { quad_order = input_quad_order; }

private:
   // Whether the region of an element attribute breaks.
   bool get_breakable(int attribute) const
   {
      MFEM_VERIFY(attribute >= 1 && attribute <= static_cast<int>(breakable.size()),
                  "No material for element attribute " << attribute << ".");
      return breakable[attribute - 1];
   }

   // psi_e(F) of the material of the element attribute at the point set in
   // elem_map, with F_kJ = delta_kJ + u_k,J from the displacement.
   double get_elastic_energy(mfem::ElementTransformation &elem_map)
   {
      disp.GetVectorGradient(elem_map, grad_u);
      Tensor2_3D F = Tensor2_3D::identity();
      for (int kk = 0; kk < 3; kk++)
         for (int JJ = 0; JJ < 3; JJ++)
            F(kk, JJ) += grad_u(kk, JJ);
      return materials[elem_map.Attribute - 1]->get_strain_energy(F);
   }

   // The rule of order quad_order, or that of LocalAssemblyTools.
   const mfem::IntegrationRule &get_quad_rule(const mfem::FiniteElement &elem,
                                              mfem::ElementTransformation &elem_map) const
   {
      return (quad_order > 0) ? mfem::IntRules.Get(elem.GetGeomType(), quad_order)
                              : LocalAssemblyTools::get_quad_rule(elem, elem_map);
   }

   const std::unique_ptr<const ISurfaceEnergy> surface_energy;
   const std::vector<std::unique_ptr<const IMaterialModel>> materials;   // by element attribute
   const std::vector<bool> breakable;   // whether the region of each attribute breaks
   const mfem::GridFunction &disp;      // displacement, not owned
   mfem::DenseMatrix grad_u;            // u_k,J at the point
   int quad_order = 0;   // quadrature order, 0 for that of LocalAssemblyTools
};

#endif

// ============================================================================
// LocalAssembly_Fracture.hpp
//
// Local (element) assembly of the phase-field problem of fracture at a fixed
// displacement, following the notes Phase-field-fracture.pdf (p. 4): the
// element energy, residual and tangent. The unknown is the phase field alpha;
// the elastic energy psi_e(F), F = I + Grad u, comes from the displacement,
// which it reads at every assembly, and g, w and c_grad from the surface
// energy. It is an MFEM NonlinearFormIntegrator, called by NonlinearForm,
// which assembles the global energy, residual and tangent. The tangent is
// symmetric.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_FRACTURE_HPP
#define LOCAL_ASSEMBLY_FRACTURE_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "IMaterialModel.hpp"
#include "ISurfaceEnergy.hpp"
#include "LocalAssemblyTools.hpp"
#include "Tensor2_3D.hpp"

class LocalAssembly_Fracture : public mfem::NonlinearFormIntegrator
{
public:
   // Takes the ownership of the surface energy and of the material;
   // input_disp, the displacement on the same mesh, stays the caller's.
   LocalAssembly_Fracture(std::unique_ptr<const ISurfaceEnergy> input_surface_energy,
                            std::unique_ptr<const IMaterialModel> input_material,
                            const mfem::GridFunction &input_disp)
      : surface_energy(std::move(input_surface_energy)),
        material(std::move(input_material)),
        disp(input_disp) {}

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // GetElementEnergy, which NonlinearForm::GetEnergy calls on every element.
   // E = int g(alpha) psi_e + G_c ( w(alpha) + c_grad alpha,J alpha,J ) dV.
   double GetElementEnergy(const mfem::FiniteElement &elem,
                           mfem::ElementTransformation &elem_map,
                           const mfem::Vector &phase) override
   {
      const int num_nodes = elem.GetDof();
      mfem::Vector N(num_nodes), dalpha_dX(3);
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);

      const double G_c = surface_energy->get_G_c();
      const double c_grad = surface_energy->get_c_grad();

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
         // alpha,J = N_c,J alpha_c
         dN_dX.MultTranspose(phase, dalpha_dX);

         const double psi_e = get_elastic_energy(elem_map);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         energy += dV * (surface_energy->get_g(alpha) * psi_e
                         + G_c * (surface_energy->get_w(alpha)
                                  + c_grad * (dalpha_dX * dalpha_dX)));
      }
      return energy;
   }

   // Required by MFEM: overrides mfem::NonlinearFormIntegrator::
   // AssembleElementVector, which NonlinearForm calls on every element.
   // c is the node index.
   // R^c = int ( g'(alpha) psi_e + G_c w'(alpha) ) N_c
   //           + 2 G_c c_grad alpha,J N_c,J dV.
   void AssembleElementVector(const mfem::FiniteElement &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Vector &phase,
                              mfem::Vector &residual) override
   {
      const int num_nodes = elem.GetDof();
      mfem::Vector N(num_nodes), dalpha_dX(3);
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);

      const double G_c = surface_energy->get_G_c();
      const double c_grad = surface_energy->get_c_grad();

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
         // alpha,J = N_c,J alpha_c
         dN_dX.MultTranspose(phase, dalpha_dX);

         const double psi_e = get_elastic_energy(elem_map);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // g'(alpha) psi_e + G_c w'(alpha)
         const double dE_dalpha = surface_energy->get_dg(alpha) * psi_e
                                  + G_c * surface_energy->get_dw(alpha);

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
   // stored at tangent(cc, dd).
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

         const double psi_e = get_elastic_energy(elem_map);

         // dV = w_q * det( dX/dxi )
         const double dV = quad_pt.weight * elem_map.Weight();

         // g''(alpha) psi_e + G_c w''(alpha)
         const double d2E_dalpha2 = surface_energy->get_d2g(alpha) * psi_e
                                    + G_c * surface_energy->get_d2w(alpha);

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
   // psi_e(F) at the point set in elem_map, with F_kJ = delta_kJ + u_k,J from
   // the displacement.
   double get_elastic_energy(mfem::ElementTransformation &elem_map)
   {
      disp.GetVectorGradient(elem_map, grad_u);
      Tensor2_3D F = Tensor2_3D::identity();
      for (int kk = 0; kk < 3; kk++)
         for (int JJ = 0; JJ < 3; JJ++)
            F(kk, JJ) += grad_u(kk, JJ);
      return material->get_strain_energy(F);
   }

   // The rule of order quad_order, or that of LocalAssemblyTools.
   const mfem::IntegrationRule &get_quad_rule(const mfem::FiniteElement &elem,
                                              mfem::ElementTransformation &elem_map) const
   {
      return (quad_order > 0) ? mfem::IntRules.Get(elem.GetGeomType(), quad_order)
                              : LocalAssemblyTools::get_quad_rule(elem, elem_map);
   }

   const std::unique_ptr<const ISurfaceEnergy> surface_energy;
   const std::unique_ptr<const IMaterialModel> material;
   const mfem::GridFunction &disp;   // displacement, not owned
   mfem::DenseMatrix grad_u;         // u_k,J at the point
   int quad_order = 0;   // quadrature order, 0 for that of LocalAssemblyTools
};

#endif

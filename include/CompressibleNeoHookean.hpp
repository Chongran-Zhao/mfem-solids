// ============================================================================
// CompressibleNeoHookean.hpp
//
// Compressible Neo-Hookean model with shear modulus mu and bulk modulus kappa,
// split into an isochoric and a volumetric part.
//
// Author: Chongran Zhao
// Date: Sep. 24, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef COMPRESSIBLE_NEO_HOOKEAN_HPP
#define COMPRESSIBLE_NEO_HOOKEAN_HPP

#include "HyperelasticMaterialModel.hpp"
#include <cmath>

class CompressibleNeoHookean : public HyperelasticMaterialModel
{
public:
   CompressibleNeoHookean(double input_kappa, double input_mu)
      : kappa(input_kappa), mu(input_mu) {}

   // Psi = mu/2 (I_1_bar - 3) + kappa/2 (J - 1)^2
   double get_strain_energy(const Tensor2_3D &F) const override
   {
      const double det_F = F.det();
      const Tensor2_3D C = F.transpose() * F;
      const double J_m23 = std::pow(det_F, -2.0 / 3.0);

      return 0.5 * mu * (J_m23 * C.trace() - 3.0)
             + 0.5 * kappa * (det_F - 1.0) * (det_F - 1.0);
   }

   // S = mu J^-2/3 I + (kappa J(J-1) - 1/3 mu J^-2/3 I_1） C^-1
   Tensor2_3D get_2nd_PK_stress(const Tensor2_3D &F) const override
   {
      const double det_F = F.det();
      const Tensor2_3D C = F.transpose() * F;
      const Tensor2_3D C_inv = C.inverse();
      const double J_m23 = std::pow(det_F, -2.0 / 3.0);
      const double tr_C = C.trace();

      return mu * J_m23 * Tensor2_3D::identity()
             + (kappa * det_F * (det_F - 1.0) - 1.0/3.0 * mu * J_m23 * tr_C )* C_inv;
   }

   // CC = (-2/3) mu J^-2/3 ( I otimes C^-1 + C^-1 otimes I)
   //    + (kappa J (2J-1) + (2/9) mu J^-2/3 I1) C^-1 otimes C^-1
   //    - 2 (kappa J (J-1) - (1/3) mu J^-2/3 I1) C^-1 odot C^-1
   Tensor4_3D get_2nd_elasticity_tensor(const Tensor2_3D &F) const override
   {
      const double det_F = F.det();
      const Tensor2_3D C = F.transpose() * F;
      const Tensor2_3D C_inv = C.inverse();
      const Tensor2_3D Id = Tensor2_3D::identity();
      const double J_m23 = std::pow(det_F, -2.0 / 3.0);
      const double tr_C = C.trace();

      return -2.0 / 3.0 * mu * J_m23 * (otimes(Id, C_inv) + otimes(C_inv, Id))
            + (kappa * det_F * (2.0 * det_F - 1.0) + 2.0 * mu / 9.0 * J_m23 * tr_C) * otimes(C_inv, C_inv)
            - 2.0 * (kappa * det_F * (det_F - 1.0) - 1.0/3.0 * mu * J_m23 * tr_C) * odot(C_inv, C_inv);
   }

private:
   double kappa, mu;
};

#endif

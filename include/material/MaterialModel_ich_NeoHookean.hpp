// ============================================================================
// MaterialModel_ich_NeoHookean.hpp
//
// Isochoric Neo-Hookean model with shear modulus mu.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_ICH_NEO_HOOKEAN_HPP
#define MATERIAL_MODEL_ICH_NEO_HOOKEAN_HPP

#include <cmath>
#include "IMaterialModel_ich.hpp"

class MaterialModel_ich_NeoHookean : public IMaterialModel_ich
{
public:
   MaterialModel_ich_NeoHookean(double input_mu) : mu(input_mu) {}

   // Psi_ich = mu/2 (J^-2/3 I_1 - 3)
   double get_energy(const Tensor2_3D &F) const override
   {
      const Tensor2_3D C = F.transpose() * F;
      return 0.5 * mu * (std::pow(F.det(), -2.0 / 3.0) * C.trace() - 3.0);
   }

   // S_ich = mu J^-2/3 (I - 1/3 I_1 C^-1)
   Tensor2_3D get_2nd_PK_stress(const Tensor2_3D &F) const override
   {
      const Tensor2_3D C = F.transpose() * F;
      const double J_m23 = std::pow(F.det(), -2.0 / 3.0);

      return mu * J_m23 * Tensor2_3D::identity()
             - 1.0 / 3.0 * mu * J_m23 * C.trace() * C.inverse();
   }

   // CC_ich = -2/3 mu J^-2/3 (I otimes C^-1 + C^-1 otimes I)
   //        + 2/9 mu J^-2/3 I_1 C^-1 otimes C^-1
   //        + 2/3 mu J^-2/3 I_1 C^-1 odot C^-1
   Tensor4_3D get_2nd_elasticity_tensor(const Tensor2_3D &F) const override
   {
      const Tensor2_3D C = F.transpose() * F;
      const Tensor2_3D C_inv = C.inverse();
      const Tensor2_3D Id = Tensor2_3D::identity();
      const double J_m23 = std::pow(F.det(), -2.0 / 3.0);
      const double tr_C = C.trace();

      return -2.0 / 3.0 * mu * J_m23 * (otimes(Id, C_inv) + otimes(C_inv, Id))
             + 2.0 / 9.0 * mu * J_m23 * tr_C * otimes(C_inv, C_inv)
             + 2.0 / 3.0 * mu * J_m23 * tr_C * odot(C_inv, C_inv);
   }

private:
   const double mu;
};

#endif

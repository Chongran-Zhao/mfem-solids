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

#include <array>
#include <cmath>

#include "IMaterialModel_ich.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

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
      const double c_mu = mu * std::pow(F.det(), -2.0 / 3.0);
      const double tr_C = C.trace();

      // Each entry at once, without temporary fourth-order tensors.
      const double coef_I_Ci = -2.0 / 3.0 * c_mu;
      const double coef_Ci_Ci = 2.0 / 9.0 * c_mu * tr_C;
      const double coef_odot = 2.0 / 3.0 * c_mu * tr_C;
      Tensor4_3D out;
      for (int II = 0; II < 3; II++)
         for (int JJ = 0; JJ < 3; JJ++)
            for (int KK = 0; KK < 3; KK++)
               for (int LL = 0; LL < 3; LL++)
                  out(II, JJ, KK, LL) =
                     coef_I_Ci * ((II == JJ) * C_inv(KK, LL) + C_inv(II, JJ) * (KK == LL))
                     + coef_Ci_Ci * C_inv(II, JJ) * C_inv(KK, LL)
                     + 0.5 * coef_odot * (C_inv(II, KK) * C_inv(JJ, LL)
                                          + C_inv(II, LL) * C_inv(JJ, KK));
      return out;
   }

   // With c = mu J^-2/3 and dc = -1/3 c C^-1 : dC,
   // dS_ich = dc (I - 1/3 I_1 C^-1) - 1/3 c tr(dC) C^-1 + 1/3 c I_1 C^-1 dC C^-1,
   // which is 1/2 CC_ich : dC for a symmetric dC; C^-1 and c are shared by
   // the three dC^J.
   std::array<Tensor2_3D, 3> get_d_2nd_PK_stress(
      const Tensor2_3D &F, const std::array<Tensor2_3D, 3> &dC) const override
   {
      const Tensor2_3D C = F.transpose() * F;
      const Tensor2_3D C_inv = C.inverse();
      const double c_mu = mu * std::pow(F.det(), -2.0 / 3.0);
      const double tr_C = C.trace();

      std::array<Tensor2_3D, 3> out;
      for (int JJ = 0; JJ < 3; JJ++)
      {
         const double dc_mu = -1.0 / 3.0 * c_mu * contract(C_inv, dC[JJ]);
         out[JJ] = dc_mu * Tensor2_3D::identity()
                   - 1.0 / 3.0 * (dc_mu * tr_C + c_mu * dC[JJ].trace()) * C_inv
                   + 1.0 / 3.0 * c_mu * tr_C * (C_inv * dC[JJ] * C_inv);
      }
      return out;
   }

   // With c = mu J^-2/3, dc = -1/3 c C^-1 : dC, dI_1 = tr dC and
   // dC^-1 = -C^-1 dC C^-1, the derivative of each term of CC_ich:
   // dCC_ich = -2/3 dc (I otimes C^-1 + C^-1 otimes I)
   //         - 2/3 c (I otimes dC^-1 + dC^-1 otimes I)
   //         + 2/9 (dc I_1 + c dI_1) C^-1 otimes C^-1
   //         + 2/9 c I_1 (dC^-1 otimes C^-1 + C^-1 otimes dC^-1)
   //         + 2/3 (dc I_1 + c dI_1) C^-1 odot C^-1
   //         + 2/3 c I_1 (dC^-1 odot C^-1 + C^-1 odot dC^-1),
   // of which only the entries (M, J, N, L) of dC^J are summed.
   std::array<Tensor2_3D, 3> get_d_2nd_elasticity_tensor_sum(
      const Tensor2_3D &F, const std::array<Tensor2_3D, 3> &dC) const override
   {
      const Tensor2_3D C = F.transpose() * F;
      const Tensor2_3D C_inv = C.inverse();
      const double c_mu = mu * std::pow(F.det(), -2.0 / 3.0);
      const double tr_C = C.trace();

      std::array<Tensor2_3D, 3> out;
      for (int JJ = 0; JJ < 3; JJ++)
      {
         const Tensor2_3D dC_inv = -1.0 * (C_inv * dC[JJ] * C_inv);
         const double dc_mu = -1.0 / 3.0 * c_mu * contract(C_inv, dC[JJ]);
         const double d_tr_C = dC[JJ].trace();

         const double coef_I_Ci = -2.0 / 3.0 * dc_mu;
         const double coef_I_dCi = -2.0 / 3.0 * c_mu;
         const double coef_Ci_Ci = 2.0 / 9.0 * (dc_mu * tr_C + c_mu * d_tr_C);
         const double coef_dCi_Ci = 2.0 / 9.0 * c_mu * tr_C;
         const double coef_odot = 2.0 / 3.0 * (dc_mu * tr_C + c_mu * d_tr_C);
         const double coef_d_odot = 2.0 / 3.0 * c_mu * tr_C;
         for (int MM = 0; MM < 3; MM++)
            for (int NN = 0; NN < 3; NN++)
               for (int LL = 0; LL < 3; LL++)
                  out[MM](NN, LL) +=
                     coef_I_Ci * ((MM == JJ) * C_inv(NN, LL) + C_inv(MM, JJ) * (NN == LL))
                     + coef_I_dCi * ((MM == JJ) * dC_inv(NN, LL) + dC_inv(MM, JJ) * (NN == LL))
                     + coef_Ci_Ci * C_inv(MM, JJ) * C_inv(NN, LL)
                     + coef_dCi_Ci * (dC_inv(MM, JJ) * C_inv(NN, LL)
                                      + C_inv(MM, JJ) * dC_inv(NN, LL))
                     + 0.5 * coef_odot * (C_inv(MM, NN) * C_inv(JJ, LL)
                                          + C_inv(MM, LL) * C_inv(JJ, NN))
                     + 0.5 * coef_d_odot * (dC_inv(MM, NN) * C_inv(JJ, LL)
                                            + dC_inv(MM, LL) * C_inv(JJ, NN)
                                            + C_inv(MM, NN) * dC_inv(JJ, LL)
                                            + C_inv(MM, LL) * dC_inv(JJ, NN));
      }
      return out;
   }

private:
   const double mu;
};

#endif

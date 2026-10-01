// ============================================================================
// MaterialModel.hpp
//
// Hyperelastic material split into a volumetric and an isochoric part,
// Psi = Psi_vol(J) + Psi_ich(F), each given by its own model.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_HPP
#define MATERIAL_MODEL_HPP

#include <memory>
#include <utility>

#include "IMaterialModel_ich.hpp"
#include "IMaterialModel_vol.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class MaterialModel
{
public:
   MaterialModel(std::unique_ptr<IMaterialModel_vol> input_vol_model,
                 std::unique_ptr<IMaterialModel_ich> input_ich_model)
      : vol_model(std::move(input_vol_model)), ich_model(std::move(input_ich_model)) {}

   // Strain energy Psi = Psi_vol + Psi_ich per reference volume.
   double get_strain_energy(const Tensor2_3D &F) const
   {
      return vol_model->get_energy(F.det()) + ich_model->get_energy(F);
   }

   // Second Piola-Kirchhoff stress S = -J p C^-1 + S_ich.
   Tensor2_3D get_2nd_PK_stress(const Tensor2_3D &F) const
   {
      const double J = F.det();
      const Tensor2_3D C_inv = (F.transpose() * F).inverse();

      return -J * vol_model->get_p(J) * C_inv + ich_model->get_2nd_PK_stress(F);
   }

   // Second elasticity tensor CC = 2 dS/dC,
   // CC = -J (p + J dp/dJ) C^-1 otimes C^-1
   //    + 2 J p C^-1 odot C^-1 + CC_ich.
   Tensor4_3D get_2nd_elasticity_tensor(const Tensor2_3D &F) const
   {
      const double J = F.det();
      const Tensor2_3D C_inv = (F.transpose() * F).inverse();
      const double p = vol_model->get_p(J);
      const double dp_dJ = vol_model->get_dp_dJ(J);

      return -J * (p + J * dp_dJ) * otimes(C_inv, C_inv)
             + 2.0 * J * p * odot(C_inv, C_inv)
             + ich_model->get_2nd_elasticity_tensor(F);
   }

   // First Piola-Kirchhoff stress P = F S.
   Tensor2_3D get_1st_PK_stress(const Tensor2_3D &F) const
   {
      return F * get_2nd_PK_stress(F);
   }

   // First elasticity tensor AA = dP/dF.
   Tensor4_3D get_1st_elasticity_tensor(const Tensor2_3D &F) const
   {
      return from_CC_to_AA(F, get_2nd_PK_stress(F), get_2nd_elasticity_tensor(F));
   }

   // Isochoric first Piola-Kirchhoff stress P_ich = F S_ich, for the mixed
   // formulation, where the pressure is a separate field.
   Tensor2_3D get_1st_PK_stress_ich(const Tensor2_3D &F) const
   {
      return F * ich_model->get_2nd_PK_stress(F);
   }

   // Isochoric first elasticity tensor AA_ich = dP_ich/dF.
   Tensor4_3D get_1st_elasticity_tensor_ich(const Tensor2_3D &F) const
   {
      return from_CC_to_AA(F, ich_model->get_2nd_PK_stress(F),
                           ich_model->get_2nd_elasticity_tensor(F));
   }

   // Pressure p(J) of the volumetric model.
   double get_p(double J) const { return vol_model->get_p(J); }

   // Volume ratio J(p) and dJ/dp of the volumetric model.
   double get_J(double p) const { return vol_model->get_J(p); }
   double get_dJ_dp(double p) const { return vol_model->get_dJ_dp(p); }

private:
   // AA from S and CC: AA_iJkL = F_iM CC_MJNL F_kN + delta_ik S_JL.
   static Tensor4_3D from_CC_to_AA(const Tensor2_3D &F, const Tensor2_3D &PK2,
                                   const Tensor4_3D &CC)
   {
      // F_iM CC_MJNL
      Tensor4_3D F_CC;
      for (int ii = 0; ii < 3; ii++)
         for (int JJ = 0; JJ < 3; JJ++)
            for (int NN = 0; NN < 3; NN++)
               for (int LL = 0; LL < 3; LL++)
                  for (int MM = 0; MM < 3; MM++)
                     F_CC(ii, JJ, NN, LL) += F(ii, MM) * CC(MM, JJ, NN, LL);

      // F_iM CC_MJNL F_kN
      Tensor4_3D AA;
      for (int ii = 0; ii < 3; ii++)
         for (int JJ = 0; JJ < 3; JJ++)
            for (int kk = 0; kk < 3; kk++)
               for (int LL = 0; LL < 3; LL++)
                  for (int NN = 0; NN < 3; NN++)
                     AA(ii, JJ, kk, LL) += F_CC(ii, JJ, NN, LL) * F(kk, NN);

      // delta_ik S_JL
      for (int ii = 0; ii < 3; ii++)
         for (int JJ = 0; JJ < 3; JJ++)
            for (int LL = 0; LL < 3; LL++)
               AA(ii, JJ, ii, LL) += PK2(JJ, LL);
      return AA;
   }

   std::unique_ptr<IMaterialModel_vol> vol_model;  // Psi_vol(J)
   std::unique_ptr<IMaterialModel_ich> ich_model;  // Psi_ich(F)
};

#endif

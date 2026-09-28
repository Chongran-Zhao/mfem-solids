// ============================================================================
// HyperelasticMaterialModel.hpp
//
// Interface of a hyperelastic material: given the deformation gradient F,
// return the strain energy, the stresses and the material stiffness.
//
// Author: Chongran Zhao
// Date: Sep. 24, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef HYPERELASTIC_MATERIAL_MODEL_HPP
#define HYPERELASTIC_MATERIAL_MODEL_HPP

#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class HyperelasticMaterialModel
{
public:
   virtual ~HyperelasticMaterialModel() = default;

   // Strain energy Psi per reference volume.
   virtual double get_strain_energy(const Tensor2_3D &F) const = 0;

   // Second Piola-Kirchhoff stress S.
   virtual Tensor2_3D get_2nd_PK_stress(const Tensor2_3D &F) const = 0;

   // Second elasticity tensor CC = 2dS/dC.
   virtual Tensor4_3D get_2nd_elasticity_tensor(const Tensor2_3D &F) const = 0;

   // First Piola-Kirchhoff stress P = F S.
   Tensor2_3D get_1st_PK_stress(const Tensor2_3D &F) const
   {
      return F * get_2nd_PK_stress(F);
   }

   // First elasticity tensor AA = dP/dF,
   // AA_iJkL = F_iM CC_MJNL F_kN + delta_ik S_JL.
   Tensor4_3D get_1st_elasticity_tensor(const Tensor2_3D &F) const
   {
      const Tensor2_3D PK2 = get_2nd_PK_stress(F);
      const Tensor4_3D CC = get_2nd_elasticity_tensor(F);

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
};

#endif

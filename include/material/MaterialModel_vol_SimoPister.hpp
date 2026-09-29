// ============================================================================
// MaterialModel_vol_SimoPister.hpp
//
// Logarithmic volumetric energy with bulk modulus kappa.
//
// Reference:
//   J.C. Simo, K.S. Pister, Remarks on rate constitutive equations for finite
//   deformation problems: computational implications, Computer Methods in
//   Applied Mechanics and Engineering 46 (1984) 201-215.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_VOL_SIMO_PISTER_HPP
#define MATERIAL_MODEL_VOL_SIMO_PISTER_HPP

#include <cmath>
#include <mfem.hpp>
#include "IMaterialModel_vol.hpp"

class MaterialModel_vol_SimoPister : public IMaterialModel_vol
{
public:
   MaterialModel_vol_SimoPister(double input_kappa) : kappa(input_kappa) {}

   // Psi_vol = kappa/2 (ln J)^2
   double get_energy(double J) const override { return 0.5 * kappa * std::log(J) * std::log(J); }

   // p = -dPsi_vol/dJ = -kappa ln J / J
   double get_p(double J) const override { return -kappa * std::log(J) / J; }

   // dp/dJ = -d^2Psi_vol/dJ^2 = -kappa (1 - ln J) / J^2
   double get_dp_dJ(double J) const override { return -kappa * (1.0 - std::log(J)) / (J * J); }

   // J(p) has no closed form, so this model is not used in the mixed
   // formulation.
   double get_J(double p) const override
   {
      MFEM_ABORT("MaterialModel_vol_SimoPister has no closed-form J(p).");
      return 0.0;
   }

   double get_dJ_dp(double p) const override
   {
      MFEM_ABORT("MaterialModel_vol_SimoPister has no closed-form J(p).");
      return 0.0;
   }

private:
   const double kappa;
};

#endif

// ============================================================================
// MaterialModel_vol_Quadratic.hpp
//
// Quadratic volumetric energy with bulk modulus kappa.
//
// Reference:
//   C.O. Horgan, J.G. Murphy, On the volumetric part of strain-energy
//   functions used in the constitutive modeling of slightly compressible
//   solid rubbers, International Journal of Solids and Structures 46 (2009)
//   3078-3085.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_VOL_QUADRATIC_HPP
#define MATERIAL_MODEL_VOL_QUADRATIC_HPP

#include "IMaterialModel_vol.hpp"

class MaterialModel_vol_Quadratic : public IMaterialModel_vol
{
public:
   MaterialModel_vol_Quadratic(double input_kappa) : kappa(input_kappa) {}

   // Psi_vol = kappa/2 (J - 1)^2
   double get_energy(double J) const override { return 0.5 * kappa * (J - 1.0) * (J - 1.0); }

   // p = -dPsi_vol/dJ = -kappa (J - 1)
   double get_p(double J) const override { return -kappa * (J - 1.0); }

   // dp/dJ = -d^2Psi_vol/dJ^2 = -kappa
   double get_dp_dJ(double J) const override { return -kappa; }

   // J = 1 - p/kappa
   double get_J(double p) const override { return 1.0 - p / kappa; }

   // dJ/dp = -1/kappa
   double get_dJ_dp(double p) const override { return -1.0 / kappa; }

private:
   const double kappa;
};

#endif

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
#include "IMaterialModel_vol.hpp"

class MaterialModel_vol_SimoPister : public IMaterialModel_vol
{
public:
   MaterialModel_vol_SimoPister(double input_kappa) : kappa(input_kappa) {}

   // Psi_vol = kappa/2 (ln J)^2
   double get_energy(double J) const override { return 0.5 * kappa * std::log(J) * std::log(J); }

   // sigma_vol = dPsi_vol/dJ = kappa ln J / J
   double get_vol_stress(double J) const override { return kappa * std::log(J) / J; }

   // dsigma_vol/dJ = d^2Psi_vol/dJ^2 = kappa (1 - ln J) / J^2
   double get_dvol_stress_dJ(double J) const override { return kappa * (1.0 - std::log(J)) / (J * J); }

private:
   const double kappa;
};

#endif

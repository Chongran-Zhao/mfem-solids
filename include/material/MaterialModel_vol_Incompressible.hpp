// ============================================================================
// MaterialModel_vol_Incompressible.hpp
//
// Fully incompressible volumetric model, J = 1 for any pressure. It has no
// strain energy as a function of J, so it is used only in the mixed
// formulation.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_VOL_INCOMPRESSIBLE_HPP
#define MATERIAL_MODEL_VOL_INCOMPRESSIBLE_HPP

#include <mfem.hpp>

#include "IMaterialModel_vol.hpp"

class MaterialModel_vol_Incompressible : public IMaterialModel_vol
{
public:
   double get_energy(double J) const override
   {
      MFEM_ABORT("MaterialModel_vol_Incompressible has no Psi_vol(J).");
      return 0.0;
   }

   double get_p(double J) const override
   {
      MFEM_ABORT("MaterialModel_vol_Incompressible has no p(J).");
      return 0.0;
   }

   double get_dp_dJ(double J) const override
   {
      MFEM_ABORT("MaterialModel_vol_Incompressible has no p(J).");
      return 0.0;
   }

   // J = 1
   double get_J(double p) const override { return 1.0; }

   // dJ/dp = 0
   double get_dJ_dp(double p) const override { return 0.0; }
};

#endif

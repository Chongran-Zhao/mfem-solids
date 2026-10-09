// ============================================================================
// MaterialModel_vol_Incompressible_Energy.hpp
//
// MaterialModel_vol_Incompressible with the volumetric strain energy
// Psi_vol(J = 1) = 0, so that csv_writer can report the strain energy of
// the incompressible body, the isochoric part alone.
//
// Author: Chongran Zhao
// Date: Oct. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_VOL_INCOMPRESSIBLE_ENERGY_HPP
#define MATERIAL_MODEL_VOL_INCOMPRESSIBLE_ENERGY_HPP

#include "MaterialModel_vol_Incompressible.hpp"

class MaterialModel_vol_Incompressible_Energy : public MaterialModel_vol_Incompressible
{
public:
   using MaterialModel_vol_Incompressible::MaterialModel_vol_Incompressible;

   // Psi_vol = 0, J being 1.
   double get_energy(double J) const override { return 0.0; }
};

#endif

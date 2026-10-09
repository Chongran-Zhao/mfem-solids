// ============================================================================
// MaterialModelData.hpp
//
// The material of sphere_rotation: incompressible Neo-Hookean, with shear
// modulus mu = 0.74 kPa and density rho_0 = 1000 kg/m^3. It replaces
// include/material/MaterialModelData.hpp for the programs of this folder.
//
// Author: Chongran Zhao
// Date: Oct. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_DATA_HPP
#define MATERIAL_MODEL_DATA_HPP

#include <memory>

#include "IMaterialModel_ich.hpp"
#include "IMaterialModel_vol.hpp"
#include "MaterialModel.hpp"
#include "MaterialModel_ich_NeoHookean.hpp"
#include "MaterialModel_vol_Incompressible_Energy.hpp"

// Shear modulus and reference density.
inline constexpr double mu = 740.0;
inline constexpr double density = 1000.0;

// Volumetric model: incompressible, J = 1.
inline std::unique_ptr<IMaterialModel_vol> set_vol_model()
{
   return std::make_unique<MaterialModel_vol_Incompressible_Energy>(density);
}

// Isochoric model: Neo-Hookean.
inline std::unique_ptr<IMaterialModel_ich> set_ich_model()
{
   return std::make_unique<MaterialModel_ich_NeoHookean>(mu);
}

// The material: the volumetric and the isochoric model together. It holds no
// state, so each owner, e.g. each local assembly, creates its own.
inline std::unique_ptr<MaterialModel> set_material_model()
{
   return std::make_unique<MaterialModel>(set_vol_model(), set_ich_model());
}

#endif

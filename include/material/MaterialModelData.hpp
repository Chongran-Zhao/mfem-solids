// ============================================================================
// MaterialModelData.hpp
//
// The material of this project, used by the drivers. Changing the material
// means changing this file.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_DATA_HPP
#define MATERIAL_MODEL_DATA_HPP

#include <memory>

#include "IMaterialModel_ich.hpp"
#include "IMaterialModel_vol.hpp"
#include "MaterialModel.hpp"
#include "MaterialModel_ich_NeoHookean.hpp"
#include "MaterialModel_vol_Quadratic.hpp"

// Young's modulus and Poisson's ratio of the reference case.
inline constexpr double young = 540.0e3;
inline constexpr double poisson = 0.324;

// Volumetric model: quadratic, with kappa from young and poisson.
inline std::unique_ptr<IMaterialModel_vol> set_vol_model()
{
   const double kappa = young / (3.0 * (1.0 - 2.0 * poisson));
   return std::make_unique<MaterialModel_vol_Quadratic>(kappa);
}

// Isochoric model: Neo-Hookean, with mu from young and poisson.
inline std::unique_ptr<IMaterialModel_ich> set_ich_model()
{
   const double mu = young / (2.0 * (1.0 + poisson));
   return std::make_unique<MaterialModel_ich_NeoHookean>(mu);
}

// The material: the volumetric and the isochoric model together. It holds no
// state, so each owner, e.g. each local assembly, creates its own.
inline std::unique_ptr<MaterialModel> set_material_model()
{
   return std::make_unique<MaterialModel>(set_vol_model(), set_ich_model());
}

#endif

// ============================================================================
// MaterialModelData.hpp
//
// The material of this project, shared by the driver, vtu_writer and
// csv_writer. Changing the material means changing this file.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_DATA_HPP
#define MATERIAL_MODEL_DATA_HPP

#include <memory>
#include "MaterialModel.hpp"
#include "MaterialModel_ich_NeoHookean.hpp"
#include "MaterialModel_vol_Quadratic.hpp"

// Young's modulus and Poisson's ratio of the reference case.
inline constexpr double young = 540.0e3;
inline constexpr double poisson = 0.324;

// Volumetric model: quadratic, with kappa from young and poisson.
inline std::unique_ptr<IMaterialModel_vol> create_vol_model()
{
   const double kappa = young / (3.0 * (1.0 - 2.0 * poisson));
   return std::make_unique<MaterialModel_vol_Quadratic>(kappa);
}

// Isochoric model: Neo-Hookean, with mu from young and poisson.
inline std::unique_ptr<IMaterialModel_ich> create_ich_model()
{
   const double mu = young / (2.0 * (1.0 + poisson));
   return std::make_unique<MaterialModel_ich_NeoHookean>(mu);
}

// The material: the volumetric and the isochoric model together.
inline MaterialModel get_material_model()
{
   return MaterialModel(create_vol_model(), create_ich_model());
}

#endif

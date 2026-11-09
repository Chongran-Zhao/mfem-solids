// ============================================================================
// MaterialModelData.hpp
//
// The material of the brittle matrix of the helical crack of Bourdin (2007,
// Section 3), as in the mef90 runs of ~/runs/HelicalCrack2007L016 (cs0001):
// E = 1, Poisson's ratio .2, the toughness G_c = 1 and the width eps = .08
// (mef90's internal length .16, not given in the paper), with the AT2
// surface energy. The unbreakable shaft and the stiffer caps (E = 1000) of
// those runs are not yet here: every element has this material. Changing
// the material means changing this file.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_DATA_HPP
#define MATERIAL_MODEL_DATA_HPP

#include <memory>

#include "IMaterialModel.hpp"
#include "ISurfaceEnergy.hpp"
#include "MaterialModel_LinearElastic.hpp"
#include "SurfaceEnergy_AT2.hpp"

// Young's modulus, Poisson's ratio and reference density; the density
// matters only to the dynamics.
inline constexpr double young = 1.0;
inline constexpr double poisson = 0.2;
inline constexpr double density = 1.0;

// Toughness G_c and width eps of the surface energy.
inline constexpr double toughness = 1.0;
inline constexpr double width = 8.0e-2;

// The elastic material. It holds no state, so each owner, the local
// assembly of the displacement and that of the phase field, creates its own.
inline std::unique_ptr<IMaterialModel> set_material_model()
{
   return std::make_unique<MaterialModel_LinearElastic>(density, young, poisson);
}

// The surface energy, one per owner as the material.
inline std::unique_ptr<ISurfaceEnergy> set_surface_energy()
{
   return std::make_unique<SurfaceEnergy_AT2>(toughness, width);
}

#endif

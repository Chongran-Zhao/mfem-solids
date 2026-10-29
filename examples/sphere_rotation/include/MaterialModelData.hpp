// ============================================================================
// MaterialModelData.hpp
//
// The materials of sphere_rotation, one per region of the mesh, by element
// attribute: 1 the gelatin, incompressible Neo-Hookean, with shear modulus
// mu = 2.8 kPa and density rho_0 = 1000 kg/m^3; 2 the skull, of acrylic
// (PMMA), linearly elastic at small strains, with E = 2.75 GPa, nu = 0.35
// and rho_0 = 1170 kg/m^3, taken as compressible Neo-Hookean, which keeps E
// and nu at small strains and stays stable at large ones,
//    Psi = kappa/2 (J - 1)^2 + Psi_ich^NH(F),
//    mu = E / (2 (1 + nu)),   kappa = E / (3 (1 - 2 nu)).
//
// Author: Chongran Zhao
// Date: Oct. 11, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_DATA_HPP
#define MATERIAL_MODEL_DATA_HPP

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <mfem.hpp>

#include "IMaterialModel_ich.hpp"
#include "IMaterialModel_vol.hpp"
#include "MaterialModel.hpp"
#include "MaterialModel_ich_NeoHookean.hpp"
#include "MaterialModel_vol_Incompressible.hpp"
#include "MaterialModel_vol_Quadratic.hpp"

// The parameters of a region.
struct region_material
{
   std::string name;   // name of the region in the mesh
   double mu;          // shear modulus
   double kappa;       // bulk modulus, infinite when incompressible
   double density;     // reference density rho_0
};

// Young's modulus and Poisson's ratio of the skull, acrylic.
inline constexpr double skull_young = 2.75e9;
inline constexpr double skull_poisson = 0.35;

// The regions, element attribute 1 and 2.
inline const std::array<region_material, 2> region_materials = {{
   {"gelatin", 2800.0, std::numeric_limits<double>::infinity(), 1000.0},
   {"skull", skull_young / (2.0 * (1.0 + skull_poisson)),
    skull_young / (3.0 * (1.0 - 2.0 * skull_poisson)), 1170.0}}};

// The parameters of the region of the given element attribute.
inline const region_material &get_region_material(int attribute)
{
   MFEM_VERIFY(attribute >= 1 && attribute <= static_cast<int>(region_materials.size()),
               "No material for element attribute " << attribute << ".");
   return region_materials[attribute - 1];
}

// Volumetric model of a region: incompressible, J = 1, or quadratic.
inline std::unique_ptr<IMaterialModel_vol> set_vol_model(int attribute)
{
   const region_material &region = get_region_material(attribute);
   if (std::isinf(region.kappa))
      return std::make_unique<MaterialModel_vol_Incompressible>(region.density);
   return std::make_unique<MaterialModel_vol_Quadratic>(region.density, region.kappa);
}

// Isochoric model of a region: Neo-Hookean.
inline std::unique_ptr<IMaterialModel_ich> set_ich_model(int attribute)
{
   return std::make_unique<MaterialModel_ich_NeoHookean>(get_region_material(attribute).mu);
}

// The material of a region: the volumetric and the isochoric model
// together. It holds no state, so each owner, e.g. each local assembly,
// creates its own.
inline std::unique_ptr<MaterialModel> set_material_model(int attribute)
{
   return std::make_unique<MaterialModel>(set_vol_model(attribute), set_ich_model(attribute));
}

#endif

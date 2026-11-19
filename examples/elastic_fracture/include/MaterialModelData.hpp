// ============================================================================
// MaterialModelData.hpp
//
// The materials of the helical crack of Bourdin, Francfort and Marigo
// (2008, Section 6, Fig. 30), one per element attribute, the physical volumes
// of mesh_files/HelicalCrack2008.geo: 1 the brittle matrix, 2 the
// unbreakable core, perfectly bonded. The values are those of a real
// spicule, in SI units (m, N, Pa, J), on the tube of the paper written in
// meters, mesh_files/HelicalCrack2008-h015-m.msh, 30 mm long and 2 mm across:
// E = 20.8 GPa and nu = .3 in both regions, G_c = 500 J/m^2, AT1. The core
// does not break, elastic: true of the mef90 runs.
//
// The width eps of the surface energy, the length scale of the phase field,
// is set here, not in config.yaml: eps = 1.5e-4 m equals the mesh size h =
// 1.5e-4 m of HelicalCrack2008-h015-m.msh, as in the mef90 runs on that
// mesh (mef90's internal length l = 2 eps = 3e-4 m). On a finer mesh keep
// eps >= h, e.g. eps = h; a smaller eps raises the critical stress
// sigma_c = sqrt(3 G_c E / (16 eps)) of AT1 (114 MPa here), so the load at
// which the notch grows.
// Changing the materials means changing this file.
//
// Author: Chongran Zhao
// Date: Nov. 10, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_DATA_HPP
#define MATERIAL_MODEL_DATA_HPP

#include <array>
#include <memory>
#include <string>

#include <mfem.hpp>

#include "IMaterialModel.hpp"
#include "ISurfaceEnergy.hpp"
#include "MaterialModel_LinearElastic.hpp"
#include "SurfaceEnergy_AT1.hpp"

// The parameters of a region.
struct region_material
{
   std::string name;   // name of the region in the mesh
   double young;       // Young's modulus (Pa)
   double poisson;     // Poisson's ratio
   double density;     // reference density rho_0, which matters only to the dynamics, unused here
   bool breakable;     // whether the region breaks, else elastic: true of mef90
};

// The regions, element attribute 1 and 2.
inline const std::array<region_material, 2> region_materials = {{
   {"matrix", 2.08e10, 0.3, 1.0, true},
   {"core", 2.08e10, 0.3, 1.0, false}}};

// Toughness G_c (J/m^2), width eps (m), the length scale of the phase field,
// and residual stiffness eta of the surface energy; eta = 1e-6 is the value
// of Section 8.3.3 of Bourdin, Francfort and Marigo (2008), and of the mef90
// runs. eps follows the mesh: see the top of this file.
inline constexpr double toughness = 500.0;
inline constexpr double width = 1.5e-4;
inline constexpr double residual_stiffness = 1.0e-6;

// The parameters of the region of the given element attribute.
inline const region_material &get_region_material(int attribute)
{
   MFEM_VERIFY(attribute >= 1 && attribute <= static_cast<int>(region_materials.size()),
               "No material for element attribute " << attribute << ".");
   return region_materials[attribute - 1];
}

// The elastic material of a region. It holds no state, so each owner, the
// local assembly of the displacement and that of the phase field, creates
// its own.
inline std::unique_ptr<IMaterialModel> set_material_model(int attribute)
{
   const region_material &region = get_region_material(attribute);
   return std::make_unique<MaterialModel_LinearElastic>(region.density, region.young,
                                                        region.poisson);
}

// The surface energy, one per owner as the material.
inline std::unique_ptr<ISurfaceEnergy> set_surface_energy()
{
   return std::make_unique<SurfaceEnergy_AT1>(toughness, width, residual_stiffness);
}

#endif

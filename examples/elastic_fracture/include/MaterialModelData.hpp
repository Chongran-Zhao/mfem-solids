// ============================================================================
// MaterialModelData.hpp
//
// The materials of the helical crack of Bourdin, Francfort and Marigo
// (2008, Section 6, Fig. 30), one per element attribute, the physical volumes
// of mesh_files/HelicalCrack2008.geo: 1 the brittle matrix, 2 the
// unbreakable core, perfectly bonded. Units N, mm, MPa: the tube of the
// mesh is read in mm. E = 20.8 GPa = 20800 MPa and nu = .3 in both regions,
// G_c = 500 J/m^2 = .5 N/mm, the values asked for on 2026-11-17; before, E = 1, nu = .2 and G_c = 1, as in the mef90 runs of
// ~/runs/ATTEMPTS.md. With the geometry and eps fixed, E and G_c only scale
// the displacement by sqrt(G_c / E) = 4.9e-3 mm and the energy by G_c = .5
// against those runs; nu changes the solution. The core does not break,
// elastic: true in the mef90 runs. The surface energy is AT1, as the mef90
// run HelicalCrack that gave the helix, with the width eps = .15 mm,
// mef90's internal length .3; AT2 is chosen by SurfaceEnergy_AT2 in
// set_surface_energy below. The phase field is solved with the bounds
// 0 <= alpha <= 1, which AT1 needs, for both.
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
   double young;       // Young's modulus, MPa
   double poisson;     // Poisson's ratio
   double density;     // reference density rho_0, which matters only to the dynamics
   bool breakable;     // whether the region breaks, else elastic: true of mef90
};

// The regions, element attribute 1 and 2.
inline const std::array<region_material, 2> region_materials = {{
   {"matrix", 2.08e4, 0.3, 1.0, true},
   {"core", 2.08e4, 0.3, 1.0, false}}};

// Toughness G_c, width eps and residual stiffness eta of the surface
// energy; eta = 1e-6 is the value of Section 8.3.3 of Bourdin, Francfort
// and Marigo (2008), and of the mef90 runs.
inline constexpr double toughness = 0.5;
inline constexpr double width = 1.5e-1;
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

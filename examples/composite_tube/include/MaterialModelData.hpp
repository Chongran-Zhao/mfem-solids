// ============================================================================
// MaterialModelData.hpp
//
// The materials of the helical crack of Bourdin, Francfort and Marigo
// (2008, Section 6, Fig. 30), one per element attribute, the physical volumes
// of mesh_files/HelicalCrack2008.geo: 1 the brittle matrix, 2 the
// unbreakable core, perfectly bonded. Poisson's ratio .2 in both (Section
// 6); E_f / E_m = 1, of the caption of Fig. 30, with E = 1, and the
// toughness 1, as in the mef90 runs of ~/runs/ATTEMPTS.md; the core does not
// break, elastic: true in those runs. The surface energy is AT1, as in the
// run of those that gave a helix (HelicalCrack), with G_c = 1 and the width
// eps = .15, mef90's internal length .3, not given in the paper; the paper's
// regularization is AT2. AT1 needs the bounds of
// solver_fracture.nonlinear_solver vi.
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
   double young;       // Young's modulus
   double poisson;     // Poisson's ratio
   double density;     // reference density rho_0, which matters only to the dynamics
   bool breakable;     // whether the region breaks, else elastic: true of mef90
};

// The regions, element attribute 1 and 2.
inline const std::array<region_material, 2> region_materials = {{
   {"matrix", 1.0, 0.2, 1.0, true},
   {"core", 1.0, 0.2, 1.0, false}}};

// Toughness G_c, width eps and residual stiffness eta of the surface
// energy; eta = 1e-6 is the value of Section 8.3.3 of Bourdin, Francfort
// and Marigo (2008), and of the mef90 runs.
inline constexpr double toughness = 1.0;
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

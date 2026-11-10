// ============================================================================
// MaterialModelData.hpp
//
// The materials of the helical crack of Bourdin (2007, Section 3), one per
// element attribute, the physical volumes of mesh_files/HelicalCrack2007.geo,
// as in the mef90 runs of ~/runs/HelicalCrack2007L016: 1 the brittle matrix
// (cs0001), 2 the shaft at the center (cs0002), 3 and 4 the bottom and the
// top cap (cs0003, cs0004). The paper gives E = 1 and Poisson's ratio .2 to
// the matrix and the shaft, and the toughness 1; the shaft and the caps do
// not break, elastic: true in those runs. The caps are rigid in the paper,
// E = 1000 in those runs. The surface energy is AT1, as in the mef90 runs of
// the 2008 helical crack that gave a helix (~/runs/ATTEMPTS.md), with
// G_c = 1 and the width eps = .08, mef90's internal length .16, not given in
// the paper; the paper's regularization is AT2. AT1 needs the bounds of
// solver_fracture.nonlinear_solver vi.
// Changing the materials means changing this file.
//
// Author: Chongran Zhao
// Date: Nov. 9, 2026
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

// The regions, element attribute 1 to 4.
inline const std::array<region_material, 4> region_materials = {{
   {"matrix", 1.0, 0.2, 1.0, true},
   {"shaft", 1.0, 0.2, 1.0, false},
   {"bottom cap", 1000.0, 0.2, 1.0, false},
   {"top cap", 1000.0, 0.2, 1.0, false}}};

// Toughness G_c and width eps of the surface energy.
inline constexpr double toughness = 1.0;
inline constexpr double width = 8.0e-2;

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
   return std::make_unique<SurfaceEnergy_AT1>(toughness, width);
}

#endif

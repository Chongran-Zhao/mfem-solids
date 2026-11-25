// ============================================================================
// MaterialModelData.hpp
//
// The materials of brain_simon, one per part of SIMon, by element attribute,
// in the order read_kfile gives the parts, checked by name against the mesh.
// All are incompressible Neo-Hookean, in place of the materials of Simon.k
// (Table S2 of Giudice et al., 2019):
//    cerebrum, cerebellum, brain stem   Kelvin-Maxwell, G0 = 1.66 kPa,
//                                       G_inf = 0.928 kPa, K = 558.47 MPa,
//                                       rho_0 = 1040 kg/m^3;
//    CSF                                Kelvin-Maxwell, G0 = 0.1 MPa,
//                                       G_inf = 0.02 MPa, K = 4.966 MPa,
//                                       rho_0 = 1050 kg/m^3;
//    ventricles                         elastic fluid, K = 2100 MPa,
//                                       rho_0 = 1000 kg/m^3.
// The shear modulus mu is the instantaneous G0 of MAT_061,
// G(t) = G_inf + (G0 - G_inf) exp(-beta t) at t = 0; the ventricles, of no
// shear modulus in SIMon, take that of the Abaqus model of Giudice et al.,
// E = 0.03 MPa at nu = 0.5, mu = E / 3. The bulk moduli are left out: J = 1.
//
// Author: Chongran Zhao
// Date: Nov. 25, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_DATA_HPP
#define MATERIAL_MODEL_DATA_HPP

#include <array>
#include <limits>
#include <memory>
#include <string>

#include <mfem.hpp>

#include "IMaterialModel_ich.hpp"
#include "IMaterialModel_vol.hpp"
#include "MaterialModel.hpp"
#include "MaterialModel_ich_NeoHookean.hpp"
#include "MaterialModel_vol_Incompressible.hpp"

// The parameters of a region.
struct region_material
{
   std::string name;   // name of the part in the mesh
   double mu;          // shear modulus
   double kappa;       // bulk modulus, infinite when incompressible
   double density;     // reference density rho_0
};

// Shear moduli and densities of the tissues.
inline constexpr double brain_mu = 1660.0;
inline constexpr double brain_density = 1040.0;
inline constexpr double csf_mu = 1.0e5;
inline constexpr double csf_density = 1050.0;
inline constexpr double ventricle_mu = 0.03e6 / 3.0;
inline constexpr double ventricle_density = 1000.0;
inline constexpr double incompressible = std::numeric_limits<double>::infinity();

// The parts, element attribute 1 to 10.
inline const std::array<region_material, 10> region_materials = {{
   {"CEREBRUM-R", brain_mu, incompressible, brain_density},
   {"CEREBELLUM-R", brain_mu, incompressible, brain_density},
   {"BSTEM-R", brain_mu, incompressible, brain_density},
   {"VENTRICLE-R", ventricle_mu, incompressible, ventricle_density},
   {"CSF-R", csf_mu, incompressible, csf_density},
   {"CEREBRUM-L", brain_mu, incompressible, brain_density},
   {"CEREBELLUM-L", brain_mu, incompressible, brain_density},
   {"BSTEM-L", brain_mu, incompressible, brain_density},
   {"VENTRICLE-L", ventricle_mu, incompressible, ventricle_density},
   {"CSF-L", csf_mu, incompressible, csf_density}}};

// The parameters of the region of the given element attribute.
inline const region_material &get_region_material(int attribute)
{
   MFEM_VERIFY(attribute >= 1 && attribute <= static_cast<int>(region_materials.size()),
               "No material for element attribute " << attribute << ".");
   return region_materials[attribute - 1];
}

// Checks that each attribute of the mesh is the part of its material.
inline void check_region_names(const mfem::Mesh &mesh)
{
   for (int attribute = 1; attribute <= static_cast<int>(region_materials.size()); attribute++)
   {
      const std::string &name = region_materials[attribute - 1].name;
      MFEM_VERIFY(mesh.attribute_sets.AttributeSetExists(name),
                  "The mesh has no part \"" << name << "\".");
      const mfem::Array<int> &attributes = mesh.attribute_sets.GetAttributeSet(name);
      MFEM_VERIFY(attributes.Size() == 1 && attributes[0] == attribute,
                  "The part \"" << name << "\" is not attribute " << attribute << ".");
   }
}

// Volumetric model of a region: incompressible, J = 1.
inline std::unique_ptr<IMaterialModel_vol> set_vol_model(int attribute)
{
   return std::make_unique<MaterialModel_vol_Incompressible>(get_region_material(attribute).density);
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

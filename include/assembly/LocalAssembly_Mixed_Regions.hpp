// ============================================================================
// LocalAssembly_Mixed_Regions.hpp
//
// LocalAssembly_Mixed with one material per region of the mesh: the element
// residual and tangent of an element are those of the material of its
// attribute. Its own material is that of attribute 1, so get_rho_0 gives the
// density of attribute 1 only, and the mass matrix is assembled with the
// density of each region elsewhere.
//
// Author: Chongran Zhao
// Date: Oct. 11, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_MIXED_REGIONS_HPP
#define LOCAL_ASSEMBLY_MIXED_REGIONS_HPP

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <mfem.hpp>

#include "IMassEquation.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "MaterialModel.hpp"

class LocalAssembly_Mixed_Regions : public LocalAssembly_Mixed
{
public:
   // Takes the ownership of the materials, materials[aa] being that of element
   // attribute aa + 1, and of the mass equation, shared by the regions.
   LocalAssembly_Mixed_Regions(std::vector<std::unique_ptr<const MaterialModel>> materials,
                               std::unique_ptr<IMassEquation> input_mass_equation)
      : LocalAssembly_Mixed(std::move(materials[0]), std::move(input_mass_equation))
   {
      for (std::size_t aa = 1; aa < materials.size(); aa++)
         other_materials.push_back(std::move(materials[aa]));
   }

protected:
   // The material of the attribute of the element.
   const MaterialModel &get_material(mfem::ElementTransformation &elem_map) const override
   {
      const int attribute = elem_map.Attribute;
      if (attribute == 1)
         return *material;
      MFEM_VERIFY(attribute >= 2 && attribute <= static_cast<int>(other_materials.size()) + 1,
                  "No material for element attribute " << attribute << ".");
      return *other_materials[attribute - 2];
   }

private:
   // The materials of attribute 2, 3, ...
   std::vector<std::unique_ptr<const MaterialModel>> other_materials;
};

#endif

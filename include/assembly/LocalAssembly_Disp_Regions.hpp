// ============================================================================
// LocalAssembly_Disp_Regions.hpp
//
// LocalAssembly_Disp with one material per region of the mesh: the element
// energy, residual and tangent of an element are those of the material of its
// attribute, times the energy factor of that region, if it has one. It is a
// LocalAssembly_Disp, whose own material is that of attribute 1, so that
// GlobalAssembly_Disp takes it unchanged; the other attributes have a
// LocalAssembly_Disp each. get_rho_0 gives the density of attribute 1 only, so
// the mass matrix of GlobalAssembly_Disp has the density of that region.
//
// Author: Chongran Zhao
// Date: Nov. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_DISP_REGIONS_HPP
#define LOCAL_ASSEMBLY_DISP_REGIONS_HPP

#include <memory>
#include <utility>
#include <vector>

#include <mfem.hpp>

#include "IMaterialModel.hpp"
#include "LocalAssembly_Disp.hpp"

class LocalAssembly_Disp_Regions : public LocalAssembly_Disp
{
public:
   // Takes the ownership of the materials; materials[aa] is that of element
   // attribute aa + 1.
   LocalAssembly_Disp_Regions(std::vector<std::unique_ptr<const IMaterialModel>> materials)
      : LocalAssembly_Disp(std::move(materials[0]))
   {
      for (std::size_t aa = 1; aa < materials.size(); aa++)
         other_regions.push_back(std::make_unique<LocalAssembly_Disp>(std::move(materials[aa])));
   }

   // Element energy of the material of the element attribute.
   double GetElementEnergy(const mfem::FiniteElement &elem,
                           mfem::ElementTransformation &elem_map,
                           const mfem::Vector &disp) override
   {
      if (elem_map.Attribute == 1)
         return LocalAssembly_Disp::GetElementEnergy(elem, elem_map, disp);
      return get_other_region(elem_map.Attribute).GetElementEnergy(elem, elem_map, disp);
   }

   // Element residual of the material of the element attribute.
   void AssembleElementVector(const mfem::FiniteElement &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Vector &disp,
                              mfem::Vector &residual) override
   {
      if (elem_map.Attribute == 1)
         LocalAssembly_Disp::AssembleElementVector(elem, elem_map, disp, residual);
      else
         get_other_region(elem_map.Attribute).AssembleElementVector(elem, elem_map, disp,
                                                                    residual);
   }

   // Element tangent of the material of the element attribute.
   void AssembleElementGrad(const mfem::FiniteElement &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Vector &disp,
                            mfem::DenseMatrix &tangent) override
   {
      if (elem_map.Attribute == 1)
         LocalAssembly_Disp::AssembleElementGrad(elem, elem_map, disp, tangent);
      else
         get_other_region(elem_map.Attribute).AssembleElementGrad(elem, elem_map, disp,
                                                                  tangent);
   }

   // Quadrature order of the elements of every region; 0 keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   void set_quad_order(int input_quad_order)
   {
      LocalAssembly_Disp::set_quad_order(input_quad_order);
      for (const std::unique_ptr<LocalAssembly_Disp> &region : other_regions)
         region->set_quad_order(input_quad_order);
   }

   // The factor of the energy density of the region of an element attribute,
   // e.g. g(alpha) where the material breaks; it stays the caller's. A region
   // without one has the factor 1.
   void set_energy_factor(int attribute, mfem::Coefficient &input_energy_factor)
   {
      if (attribute == 1)
         LocalAssembly_Disp::set_energy_factor(input_energy_factor);
      else
         get_other_region(attribute).set_energy_factor(input_energy_factor);
   }

private:
   // The local assembly of attribute 2, 3, ...
   LocalAssembly_Disp &get_other_region(int attribute) const
   {
      MFEM_VERIFY(attribute >= 2 && attribute <= static_cast<int>(other_regions.size()) + 1,
                  "No material for element attribute " << attribute << ".");
      return *other_regions[attribute - 2];
   }

   // The local assemblies of attribute 2, 3, ...
   std::vector<std::unique_ptr<LocalAssembly_Disp>> other_regions;
};

#endif

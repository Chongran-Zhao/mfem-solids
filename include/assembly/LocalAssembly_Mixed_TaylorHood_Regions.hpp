// ============================================================================
// LocalAssembly_Mixed_TaylorHood_Regions.hpp
//
// LocalAssembly_Mixed_TaylorHood with one material per region of the mesh:
// the element residual and tangent of an element are those of the material of
// its attribute. It is a LocalAssembly_Mixed_TaylorHood, whose own material is
// that of attribute 1, so that GlobalAssembly_Mixed takes it unchanged; the
// other attributes have a LocalAssembly_Mixed_TaylorHood each. get_rho_0
// gives the density of attribute 1 only, so the mass matrix is assembled with
// the density of each region elsewhere.
//
// Author: Chongran Zhao
// Date: Oct. 11, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_MIXED_TAYLOR_HOOD_REGIONS_HPP
#define LOCAL_ASSEMBLY_MIXED_TAYLOR_HOOD_REGIONS_HPP

#include <memory>
#include <utility>
#include <vector>

#include <mfem.hpp>

#include "LocalAssembly_Mixed_TaylorHood.hpp"
#include "MaterialModel.hpp"

class LocalAssembly_Mixed_TaylorHood_Regions : public LocalAssembly_Mixed_TaylorHood
{
public:
   // Takes the ownership of the materials; materials[aa] is that of element
   // attribute aa + 1.
   LocalAssembly_Mixed_TaylorHood_Regions(
      std::vector<std::unique_ptr<const MaterialModel>> materials)
      : LocalAssembly_Mixed_TaylorHood(std::move(materials[0]))
   {
      for (std::size_t aa = 1; aa < materials.size(); aa++)
         other_regions.push_back(
            std::make_unique<LocalAssembly_Mixed_TaylorHood>(std::move(materials[aa])));
   }

   // Element residual of the material of the element attribute.
   void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Array<const mfem::Vector *> &elem_sol,
                              const mfem::Array<mfem::Vector *> &residual) override
   {
      if (elem_map.Attribute == 1)
         LocalAssembly_Mixed_TaylorHood::AssembleElementVector(elem, elem_map, elem_sol, residual);
      else
         get_other_region(elem_map.Attribute).AssembleElementVector(elem, elem_map, elem_sol,
                                                                    residual);
   }

   // Element tangent of the material of the element attribute.
   void AssembleElementGrad(const mfem::Array<const mfem::FiniteElement *> &elem,
                            mfem::ElementTransformation &elem_map,
                            const mfem::Array<const mfem::Vector *> &elem_sol,
                            const mfem::Array2D<mfem::DenseMatrix *> &tangent) override
   {
      if (elem_map.Attribute == 1)
         LocalAssembly_Mixed_TaylorHood::AssembleElementGrad(elem, elem_map, elem_sol, tangent);
      else
         get_other_region(elem_map.Attribute).AssembleElementGrad(elem, elem_map, elem_sol,
                                                                  tangent);
   }

   // Quadrature order of the elements of every region; 0 keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   void set_quad_order(int input_quad_order)
   {
      LocalAssembly_Mixed_TaylorHood::set_quad_order(input_quad_order);
      for (const std::unique_ptr<LocalAssembly_Mixed_TaylorHood> &region : other_regions)
         region->set_quad_order(input_quad_order);
   }

private:
   // The local assembly of attribute 2, 3, ...
   LocalAssembly_Mixed_TaylorHood &get_other_region(int attribute) const
   {
      MFEM_VERIFY(attribute >= 2 && attribute <= static_cast<int>(other_regions.size()) + 1,
                  "No material for element attribute " << attribute << ".");
      return *other_regions[attribute - 2];
   }

   // The local assemblies of attribute 2, 3, ...
   std::vector<std::unique_ptr<LocalAssembly_Mixed_TaylorHood>> other_regions;
};

#endif

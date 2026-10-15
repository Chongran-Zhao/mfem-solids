// ============================================================================
// LocalAssembly_Mixed_VMS_Regions.hpp
//
// LocalAssembly_Mixed_VMS with one material per region of the mesh: the
// element residual and tangent of an element are those of the material, the
// shear modulus and the tau_factor of its attribute. It is a
// LocalAssembly_Mixed_VMS, whose own material is that of attribute 1, so that
// GlobalAssembly_Mixed takes it unchanged; the other attributes have a
// LocalAssembly_Mixed_VMS each, and set_rates gives the rates to all.
// get_rho_0 gives the density of attribute 1 only, so the mass matrix is
// assembled with the density of each region elsewhere.
//
// Author: Chongran Zhao
// Date: Oct. 15, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOCAL_ASSEMBLY_MIXED_VMS_REGIONS_HPP
#define LOCAL_ASSEMBLY_MIXED_VMS_REGIONS_HPP

#include <memory>
#include <utility>
#include <vector>

#include <mfem.hpp>

#include "LocalAssembly_Mixed_VMS.hpp"
#include "MaterialModel.hpp"

class LocalAssembly_Mixed_VMS_Regions : public LocalAssembly_Mixed_VMS
{
public:
   // Takes the ownership of the materials; materials[aa], shear_moduli[aa]
   // and tau_factors[aa] are those of element attribute aa + 1.
   LocalAssembly_Mixed_VMS_Regions(std::vector<std::unique_ptr<const MaterialModel>> materials,
                                   const std::vector<double> &shear_moduli,
                                   const std::vector<double> &tau_factors)
      : LocalAssembly_Mixed_VMS(std::move(materials[0]), shear_moduli[0], tau_factors[0])
   {
      MFEM_VERIFY(shear_moduli.size() == materials.size() && tau_factors.size() == materials.size(),
                  "Every region needs a material, a shear modulus and a tau_factor.");
      for (std::size_t aa = 1; aa < materials.size(); aa++)
         other_regions.push_back(std::make_unique<LocalAssembly_Mixed_VMS>(
            std::move(materials[aa]), shear_moduli[aa], tau_factors[aa]));
   }

   // The rates of LocalAssembly_Mixed_VMS::set_rates, for every region.
   void set_rates(const mfem::GridFunction &input_disp_offset, double input_disp_factor,
                  const mfem::GridFunction &input_dot_velo_offset,
                  const mfem::GridFunction &input_dot_pres_offset, double input_dot_factor)
   {
      LocalAssembly_Mixed_VMS::set_rates(input_disp_offset, input_disp_factor,
                                         input_dot_velo_offset, input_dot_pres_offset,
                                         input_dot_factor);
      for (const std::unique_ptr<LocalAssembly_Mixed_VMS> &region : other_regions)
         region->set_rates(input_disp_offset, input_disp_factor, input_dot_velo_offset,
                           input_dot_pres_offset, input_dot_factor);
   }

   // Element residual of the material of the element attribute.
   void AssembleElementVector(const mfem::Array<const mfem::FiniteElement *> &elem,
                              mfem::ElementTransformation &elem_map,
                              const mfem::Array<const mfem::Vector *> &elem_sol,
                              const mfem::Array<mfem::Vector *> &residual) override
   {
      if (elem_map.Attribute == 1)
         LocalAssembly_Mixed_VMS::AssembleElementVector(elem, elem_map, elem_sol, residual);
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
         LocalAssembly_Mixed_VMS::AssembleElementGrad(elem, elem_map, elem_sol, tangent);
      else
         get_other_region(elem_map.Attribute).AssembleElementGrad(elem, elem_map, elem_sol,
                                                                  tangent);
   }

   // Quadrature order of the elements of every region; 0 keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   void set_quad_order(int input_quad_order)
   {
      LocalAssembly_Mixed_VMS::set_quad_order(input_quad_order);
      for (const std::unique_ptr<LocalAssembly_Mixed_VMS> &region : other_regions)
         region->set_quad_order(input_quad_order);
   }

private:
   // The local assembly of attribute 2, 3, ...
   LocalAssembly_Mixed_VMS &get_other_region(int attribute) const
   {
      MFEM_VERIFY(attribute >= 2 && attribute <= static_cast<int>(other_regions.size()) + 1,
                  "No material for element attribute " << attribute << ".");
      return *other_regions[attribute - 2];
   }

   // The local assemblies of attribute 2, 3, ...
   std::vector<std::unique_ptr<LocalAssembly_Mixed_VMS>> other_regions;
};

#endif

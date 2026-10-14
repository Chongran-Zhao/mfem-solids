// ============================================================================
// ILocalAssembly.hpp
//
// Interface of the local (element) assemblies, LocalAssembly_Disp and
// LocalAssembly_Mixed, each also the MFEM integrator of its form: it holds
// what they share, the material they own, its reference density for the
// mass matrix, and the quadrature rule of the elements.
//
// Author: Chongran Zhao
// Date: Oct. 14, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef ILOCAL_ASSEMBLY_HPP
#define ILOCAL_ASSEMBLY_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>

#include "LocalAssemblyTools.hpp"
#include "MaterialModel.hpp"

class ILocalAssembly
{
public:
   virtual ~ILocalAssembly() = default;

   // Reference density rho_0 of the material, for the mass matrix.
   double get_rho_0() const { return material->get_rho_0(); }

   // Quadrature order of the elements; 0, the default, keeps the rule of
   // LocalAssemblyTools::get_quad_rule.
   virtual void set_quad_order(int input_quad_order) { quad_order = input_quad_order; }

protected:
   // Takes the ownership of the material.
   ILocalAssembly(std::unique_ptr<const MaterialModel> input_material)
      : material(std::move(input_material)) {}

   // The rule of order quad_order, or that of LocalAssemblyTools.
   const mfem::IntegrationRule &get_quad_rule(const mfem::FiniteElement &elem,
                                              mfem::ElementTransformation &elem_map) const
   {
      return (quad_order > 0) ? mfem::IntRules.Get(elem.GetGeomType(), quad_order)
                              : LocalAssemblyTools::get_quad_rule(elem, elem_map);
   }

   const std::unique_ptr<const MaterialModel> material;

private:
   int quad_order = 0;   // quadrature order, 0 for that of LocalAssemblyTools
};

#endif

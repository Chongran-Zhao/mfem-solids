// ============================================================================
// HyperelasticMaterialModel.hpp
//
// Interface of a hyperelastic material: given the deformation gradient F,
// return the strain energy, the stresses and the material stiffness.
//
// Author: Chongran Zhao
// Date: Sep. 24, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef HYPERELASTIC_MATERIAL_MODEL_HPP
#define HYPERELASTIC_MATERIAL_MODEL_HPP

#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class HyperelasticMaterialModel
{
public:
   virtual ~HyperelasticMaterialModel() = default;

   // Strain energy Psi per reference volume.
   virtual double get_strain_energy(const Tensor2_3D &F) const = 0;

   // Second Piola-Kirchhoff stress S.
   virtual Tensor2_3D get_2nd_PK_stress(const Tensor2_3D &F) const = 0;

   // Material elasticity tensor CC = 2dS/dC.
   virtual Tensor4_3D get_stiffness(const Tensor2_3D &F) const = 0;

   // First Piola-Kirchhoff stress P = F S.
   Tensor2_3D get_1st_PK_stress(const Tensor2_3D &F) const
   {
      return F * get_2nd_PK_stress(F);
   }
};

#endif

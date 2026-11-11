// ============================================================================
// IMaterialModel.hpp
//
// Interface of an elastic material for LocalAssembly_Disp: given the
// deformation gradient F, return the strain energy per reference volume, the
// first Piola-Kirchhoff stress P = dPsi/dF and the first elasticity tensor
// AA = dP/dF, and the reference density. MaterialModel, the hyperelastic
// materials, and MaterialModel_LinearElastic implement it.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef IMATERIAL_MODEL_HPP
#define IMATERIAL_MODEL_HPP

#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class IMaterialModel
{
public:
   virtual ~IMaterialModel() = default;

   // Strain energy Psi per reference volume.
   virtual double get_strain_energy(const Tensor2_3D &F) const = 0;

   // First Piola-Kirchhoff stress P = dPsi/dF.
   virtual Tensor2_3D get_1st_PK_stress(const Tensor2_3D &F) const = 0;

   // First elasticity tensor AA = dP/dF.
   virtual Tensor4_3D get_1st_elasticity_tensor(const Tensor2_3D &F) const = 0;

   // Reference density rho_0, mass per reference volume.
   virtual double get_rho_0() const = 0;
};

#endif

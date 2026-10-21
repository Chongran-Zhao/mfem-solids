// ============================================================================
// IMaterialModel_ich.hpp
//
// Interface of the isochoric part of a hyperelastic material: given the
// deformation gradient F, return its strain energy, stress and stiffness,
// and the derivative of the stiffness.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef IMATERIAL_MODEL_ICH_HPP
#define IMATERIAL_MODEL_ICH_HPP

#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class IMaterialModel_ich
{
public:
   virtual ~IMaterialModel_ich() = default;

   // Isochoric strain energy Psi_ich per reference volume.
   virtual double get_energy(const Tensor2_3D &F) const = 0;

   // Isochoric second Piola-Kirchhoff stress S_ich.
   virtual Tensor2_3D get_2nd_PK_stress(const Tensor2_3D &F) const = 0;

   // Isochoric second elasticity tensor CC_ich = 2 dS_ich/dC.
   virtual Tensor4_3D get_2nd_elasticity_tensor(const Tensor2_3D &F) const = 0;

   // Derivative of S_ich along a symmetric dC, dS_ich = 1/2 CC_ich : dC,
   // without CC_ich, for the divergence of the stress in the VMS
   // stabilization.
   virtual Tensor2_3D get_d_2nd_PK_stress(const Tensor2_3D &F, const Tensor2_3D &dC) const = 0;

   // Derivative of CC_ich along dC, dCC_ich = dCC_ich/dC : dC, for the
   // tangent of the VMS stabilization.
   virtual Tensor4_3D get_d_2nd_elasticity_tensor(const Tensor2_3D &F,
                                                  const Tensor2_3D &dC) const = 0;
};

#endif

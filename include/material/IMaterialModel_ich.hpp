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

#include <array>

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

   // Derivatives of S_ich along three symmetric dC^J, dS_ich^J = 1/2 CC_ich : dC^J,
   // without CC_ich, for the divergence of the stress in the VMS
   // stabilization, where the dC^J come from the three slices of the
   // displacement Hessian.
   virtual std::array<Tensor2_3D, 3> get_d_2nd_PK_stress(
      const Tensor2_3D &F, const std::array<Tensor2_3D, 3> &dC) const = 0;

   // sum_J dCC_ich[dC^J]_MJNL, dCC_ich[dC] = dCC_ich/dC : dC, the derivatives
   // of CC_ich along three symmetric dC^J summed over the second index,
   // stored as [M](N, L), for the tangent of the VMS stabilization.
   virtual std::array<Tensor2_3D, 3> get_d_2nd_elasticity_tensor_sum(
      const Tensor2_3D &F, const std::array<Tensor2_3D, 3> &dC) const = 0;
};

#endif

// ============================================================================
// MaterialModel_LinearElastic.hpp
//
// Isotropic linear elasticity in small strain, with Young's modulus E,
// Poisson's ratio nu and the reference density rho_0, written with
// F = I + Grad u so that LocalAssembly_Disp assembles it as it is:
//    eps     = (F + F^T)/2 - I,
//    Psi     = lambda/2 (tr eps)^2 + mu eps : eps,
//    P       = lambda tr(eps) I + 2 mu eps = sigma,
//    AA_kJlL = lambda d_kJ d_lL + mu (d_kl d_JL + d_kL d_Jl),
// with lambda = E nu / ((1 + nu)(1 - 2 nu)) and mu = E / (2 (1 + nu)). Since
// P is linear in F, Newton's method converges in one iteration.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_LINEAR_ELASTIC_HPP
#define MATERIAL_MODEL_LINEAR_ELASTIC_HPP

#include "IMaterialModel.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class MaterialModel_LinearElastic : public IMaterialModel
{
public:
   MaterialModel_LinearElastic(double input_rho_0, double input_E, double input_nu)
      : rho_0(input_rho_0),
        lambda(input_E * input_nu / ((1.0 + input_nu) * (1.0 - 2.0 * input_nu))),
        mu(input_E / (2.0 * (1.0 + input_nu))) {}

   // Psi = lambda/2 (tr eps)^2 + mu eps : eps
   double get_strain_energy(const Tensor2_3D &F) const override
   {
      const Tensor2_3D eps = get_strain(F);
      return 0.5 * lambda * eps.trace() * eps.trace() + mu * contract(eps, eps);
   }

   // P = sigma = lambda tr(eps) I + 2 mu eps
   Tensor2_3D get_1st_PK_stress(const Tensor2_3D &F) const override
   {
      const Tensor2_3D eps = get_strain(F);
      return lambda * eps.trace() * Tensor2_3D::identity() + 2.0 * mu * eps;
   }

   // AA = lambda I otimes I + 2 mu I odot I
   Tensor4_3D get_1st_elasticity_tensor(const Tensor2_3D &F) const override
   {
      const Tensor2_3D II = Tensor2_3D::identity();
      return lambda * otimes(II, II) + 2.0 * mu * odot(II, II);
   }

   double get_rho_0() const override { return rho_0; }

private:
   // eps = (F + F^T)/2 - I, the small strain sym(Grad u).
   static Tensor2_3D get_strain(const Tensor2_3D &F)
   {
      return 0.5 * (F + F.transpose()) - Tensor2_3D::identity();
   }

   const double rho_0;    // reference density
   const double lambda;   // first Lame parameter
   const double mu;       // shear modulus
};

#endif

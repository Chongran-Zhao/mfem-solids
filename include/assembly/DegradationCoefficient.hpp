// ============================================================================
// DegradationCoefficient.hpp
//
// The degradation g(alpha) = alpha^2 + eta of the elastic energy at a point,
// from the phase field alpha: the factor of the energy density of
// LocalAssembly_Disp in phase-field fracture. It reads alpha at every
// evaluation, and neither the surface energy nor alpha are its own.
//
// Author: Chongran Zhao
// Date: Nov. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef DEGRADATION_COEFFICIENT_HPP
#define DEGRADATION_COEFFICIENT_HPP

#include <mfem.hpp>

#include "ISurfaceEnergy.hpp"

class DegradationCoefficient : public mfem::Coefficient
{
public:
   DegradationCoefficient(const ISurfaceEnergy &input_surface_energy,
                          const mfem::GridFunction &input_alpha)
      : surface_energy(input_surface_energy), alpha(input_alpha) {}

   // Required by MFEM: overrides mfem::Coefficient::Eval.
   double Eval(mfem::ElementTransformation &elem_map,
               const mfem::IntegrationPoint &quad_pt) override
   {
      return surface_energy.get_g(alpha.GetValue(elem_map, quad_pt));
   }

private:
   const ISurfaceEnergy &surface_energy;   // g(alpha)
   const mfem::GridFunction &alpha;        // phase field
};

#endif

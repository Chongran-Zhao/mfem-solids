// ============================================================================
// SurfaceEnergy_AT2.hpp
//
// The AT2 surface energy of Bourdin, Francfort and Marigo (J. Elasticity
// 2008, eq. 8.2),
//    S_eps(alpha) = int ( (1 - alpha)^2 / (4 eps) + eps |Grad alpha|^2 ) dV,
// so w = (1 - alpha)^2 / (4 eps) and c_grad = eps. Since w'(1) = 0, the
// phase field drops below 1 from the start of the loading.
//
// Author: Chongran Zhao
// Date: Nov. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef SURFACE_ENERGY_AT2_HPP
#define SURFACE_ENERGY_AT2_HPP

#include "ISurfaceEnergy.hpp"

class SurfaceEnergy_AT2 : public ISurfaceEnergy
{
public:
   using ISurfaceEnergy::ISurfaceEnergy;

   // w = (1 - alpha)^2 / (4 eps)
   double get_w(double alpha) const override
   {
      return (1.0 - alpha) * (1.0 - alpha) / (4.0 * eps);
   }

   // w' = (alpha - 1) / (2 eps)
   double get_dw(double alpha) const override { return (alpha - 1.0) / (2.0 * eps); }

   // w'' = 1 / (2 eps)
   double get_d2w(double alpha) const override { return 1.0 / (2.0 * eps); }

   // c_grad = eps
   double get_c_grad() const override { return eps; }
};

#endif

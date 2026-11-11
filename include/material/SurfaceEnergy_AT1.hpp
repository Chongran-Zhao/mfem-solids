// ============================================================================
// SurfaceEnergy_AT1.hpp
//
// The AT1 surface energy, with the phase field alpha = 1 sound and 0 cracked,
//    S_eps(alpha) = 3/8 int ( (1 - alpha) / (2 eps) + 2 eps |Grad alpha|^2 ) dV,
// so w = 3 (1 - alpha) / (16 eps) and c_grad = 3 eps / 4. Since w'(1) < 0,
// the phase field stays 1 up to a critical energy, psi_e = 3 G_c / (32 eps)
// for a homogeneous state, and needs the bound alpha <= 1.
//
// Author: Chongran Zhao
// Date: Nov. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef SURFACE_ENERGY_AT1_HPP
#define SURFACE_ENERGY_AT1_HPP

#include "ISurfaceEnergy.hpp"

class SurfaceEnergy_AT1 : public ISurfaceEnergy
{
public:
   using ISurfaceEnergy::ISurfaceEnergy;

   // w = 3 (1 - alpha) / (16 eps)
   double get_w(double alpha) const override { return 3.0 * (1.0 - alpha) / (16.0 * eps); }

   // w' = -3 / (16 eps)
   double get_dw(double alpha) const override { return -3.0 / (16.0 * eps); }

   // w'' = 0
   double get_d2w(double alpha) const override { return 0.0; }

   // c_grad = 3 eps / 4
   double get_c_grad() const override { return 0.75 * eps; }
};

#endif

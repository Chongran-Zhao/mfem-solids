// ============================================================================
// SurfaceEnergy_AT1.hpp
//
// The AT1 surface energy, the AT1 damage model of mef90 (m_MEF90_DefMechAT1.F90):
//    a(alpha) = eta + (1 - eta) (1 - alpha)^2,
//    w(alpha) = alpha,   c_w = 2/3.
// Since w'(0) = 1, the damage stays zero up to a critical stress.
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

   // a = eta + (1 - eta) (1 - alpha)^2
   double get_a(double alpha) const override
   {
      return eta + (1.0 - eta) * (1.0 - alpha) * (1.0 - alpha);
   }

   // a' = -2 (1 - eta) (1 - alpha)
   double get_da(double alpha) const override
   {
      return -2.0 * (1.0 - eta) * (1.0 - alpha);
   }

   // a'' = 2 (1 - eta)
   double get_d2a(double alpha) const override { return 2.0 * (1.0 - eta); }

   // w = alpha
   double get_w(double alpha) const override { return alpha; }

   // w' = 1
   double get_dw(double alpha) const override { return 1.0; }

   // w'' = 0
   double get_d2w(double alpha) const override { return 0.0; }

   double get_c_w() const override { return 2.0 / 3.0; }
};

#endif

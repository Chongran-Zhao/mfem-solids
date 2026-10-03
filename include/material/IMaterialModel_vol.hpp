// ============================================================================
// IMaterialModel_vol.hpp
//
// Interface of the volumetric part of a hyperelastic material: given the
// volume ratio J, return its strain energy Psi_vol(J), the pressure p(J) and
// dp/dJ; given the pressure p, return the inverse J(p) and dJ/dp, used by
// the mixed formulation; and the reference density rho_0, used by the
// dynamics. As in PERIGEE, the density belongs to the volumetric model.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef IMATERIAL_MODEL_VOL_HPP
#define IMATERIAL_MODEL_VOL_HPP

class IMaterialModel_vol
{
public:
   virtual ~IMaterialModel_vol() = default;

   // Volumetric strain energy Psi_vol per reference volume.
   virtual double get_energy(double J) const = 0;

   // Pressure p = -dPsi_vol/dJ, positive in compression.
   virtual double get_p(double J) const = 0;

   // dp/dJ = -d^2Psi_vol/dJ^2.
   virtual double get_dp_dJ(double J) const = 0;

   // Volume ratio J at pressure p, the inverse of get_p.
   virtual double get_J(double p) const = 0;

   // dJ/dp, the derivative of get_J.
   virtual double get_dJ_dp(double p) const = 0;

   // Reference density rho_0, mass per reference volume.
   virtual double get_rho_0() const = 0;
};

#endif

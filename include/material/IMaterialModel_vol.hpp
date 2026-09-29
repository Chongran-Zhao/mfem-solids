// ============================================================================
// IMaterialModel_vol.hpp
//
// Interface of the volumetric part of a hyperelastic material: given the
// volume ratio J, return its strain energy Psi_vol(J) and its first two
// derivatives.
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

   // Volumetric stress sigma_vol = dPsi_vol/dJ = -p.
   virtual double get_vol_stress(double J) const = 0;

   // dsigma_vol/dJ = d^2Psi_vol/dJ^2.
   virtual double get_dvol_stress_dJ(double J) const = 0;
};

#endif

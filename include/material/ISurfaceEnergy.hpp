// ============================================================================
// ISurfaceEnergy.hpp
//
// Interface of the regularized surface energies of phase-field fracture, the
// generalized Ambrosio-Tortorelli models of mef90, its damage models
// (m_MEF90_DefMechATClass.F90), with the energy of a phase field alpha in
// [0, 1], the damage of mef90,
//    E(u, alpha) = int a(alpha) W(u) dV
//                + G_c / (4 c_w) int ( w(alpha) / ell + ell |Grad alpha|^2 ) dV,
// where alpha = 0 is the sound material and alpha = 1 the cracked one. Each
// model gives the dissipation function w(alpha) of the surface energy, the
// degradation function a(alpha) of the elastic energy, which goes with it as
// in mef90, their derivatives, and c_w = int_0^1 sqrt(w(s)) ds, which makes
// the energy of a fully developed crack G_c per unit area. This class holds
// the parameters the models share: the fracture toughness G_c, the internal
// length ell and the residual stiffness eta.
//
// Author: Chongran Zhao
// Date: Nov. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef ISURFACE_ENERGY_HPP
#define ISURFACE_ENERGY_HPP

class ISurfaceEnergy
{
public:
   ISurfaceEnergy(double input_G_c, double input_ell, double input_eta)
      : G_c(input_G_c), ell(input_ell), eta(input_eta) {}

   virtual ~ISurfaceEnergy() = default;

   // Degradation function a(alpha) and its derivatives.
   virtual double get_a(double alpha) const = 0;
   virtual double get_da(double alpha) const = 0;
   virtual double get_d2a(double alpha) const = 0;

   // Dissipation function w(alpha) and its derivatives.
   virtual double get_w(double alpha) const = 0;
   virtual double get_dw(double alpha) const = 0;
   virtual double get_d2w(double alpha) const = 0;

   // c_w = int_0^1 sqrt(w(s)) ds.
   virtual double get_c_w() const = 0;

   double get_G_c() const { return G_c; }
   double get_ell() const { return ell; }
   double get_eta() const { return eta; }

protected:
   const double G_c;   // fracture toughness
   const double ell;   // internal length
   const double eta;   // residual stiffness
};

#endif

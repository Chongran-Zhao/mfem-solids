// ============================================================================
// ISurfaceEnergy.hpp
//
// Interface of the regularized surface energies of phase-field fracture, as
// in Bourdin, Francfort and Marigo (J. Elasticity 2008, eq. 8.2), with the
// phase field alpha in [0, 1], alpha = 1 the sound material and alpha = 0
// the crack:
//    Pi(u, alpha) = int g(alpha) psi_e dV + G_c S_eps(alpha),
//    S_eps(alpha) = int ( w(alpha) + c_grad |Grad alpha|^2 ) dV,
//    g(alpha)     = alpha^2 + eta,
// with G_c the critical energy release rate, k in the paper, eps the width of
// the transition, and eta the residual stiffness, which keeps g > 0 in the
// crack; MaterialModelData of each case gives G_c, eps and eta. Each
// model gives w(alpha), its derivatives and c_grad, scaled so that a fully
// developed crack costs G_c per unit area.
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
   ISurfaceEnergy(double input_G_c, double input_eps, double input_eta)
      : G_c(input_G_c), eps(input_eps), eta(input_eta) {}

   virtual ~ISurfaceEnergy() = default;

   // g = alpha^2 + eta, the degradation of the elastic energy, and its
   // derivatives.
   double get_g(double alpha) const { return alpha * alpha + eta; }
   double get_dg(double alpha) const { return 2.0 * alpha; }
   double get_d2g(double alpha) const { return 2.0; }

   // w(alpha), the local part of the density of S_eps, and its derivatives.
   virtual double get_w(double alpha) const = 0;
   virtual double get_dw(double alpha) const = 0;
   virtual double get_d2w(double alpha) const = 0;

   // c_grad, the factor of |Grad alpha|^2 in the density of S_eps.
   virtual double get_c_grad() const = 0;

   double get_G_c() const { return G_c; }
   double get_eps() const { return eps; }
   double get_eta() const { return eta; }

protected:
   const double G_c;                            // critical energy release rate
   const double eps;                            // width of the transition
   const double eta;                            // residual stiffness
};

#endif

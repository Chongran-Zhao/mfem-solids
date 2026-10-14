// ============================================================================
// LoadData.hpp
//
// The loading of sphere_rotation: the surface of the ball, the face "outer",
// turns rigidly about the z axis through the origin, while the body starts at
// rest, without body force or traction. The angular velocity changes in two
// smooth stages, each by the bump pulse of Wan et al.: from 0 to omega_1
// over tau_1, then from omega_1 to omega_2 = -2 omega_1 over tau_2, and
// stays at omega_2 after:
//    omega(t) = omega_1 g(t / tau_1) + (omega_2 - omega_1) g((t - tau_1) / tau_2),
//    Omega(t) = omega_1 tau_1 G(t / tau_1)
//             + (omega_2 - omega_1) tau_2 G((t - tau_1) / tau_2),
// with the angle Omega the integral of omega, and
//    psi(s) = exp(1 - 1 / (1 - (2 s - 1)^2))   on (0, 1),
//    g(s)   = int_0^s psi / int_0^1 psi,
//    G(s)   = int_0^s g = s g(s) - int_0^s sigma psi / int_0^1 psi,
// extended by g = G = 0 for s <= 0, and g = 1, G = s - 1/2 for s >= 1, psi
// being symmetric about 1/2. The integrals are taken by Gauss quadrature.
//
// Author: Chongran Zhao
// Date: Oct. 11, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

#include <cmath>
#include <string>

#include <mfem.hpp>

#include "Vector_3D.hpp"

class LoadData
{
public:
   // Angular velocities about z after the two stages, and their durations.
   inline static constexpr double omega_1 = -8.57;
   inline static constexpr double omega_2 = -2.0 * omega_1;
   inline static constexpr double tau_1 = 0.01;
   inline static constexpr double tau_2 = 0.02;

   // The body starts at rest.
   static Vector_3D initial_velo(const mfem::Vector &pt)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // No body force.
   static Vector_3D body_force(const mfem::Vector &pt, double tt)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // No traction.
   static Vector_3D surface_traction(const mfem::Vector &pt, double tt,
                                     const std::string &face)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // Rigid rotation by Omega(tt) about z: u = R X - X.
   static Vector_3D disp_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      check_face(face);
      const double Omega = get_Omega(tt);
      const double cos_Omega = std::cos(Omega), sin_Omega = std::sin(Omega);
      return Vector_3D(cos_Omega * pt(0) - sin_Omega * pt(1) - pt(0),
                       sin_Omega * pt(0) + cos_Omega * pt(1) - pt(1), 0.0);
   }

   // Its velocity, v = omega(tt) e_z x R X.
   static Vector_3D velo_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      check_face(face);
      const double Omega = get_Omega(tt);
      const double omega = get_omega(tt);
      const double cos_Omega = std::cos(Omega), sin_Omega = std::sin(Omega);
      return Vector_3D(-omega * (sin_Omega * pt(0) + cos_Omega * pt(1)),
                       omega * (cos_Omega * pt(0) - sin_Omega * pt(1)), 0.0);
   }

private:
   // The only face of the ball.
   static void check_face(const std::string &face)
   {
      MFEM_VERIFY(face == "outer", "Unknown face \"" << face << "\".");
   }

   // psi(s), zero outside (0, 1).
   static double psi(double ss)
   {
      if (ss <= 0.0 || ss >= 1.0)
         return 0.0;
      const double xx = 2.0 * ss - 1.0;
      return std::exp(1.0 - 1.0 / (1.0 - xx * xx));
   }

   // int_0^s sigma^power psi(sigma) d sigma, power 0 or 1, s in [0, 1].
   static double integrate_psi(double ss, int power)
   {
      const mfem::IntegrationRule &rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 60);
      double out = 0.0;
      for (int ii = 0; ii < rule.GetNPoints(); ii++)
      {
         const double sigma = ss * rule.IntPoint(ii).x;
         out += rule.IntPoint(ii).weight * std::pow(sigma, power) * psi(sigma);
      }
      return ss * out;
   }

   // g(s), from 0 to 1.
   static double get_g(double ss)
   {
      if (ss <= 0.0)
         return 0.0;
      if (ss >= 1.0)
         return 1.0;
      static const double psi_total = integrate_psi(1.0, 0);
      return integrate_psi(ss, 0) / psi_total;
   }

   // G(s), the integral of g.
   static double get_G(double ss)
   {
      if (ss <= 0.0)
         return 0.0;
      if (ss >= 1.0)
         return ss - 0.5;
      static const double psi_total = integrate_psi(1.0, 0);
      return (ss * integrate_psi(ss, 0) - integrate_psi(ss, 1)) / psi_total;
   }

   // omega(tt).
   static double get_omega(double tt)
   {
      return omega_1 * get_g(tt / tau_1) + (omega_2 - omega_1) * get_g((tt - tau_1) / tau_2);
   }

   // Omega(tt).
   static double get_Omega(double tt)
   {
      return omega_1 * tau_1 * get_G(tt / tau_1)
             + (omega_2 - omega_1) * tau_2 * get_G((tt - tau_1) / tau_2);
   }
};

#endif

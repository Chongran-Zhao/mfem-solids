// ============================================================================
// LoadData.hpp
//
// The loading of sphere_rotation: the surface of the ball, the face "outer",
// turns rigidly about the z axis through the origin, while the body starts at
// rest, without body force or traction. The angular velocity rises smoothly
// from 0 to omega_max over the rise time tau, with the bump pulse of Wan et
// al., and then stays constant:
//    omega(t) = omega_max g(t / tau),   g(s) = int_0^s psi / int_0^1 psi,
//    psi(s)   = exp(1 - 1 / (1 - (2 s - 1)^2))   on (0, 1),
// g(s) = 1 for s >= 1. The angle is its integral, by parts,
//    theta(t) = omega_max tau (s g(s) - int_0^s sigma psi / int_0^1 psi),
// and theta(t) = omega_max (t - tau / 2) for t >= tau, psi being symmetric
// about 1/2. The integrals are taken by Gauss quadrature.
//
// It replaces include/boundary/LoadData.hpp: driver.cpp includes it first,
// and the include guard keeps the other one out.
//
// Author: Chongran Zhao
// Date: Oct. 9, 2026
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
   // Angular velocity after the rise, about z, and the rise time.
   inline static constexpr double omega_max = -8.57;
   inline static constexpr double tau = 0.01;

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

   // Rigid rotation by theta(tt) about z: u = R X - X.
   static Vector_3D disp_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      check_face(face);
      const double theta = get_theta(tt);
      const double cos_theta = std::cos(theta), sin_theta = std::sin(theta);
      return Vector_3D(cos_theta * pt(0) - sin_theta * pt(1) - pt(0),
                       sin_theta * pt(0) + cos_theta * pt(1) - pt(1), 0.0);
   }

   // Its velocity, v = omega(tt) e_z x R X.
   static Vector_3D velo_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      check_face(face);
      const double theta = get_theta(tt);
      const double omega = get_omega(tt);
      const double cos_theta = std::cos(theta), sin_theta = std::sin(theta);
      return Vector_3D(-omega * (sin_theta * pt(0) + cos_theta * pt(1)),
                       omega * (cos_theta * pt(0) - sin_theta * pt(1)), 0.0);
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

   // omega(tt).
   static double get_omega(double tt)
   {
      if (tt >= tau)
         return omega_max;
      static const double psi_total = integrate_psi(1.0, 0);
      return omega_max * integrate_psi(tt / tau, 0) / psi_total;
   }

   // theta(tt).
   static double get_theta(double tt)
   {
      if (tt >= tau)
         return omega_max * (tt - 0.5 * tau);
      static const double psi_total = integrate_psi(1.0, 0);
      const double ss = tt / tau;
      return omega_max * tau * (ss * integrate_psi(ss, 0) - integrate_psi(ss, 1)) / psi_total;
   }
};

#endif

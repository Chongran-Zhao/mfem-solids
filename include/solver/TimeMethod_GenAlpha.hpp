// ============================================================================
// TimeMethod_GenAlpha.hpp
//
// Parameters of the generalized-alpha method for second-order dynamics,
// from the spectral radius rho_inf in [0, 1] at infinite frequency. The
// intermediate states weight the new time, as in PERIGEE,
//    x_alpha = (1 - alpha) x_n + alpha x_{n+1},
// so that
//    alpha_m = (2 - rho_inf) / (1 + rho_inf),   alpha_f = 1 / (1 + rho_inf),
//    gamma   = 1/2 + alpha_m - alpha_f,         beta = (1 + alpha_m - alpha_f)^2 / 4.
// rho_inf = 1 is the midpoint rule, with no numerical damping; smaller values
// damp the high frequencies.
// It also gives the formulas the dynamic nonlinear solvers share: Newmark's
// u_pred, a_{n+1} and v_{n+1}, and the intermediate states x_alpha and
// a_alpha with the factor of M in the tangent.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_METHOD_GENALPHA_HPP
#define TIME_METHOD_GENALPHA_HPP

#include <mfem.hpp>

class TimeMethod_GenAlpha
{
public:
   TimeMethod_GenAlpha(double rho_inf)
      : alpha_m((2.0 - rho_inf) / (1.0 + rho_inf)),
        alpha_f(1.0 / (1.0 + rho_inf)),
        gamma(0.5 + alpha_m - alpha_f),
        beta(0.25 * (1.0 + alpha_m - alpha_f) * (1.0 + alpha_m - alpha_f))
   {
      MFEM_VERIFY(rho_inf >= 0.0 && rho_inf <= 1.0, "rho_inf must be in [0, 1].");
   }

   double get_alpha_m() const { return alpha_m; }
   double get_alpha_f() const { return alpha_f; }
   double get_gamma() const { return gamma; }
   double get_beta() const { return beta; }

   // t_alpha = t_n + alpha_f dt, the time of the loads of the step.
   double get_time_alpha(double time_n, double dt) const { return time_n + alpha_f * dt; }

   // The known part of Newmark's formula for u_{n+1},
   //    u_pred = u_n + dt v_n + dt^2 (1/2 - beta) a_n,
   // so that a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2).
   mfem::Vector get_disp_predict(const mfem::Vector &disp_n, const mfem::Vector &velo_n,
                                 const mfem::Vector &acce_n, double dt) const
   {
      mfem::Vector out(disp_n);
      out.Add(dt, velo_n);
      out.Add(dt * dt * (0.5 - beta), acce_n);
      return out;
   }

   // x_alpha = (1 - alpha_f) x_n + alpha_f x_{n+1}, of the displacement, and
   // in the mixed form of the pressure too.
   mfem::Vector get_state_alpha(const mfem::Vector &state_n, const mfem::Vector &state) const
   {
      mfem::Vector out(state_n);
      out *= 1.0 - alpha_f;
      out.Add(alpha_f, state);
      return out;
   }

   // a_alpha = (1 - alpha_m) a_n + alpha_m (u_{n+1} - u_pred) / (beta dt^2).
   mfem::Vector get_acce_alpha(const mfem::Vector &disp, const mfem::Vector &disp_predict,
                               const mfem::Vector &acce_n, double dt) const
   {
      mfem::Vector out(disp);
      out -= disp_predict;
      out *= get_acce_alpha_factor(dt);
      out.Add(1.0 - alpha_m, acce_n);
      return out;
   }

   // d a_alpha / d u_{n+1} = alpha_m / (beta dt^2), the factor of M in the
   // tangent of the step.
   double get_acce_alpha_factor(double dt) const { return alpha_m / (beta * dt * dt); }

   // a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2).
   mfem::Vector get_acce(const mfem::Vector &disp, const mfem::Vector &disp_predict,
                         double dt) const
   {
      mfem::Vector out(disp);
      out -= disp_predict;
      out /= beta * dt * dt;
      return out;
   }

   // v_{n+1} = v_n + dt ( (1 - gamma) a_n + gamma a_{n+1} ).
   mfem::Vector get_velo(const mfem::Vector &velo_n, const mfem::Vector &acce_n,
                         const mfem::Vector &acce, double dt) const
   {
      mfem::Vector out(velo_n);
      out.Add(dt * (1.0 - gamma), acce_n);
      out.Add(dt * gamma, acce);
      return out;
   }

private:
   const double alpha_m;   // weight of a_{n+1} in a_alpha
   const double alpha_f;   // weight of u_{n+1} in u_alpha, and of t_{n+1} in t_alpha
   const double gamma;     // v_{n+1} = v_n + dt ( (1 - gamma) a_n + gamma a_{n+1} )
   const double beta;      // u_{n+1} = u_n + dt v_n + dt^2 ( (1/2 - beta) a_n + beta a_{n+1} )
};

#endif

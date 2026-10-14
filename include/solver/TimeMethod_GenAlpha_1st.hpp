// ============================================================================
// TimeMethod_GenAlpha_1st.hpp
//
// Parameters of the generalized-alpha method for first-order dynamics, from
// the spectral radius rho_inf in [0, 1] at infinite frequency, as in PERIGEE
// solids, for the rate form
//    du/dt = v,   M dv/dt + F_int(u,p) = F_ext,   J F^-T : Grad v - J'(p) dp/dt = 0.
// Every variable y advances by
//    y_{n+1} = y_n + dt dy_n + gamma dt (dy_{n+1} - dy_n),
// the equations hold at the intermediate states
//    y_alpha  = (1 - alpha_f) y_n  + alpha_f y_{n+1},
//    dy_alpha = (1 - alpha_m) dy_n + alpha_m dy_{n+1},
// and the kinematics at du_alpha = v_alpha, with
//    alpha_m = (3 - rho_inf) / (2 (1 + rho_inf)),   alpha_f = 1 / (1 + rho_inf),
//    gamma   = 1/2 + alpha_m - alpha_f.
// rho_inf = 1 is the midpoint rule, with no numerical damping; smaller values
// damp the high frequencies. The unknowns of a step are v_{n+1} and p_{n+1};
// u_{n+1} follows from v_alpha through the kinematics, and every
// intermediate state is an affine function of v_alpha or p_alpha, whose
// factors this class also gives.
//
// Author: Chongran Zhao
// Date: Oct. 14, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_METHOD_GENALPHA_1ST_HPP
#define TIME_METHOD_GENALPHA_1ST_HPP

#include <mfem.hpp>

class TimeMethod_GenAlpha_1st
{
public:
   TimeMethod_GenAlpha_1st(double rho_inf)
      : alpha_m(0.5 * (3.0 - rho_inf) / (1.0 + rho_inf)),
        alpha_f(1.0 / (1.0 + rho_inf)),
        gamma(0.5 + alpha_m - alpha_f)
   {
      MFEM_VERIFY(rho_inf >= 0.0 && rho_inf <= 1.0, "rho_inf must be in [0, 1].");
   }

   double get_alpha_m() const { return alpha_m; }
   double get_alpha_f() const { return alpha_f; }
   double get_gamma() const { return gamma; }

   // t_alpha = t_n + alpha_f dt, the time of the loads of the step.
   double get_time_alpha(double time_n, double dt) const { return time_n + alpha_f * dt; }

   // y_alpha = (1 - alpha_f) y_n + alpha_f y_{n+1}.
   mfem::Vector get_state_alpha(const mfem::Vector &state_n, const mfem::Vector &state) const
   {
      mfem::Vector out(state_n);
      out *= 1.0 - alpha_f;
      out.Add(alpha_f, state);
      return out;
   }

   // y_{n+1} = (y_alpha - (1 - alpha_f) y_n) / alpha_f, the inverse of
   // get_state_alpha.
   mfem::Vector get_state(const mfem::Vector &state_n, const mfem::Vector &state_alpha) const
   {
      mfem::Vector out(state_alpha);
      out.Add(-(1.0 - alpha_f), state_n);
      out /= alpha_f;
      return out;
   }

   // dy_{n+1} = (y_{n+1} - y_n) / (gamma dt) - (1 - gamma) / gamma dy_n.
   mfem::Vector get_dot(const mfem::Vector &state_n, const mfem::Vector &dot_n,
                        const mfem::Vector &state, double dt) const
   {
      mfem::Vector out(state);
      out -= state_n;
      out /= gamma * dt;
      out.Add(-(1.0 - gamma) / gamma, dot_n);
      return out;
   }

   // dy_alpha = (1 - alpha_m) dy_n + alpha_m dy_{n+1}, with dy_{n+1} of
   // get_dot.
   mfem::Vector get_dot_alpha(const mfem::Vector &state_n, const mfem::Vector &dot_n,
                              const mfem::Vector &state, double dt) const
   {
      mfem::Vector out(dot_n);
      out *= 1.0 - alpha_m;
      out.Add(alpha_m, get_dot(state_n, dot_n, state, dt));
      return out;
   }

   // d dy_alpha / d y_alpha = alpha_m / (gamma dt alpha_f).
   double get_dot_alpha_factor(double dt) const { return alpha_m / (gamma * dt * alpha_f); }

   // The kinematics du_alpha = v_alpha give
   //    du_{n+1} = (v_alpha - (1 - alpha_m) du_n) / alpha_m.
   mfem::Vector get_dot_disp(const mfem::Vector &dot_disp_n, const mfem::Vector &velo_alpha) const
   {
      mfem::Vector out(velo_alpha);
      out.Add(-(1.0 - alpha_m), dot_disp_n);
      out /= alpha_m;
      return out;
   }

   // u_{n+1} = u_n + dt du_n + gamma dt (du_{n+1} - du_n), with du_{n+1} of
   // get_dot_disp.
   mfem::Vector get_disp(const mfem::Vector &disp_n, const mfem::Vector &dot_disp_n,
                         const mfem::Vector &velo_alpha, double dt) const
   {
      mfem::Vector out(disp_n);
      out.Add(dt * (1.0 - gamma), dot_disp_n);
      out.Add(gamma * dt, get_dot_disp(dot_disp_n, velo_alpha));
      return out;
   }

   // d u_alpha / d v_alpha = alpha_f gamma dt / alpha_m.
   double get_disp_alpha_factor(double dt) const { return alpha_f * gamma * dt / alpha_m; }

private:
   const double alpha_m;   // weight of dy_{n+1} in dy_alpha
   const double alpha_f;   // weight of y_{n+1} in y_alpha, and of t_{n+1} in t_alpha
   const double gamma;     // y_{n+1} = y_n + dt ( (1 - gamma) dy_n + gamma dy_{n+1} )
};

#endif

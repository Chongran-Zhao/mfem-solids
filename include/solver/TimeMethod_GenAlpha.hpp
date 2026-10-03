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

private:
   const double alpha_m;   // weight of a_{n+1} in a_alpha
   const double alpha_f;   // weight of u_{n+1} in u_alpha, and of t_{n+1} in t_alpha
   const double gamma;     // v_{n+1} = v_n + dt ( (1 - gamma) a_n + gamma a_{n+1} )
   const double beta;      // u_{n+1} = u_n + dt v_n + dt^2 ( (1/2 - beta) a_n + beta a_{n+1} )
};

#endif

// Generalized-alpha parameters for second-order displacement dynamics.
// Endpoint weights: x_alpha = (1 - alpha) x_n + alpha x_{n+1}, as in PERIGEE.
#ifndef TIME_METHOD_GENALPHA_HPP
#define TIME_METHOD_GENALPHA_HPP

#include <cmath>

#include <mfem.hpp>

class TimeMethod_GenAlpha
{
public:
   explicit TimeMethod_GenAlpha(double rho_inf)
   {
      MFEM_VERIFY(std::isfinite(rho_inf) && rho_inf >= 0.0 && rho_inf <= 1.0,
                  "rho_inf must be in [0, 1].");
      alpha_m = (2.0 - rho_inf) / (1.0 + rho_inf);
      alpha_f = 1.0 / (1.0 + rho_inf);
      gamma = 0.5 + alpha_m - alpha_f;
      beta = 0.25 * (1.0 + alpha_m - alpha_f) * (1.0 + alpha_m - alpha_f);
   }

   double get_alpha_m() const { return alpha_m; }
   double get_alpha_f() const { return alpha_f; }
   double get_gamma() const { return gamma; }
   double get_beta() const { return beta; }

private:
   double alpha_m, alpha_f, gamma, beta;
};

#endif

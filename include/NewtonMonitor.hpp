// ============================================================================
// NewtonMonitor.hpp
//
// Prints the residual norm of every Newton iteration, absolute and relative
// to the first iteration of the load step, below the header printed by
// print_load_step. NewtonSolver calls MonitorResidual once per iteration,
// and once more with final = true, which is skipped.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NEWTON_MONITOR_HPP
#define NEWTON_MONITOR_HPP

#include "mfem.hpp"
#include <iomanip>

class NewtonMonitor : public mfem::IterativeSolverMonitor
{
public:
   void MonitorResidual(int it, mfem::real_t norm, const mfem::Vector &,
                        bool final) override
   {
      if (final)
         return;
      if (it == 0)
         initial_norm = norm;
      mfem::out << std::left << std::setw(12) << it << std::scientific
                << std::setprecision(6) << std::setw(18) << norm
                << norm / initial_norm << std::defaultfloat << '\n';
   }

private:
   double initial_norm = 1.0;
};

#endif

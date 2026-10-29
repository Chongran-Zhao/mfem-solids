// ============================================================================
// TimeSteps.hpp
//
// The time steps of the time solvers, from the time section of config.yaml:
// final_time and either
//    dt         steps of dt, the last one shortened to end at final_time; a
//               last step shorter than 1e-12 dt is left out;
//    num_steps  N equal steps, t_n = final_time n / N.
// A static case with final_time 1 and num_steps N has the load factors n / N
// as its times.
//
// Author: Chongran Zhao
// Date: Oct. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_STEPS_HPP
#define TIME_STEPS_HPP

#include <cmath>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

class TimeSteps
{
public:
   // Reads final_time and dt or num_steps, exactly one of them, from the
   // time section of config.yaml.
   TimeSteps(const YAML::Node &time)
      : final_time(time["final_time"].as<double>()),
        nominal_dt(time["dt"] ? time["dt"].as<double>() : 0.0),
        num_steps(time["dt"] ? static_cast<int>(std::ceil(final_time / nominal_dt - 1.0e-12))
                             : time["num_steps"].as<int>())
   {
      MFEM_VERIFY(time["dt"].IsDefined() != time["num_steps"].IsDefined(),
                  "The time section needs either dt or num_steps.");
      MFEM_VERIFY(final_time > 0.0 && num_steps >= 1,
                  "The time section needs final_time > 0 and at least one step.");
   }

   // N, the number of the time steps.
   int get_num_steps() const { return num_steps; }

   // t_n, the time at the end of step n, 0 <= n <= N.
   double get_time(int step) const
   {
      if (nominal_dt == 0.0)
         return final_time * step / num_steps;
      return (step == num_steps) ? final_time : step * nominal_dt;
   }

private:
   const double final_time;   // t_N
   const double nominal_dt;   // dt of config.yaml, 0 with num_steps
   const int num_steps;       // N
};

#endif

// Physical-time loop for generalized-alpha displacement dynamics.
#ifndef TIME_SOLVER_DYNAMIC_DISP_HPP
#define TIME_SOLVER_DYNAMIC_DISP_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>

#include "NonlinearSolver_Dynamic_Disp.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha.hpp"

class TimeSolver_Dynamic_Disp
{
public:
   TimeSolver_Dynamic_Disp(std::unique_ptr<NonlinearSolver_Dynamic_Disp> input_solver,
                           double input_dt, double input_final_time, double rho_inf,
                           const std::filesystem::path &input_results_dir)
      : nonlinear_solver(std::move(input_solver)), time_method(rho_inf),
        dt(input_dt), final_time(input_final_time), results_dir(input_results_dir)
   {
      MFEM_VERIFY(std::isfinite(dt) && dt > 0.0 &&
                  std::isfinite(final_time) && final_time > 0.0,
                  "dt and final_time must be finite and positive.");
      SystemTools::make_empty_dir(results_dir);
   }

   void run(mfem::GridFunction &disp, mfem::GridFunction &velo, mfem::GridFunction &acce)
   {
      nonlinear_solver->initialize(0.0, disp, velo, acce);
      SystemTools::save_gf(results_dir, "disp", 0, disp);
      SystemTools::save_gf(results_dir, "velo", 0, velo);
      SystemTools::save_gf(results_dir, "acce", 0, acce);
      std::ofstream history(results_dir / "time.csv");
      MFEM_VERIFY(history, "Cannot open the dynamic time history.");
      history << "step,time,dt,newton_iterations,kinetic_energy\n"
              << std::scientific << std::setprecision(16)
              << "0,0,0,0," << nonlinear_solver->get_kinetic_energy(velo) << '\n';

      double time = 0.0;
      for (int step = 1; time < final_time; ++step)
      {
         // Shorten the last step and save its actual physical time.
         double next_time = std::min(step * dt, final_time);
         if (final_time - next_time <=
             8.0 * std::numeric_limits<double>::epsilon() * final_time)
            next_time = final_time;
         const double step_dt = next_time - time;
         MFEM_VERIFY(step_dt > 0.0, "The time step is below floating-point resolution.");
         mfem::out << std::string(74, '=') << '\n'
                   << "Time step " << step << ", t = " << next_time
                   << ", dt = " << step_dt << '\n';
         const int iterations = solve_step(time, step_dt, disp, velo, acce);
         time = next_time;
         SystemTools::save_gf(results_dir, "disp", step, disp);
         SystemTools::save_gf(results_dir, "velo", step, velo);
         SystemTools::save_gf(results_dir, "acce", step, acce);
         history << step << ',' << time << ',' << step_dt << ',' << iterations
                 << ',' << nonlinear_solver->get_kinetic_energy(velo) << '\n';
      }
   }

   // Predict the state, request a solve at the stage/end times, and commit Newmark states.
   int solve_step(double time, double dt,
                  mfem::GridFunction &disp, mfem::GridFunction &velo, mfem::GridFunction &acce)
   {
      MFEM_VERIFY(std::isfinite(dt) && dt > 0.0, "The time step must be positive.");
      MFEM_VERIFY(disp.FESpace() == velo.FESpace() && disp.FESpace() == acce.FESpace(),
                  "Dynamic fields must share one displacement space.");
      mfem::Vector pre_disp, pre_velo, pre_dot_velo;
      disp.GetTrueDofs(pre_disp); velo.GetTrueDofs(pre_velo); acce.GetTrueDofs(pre_dot_velo);
      const double alpha_m = time_method.get_alpha_m();
      const double alpha_f = time_method.get_alpha_f();
      const double acce_factor = 1.0 / (time_method.get_beta() * dt * dt);
      mfem::Vector predictor(pre_disp);
      predictor.Add(dt, pre_velo);
      predictor.Add(dt * dt * (0.5 - time_method.get_beta()), pre_dot_velo);
      mfem::Vector u(pre_disp);
      u.Add(dt, pre_velo); u.Add(0.5 * dt * dt, pre_dot_velo);
      mfem::GridFunction next_disp(disp.FESpace());
      next_disp.SetFromTrueDofs(u);
      const int iterations = nonlinear_solver->solve(time + alpha_f * dt, time + dt,
         alpha_m, alpha_f, acce_factor, pre_disp, pre_dot_velo, predictor, next_disp);
      next_disp.GetTrueDofs(u);

      mfem::Vector a(u);
      a -= pre_disp; a.Add(-dt, pre_velo);
      a.Add(-dt * dt * (0.5 - time_method.get_beta()), pre_dot_velo);
      a /= time_method.get_beta() * dt * dt;
      mfem::Vector v(pre_velo);
      v.Add(dt * (1.0 - time_method.get_gamma()), pre_dot_velo);
      v.Add(dt * time_method.get_gamma(), a);
      disp.SetFromTrueDofs(u); velo.SetFromTrueDofs(v); acce.SetFromTrueDofs(a);
      return iterations;
   }

private:
   const std::unique_ptr<NonlinearSolver_Dynamic_Disp> nonlinear_solver;
   const TimeMethod_GenAlpha time_method;
   const double dt, final_time;
   const std::filesystem::path results_dir;
};

#endif

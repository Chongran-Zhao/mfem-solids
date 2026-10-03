// ============================================================================
// TimeSolver_Static_Mixed.hpp
//
// The loop over the load steps of the static mixed form: step n = 1, ..., N
// is solved by the nonlinear solver from the converged state of step n - 1,
// and the displacement and the pressure of every step are saved as
// <results>/disp_XXXX.gf and <results>/pres_XXXX.gf. It owns the nonlinear
// solver.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_STATIC_MIXED_HPP
#define TIME_SOLVER_STATIC_MIXED_HPP

#include <filesystem>
#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>

#include "NonlinearSolver_Static_Mixed.hpp"
#include "SystemTools.hpp"

class TimeSolver_Static_Mixed
{
public:
   // Takes the ownership of the nonlinear solver; the results go to
   // results_dir, which is emptied first.
   TimeSolver_Static_Mixed(std::unique_ptr<NonlinearSolver_Static_Mixed> input_nonlinear_solver,
                           int input_num_load_steps,
                           const std::filesystem::path &input_results_dir)
      : nonlinear_solver(std::move(input_nonlinear_solver)),
        num_load_steps(input_num_load_steps),
        results_dir(input_results_dir)
   {
      SystemTools::make_empty_dir(results_dir);
   }

   // Solves the load steps 1, ..., N from disp and pres, the state of step
   // 0, and saves them at every step, step 0 included.
   void run(mfem::GridFunction &disp, mfem::GridFunction &pres)
   {
      SystemTools::save_gf(results_dir, "disp", 0, disp);
      SystemTools::save_gf(results_dir, "pres", 0, pres);

      mfem::StopWatch step_timer;
      for (int step = 1; step <= num_load_steps; step++)
      {
         // Wall-clock time of the step, up to the convergence.
         step_timer.Restart();

         mfem::out << std::string(74, '=') << '\n'
                   << "Load step " << step << " / " << num_load_steps << '\n';

         // Load factor t = n / N, the time of the loading functions.
         const double load_factor = static_cast<double>(step) / num_load_steps;
         const int iterations = nonlinear_solver->solve(load_factor, disp, pres);

         mfem::out << "converged in " << iterations
                   << " iterations. Time taken: " << std::fixed << std::setprecision(2)
                   << step_timer.RealTime() << " sec. " << SystemTools::get_time()
                   << std::defaultfloat << '\n';

         SystemTools::save_gf(results_dir, "disp", step, disp);
         SystemTools::save_gf(results_dir, "pres", step, pres);
      }
   }

private:
   const std::unique_ptr<NonlinearSolver_Static_Mixed> nonlinear_solver;  // one load step
   const int num_load_steps;                                              // N
   const std::filesystem::path results_dir;                               // folder of disp_XXXX.gf and pres_XXXX.gf
};

#endif

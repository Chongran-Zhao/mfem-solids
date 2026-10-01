// ============================================================================
// TimeSolver_Static_Disp.hpp
//
// The loop over the load steps of the static displacement form: step
// n = 1, ..., N is solved by the nonlinear solver from the converged state of
// step n - 1, and the results of every step are saved. It owns the nonlinear
// solver; what is saved is given by its caller.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_STATIC_DISP_HPP
#define TIME_SOLVER_STATIC_DISP_HPP

#include <functional>
#include <iomanip>
#include <memory>
#include <utility>
#include <mfem.hpp>
#include "NonlinearSolver_Static_Disp.hpp"
#include "SystemTools.hpp"

class TimeSolver_Static_Disp
{
public:
   // Takes the ownership of the nonlinear solver.
   TimeSolver_Static_Disp(std::unique_ptr<NonlinearSolver_Static_Disp> input_nonlinear_solver,
                          int input_num_load_steps)
      : nonlinear_solver(std::move(input_nonlinear_solver)),
        num_load_steps(input_num_load_steps) {}

   // Solves the load steps 1, ..., N from disp, the state of step 0, and
   // calls save_results(n) for every step, step 0 included.
   void run(mfem::GridFunction &disp, const std::function<void(int)> &save_results)
   {
      save_results(0);

      mfem::StopWatch step_timer;
      for (int step = 1; step <= num_load_steps; step++)
      {
         // Wall-clock time of the step, up to the convergence.
         step_timer.Restart();

         const int iterations = nonlinear_solver->solve(step, disp);

         mfem::out << "converged in " << iterations
                   << " iterations. Time taken: " << std::fixed << std::setprecision(2)
                   << step_timer.RealTime() << " sec. " << SystemTools::get_time()
                   << std::defaultfloat << '\n';

         save_results(step);
      }
   }

private:
   const std::unique_ptr<NonlinearSolver_Static_Disp> nonlinear_solver;   // one load step
   const int num_load_steps;                                              // N
};

#endif

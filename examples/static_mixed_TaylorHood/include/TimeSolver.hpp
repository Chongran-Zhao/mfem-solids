// ============================================================================
// TimeSolver.hpp
//
// The loop over the load steps of the static mixed form: step n = 1, ..., N
// is solved by the nonlinear solver from the converged state of step n - 1,
// and the displacement and the pressure of step 0, of every
// save_interval-th step and of the last are saved as
// <results>/disp_XXXX.gf and <results>/pres_XXXX.gf. Rank 0 gathers them
// from all ranks onto the serial mesh of ParMesh::GetSerialMesh, saved as
// <results>/mesh.mesh; its element order is that of the ranks, not that of
// the mesh file read by the driver. It owns the nonlinear solver.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_HPP
#define TIME_SOLVER_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>

#include "ITimeSolver.hpp"
#include "NonlinearSolver.hpp"

class TimeSolver : public ITimeSolver
{
public:
   // Takes the ownership of the nonlinear solver; the results go to
   // results_dir, which rank 0 empties first and where it saves the serial
   // mesh of mesh, with its named faces; the fields are saved every
   // input_save_interval-th step.
   TimeSolver(std::unique_ptr<NonlinearSolver> input_nonlinear_solver,
              int input_num_load_steps,
              const std::filesystem::path &input_results_dir,
              mfem::ParMesh &mesh, int input_save_interval)
      : ITimeSolver(input_results_dir, mesh, input_save_interval),
        nonlinear_solver(std::move(input_nonlinear_solver)),
        num_load_steps(input_num_load_steps)
   {
   }

   // Solves the load steps 1, ..., N from disp and pres, the state of step
   // 0, and saves them at step 0, every save_interval-th step and the last.
   void run(mfem::ParGridFunction &disp, mfem::ParGridFunction &pres)
   {
      save_fields(0, {{"disp", &disp}, {"pres", &pres}});

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
         print_converged(iterations, step_timer);

         if (is_saved(step, num_load_steps))
            save_fields(step, {{"disp", &disp}, {"pres", &pres}});
      }
   }

private:
   const std::unique_ptr<NonlinearSolver> nonlinear_solver;   // one load step
   const int num_load_steps;                                  // N
};

#endif

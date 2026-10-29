// ============================================================================
// TimeSolver.hpp
//
// The loop over the time steps of hyperelasticity in the mixed
// displacement-pressure form, static or dynamic by the nonlinear solver: the
// initial state is completed by the nonlinear solver, and step n = 1, ..., N
// is solved from the state of step n - 1, at the times of TimeSteps. The
// displacement and the pressure of step 0, of every save_interval-th step
// and of the last are saved as <results>/disp_XXXX.gf and pres_XXXX.gf, in
// the dynamics with the velocity and the acceleration, velo_XXXX.gf and
// acce_XXXX.gf, and the time of each saved step in <results>/time.csv. Rank 0 gathers the fields from
// all ranks onto the serial mesh of ParMesh::GetSerialMesh, saved as
// <results>/mesh.mesh; its element order is that of the ranks, not that of
// the mesh file read by the driver. It owns the nonlinear solver.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_HPP
#define TIME_SOLVER_HPP

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>

#include "ITimeSolver.hpp"
#include "NonlinearSolver.hpp"
#include "TimeSteps.hpp"

class TimeSolver : public ITimeSolver
{
public:
   // Takes the ownership of the nonlinear solver; the results go to
   // results_dir, which rank 0 empties first and where it saves the serial
   // mesh of mesh, with its named faces; the fields are saved every
   // input_save_interval-th step.
   TimeSolver(std::unique_ptr<NonlinearSolver> input_nonlinear_solver,
              const TimeSteps &input_time_steps,
              const std::filesystem::path &input_results_dir,
              mfem::ParMesh &mesh, int input_save_interval)
      : ITimeSolver(input_results_dir, mesh, input_save_interval),
        nonlinear_solver(std::move(input_nonlinear_solver)),
        time_steps(input_time_steps)
   {
   }

   // Completes disp, velo and acce, the initial state at t = 0 with pres,
   // solves the time steps, and saves the state of step 0, of every
   // save_interval-th step and of the last.
   void run(mfem::ParGridFunction &disp, mfem::ParGridFunction &velo,
            mfem::ParGridFunction &acce, mfem::ParGridFunction &pres)
   {
      // The fields of a step: the displacement and the pressure, in the
      // dynamics with the velocity and the acceleration.
      auto save_state = [&](int step)
      {
         if (nonlinear_solver->is_dynamic())
            save_fields(step, {{"disp", &disp}, {"velo", &velo}, {"acce", &acce}, {"pres", &pres}});
         else
            save_fields(step, {{"disp", &disp}, {"pres", &pres}});
      };

      nonlinear_solver->initialize(0.0, disp, velo, acce, pres);
      save_state(0);
      std::ofstream time_file = open_time_file();

      const int num_steps = time_steps.get_num_steps();

      mfem::StopWatch step_timer;
      double time_n = 0.0;
      for (int step = 1; step <= num_steps; step++)
      {
         // Wall-clock time of the step, up to the convergence.
         step_timer.Restart();

         const double time = time_steps.get_time(step);
         const double dt = time - time_n;

         mfem::out << std::string(74, '=') << '\n'
                   << "Time step " << step << " / " << num_steps << ", t = " << time
                   << ", dt = " << dt << '\n';

         const int iterations = nonlinear_solver->solve(time_n, time, disp, velo, acce, pres);
         print_converged(iterations, step_timer);

         if (is_saved(step, num_steps))
         {
            save_state(step);
            write_time(time_file, step, time, dt, iterations);
         }

         time_n = time;
      }
   }

private:
   const std::unique_ptr<NonlinearSolver> nonlinear_solver;   // one time step
   const TimeSteps time_steps;                                // t_n of the steps
};

#endif

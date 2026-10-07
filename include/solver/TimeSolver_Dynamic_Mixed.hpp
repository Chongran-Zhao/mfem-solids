// ============================================================================
// TimeSolver_Dynamic_Mixed.hpp
//
// The loop over the time steps of the mixed displacement-pressure dynamics:
// the initial state is completed by the nonlinear solver, and step
// n = 1, ..., N is solved from the state of step n - 1, in steps of dt up to
// the final time, the last one shortened to end there. The displacement, the
// velocity, the acceleration and the pressure of every step are saved as
// <results>/disp_XXXX.gf, velo_XXXX.gf, acce_XXXX.gf and pres_XXXX.gf, and the
// time of each step in <results>/time.csv. Rank 0 gathers the fields from
// the ranks onto the serial mesh of ParMesh::GetSerialMesh, saved as
// <results>/mesh.mesh, on which the postprocessors read them. It owns the
// nonlinear solver.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_DYNAMIC_MIXED_HPP
#define TIME_SOLVER_DYNAMIC_MIXED_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>

#include "NonlinearSolver_Dynamic_Mixed.hpp"
#include "SystemTools.hpp"

class TimeSolver_Dynamic_Mixed
{
public:
   // Takes the ownership of the nonlinear solver; the results go to
   // results_dir, which is emptied first, with the serial mesh of pmesh.
   TimeSolver_Dynamic_Mixed(std::unique_ptr<NonlinearSolver_Dynamic_Mixed> input_nonlinear_solver,
                            double input_nominal_dt, double input_final_time,
                            const std::filesystem::path &input_results_dir,
                            mfem::ParMesh &pmesh)
      : nonlinear_solver(std::move(input_nonlinear_solver)),
        nominal_dt(input_nominal_dt),
        final_time(input_final_time),
        results_dir(input_results_dir),
        serial_mesh(pmesh.GetSerialMesh(0))
   {
      SystemTools::make_empty_dir(results_dir);
      SystemTools::save_serial_mesh(results_dir, pmesh, serial_mesh);
   }

   // Completes disp, velo and acce, the initial state at t = 0 with pres,
   // solves the time steps, and saves the state at every step, step 0
   // included.
   void run(mfem::ParGridFunction &disp, mfem::ParGridFunction &velo,
            mfem::ParGridFunction &acce, mfem::ParGridFunction &pres)
   {
      // The unknowns of the nonlinear solver, the true dofs of this rank:
      // the state at t_{n+1} and at t_n, from which the step to t_{n+1} is
      // solved.
      const int num_dofs_u = disp.ParFESpace()->GetTrueVSize();
      const int num_dofs_p = pres.ParFESpace()->GetTrueVSize();
      mfem::Vector disp_true(num_dofs_u), velo_true(num_dofs_u), acce_true(num_dofs_u),
                   pres_true(num_dofs_p);
      disp.GetTrueDofs(disp_true);
      velo.GetTrueDofs(velo_true);
      acce.GetTrueDofs(acce_true);
      pres.GetTrueDofs(pres_true);

      nonlinear_solver->initialize(0.0, disp_true, velo_true, acce_true, pres_true);
      save_step(0, disp_true, velo_true, acce_true, pres_true, disp, velo, acce, pres);

      std::ofstream time_file;
      if (mfem::Mpi::Root())
      {
         time_file.open(results_dir / "time.csv");
         time_file << "step,time,dt,iterations\n" << std::scientific << std::setprecision(16)
                   << 0 << ',' << 0.0 << ',' << 0.0 << ',' << 0 << '\n';
      }

      // N steps; a last step shorter than 1e-12 dt is left out.
      const int num_steps = static_cast<int>(std::ceil(final_time / nominal_dt - 1.0e-12));

      mfem::Vector disp_n(disp_true), velo_n(velo_true), acce_n(acce_true), pres_n(pres_true);

      mfem::StopWatch step_timer;
      double time_n = 0.0;
      for (int step = 1; step <= num_steps; step++)
      {
         // Wall-clock time of the step, up to the convergence.
         step_timer.Restart();

         // t_{n+1} = step dt, the last one the final time.
         const double time = (step == num_steps) ? final_time : step * nominal_dt;
         const double dt = time - time_n;

         mfem::out << std::string(74, '=') << '\n'
                   << "Time step " << step << " / " << num_steps << ", t = " << time
                   << ", dt = " << dt << '\n';

         const int iterations = nonlinear_solver->solve(time_n, dt, disp_n, velo_n, acce_n,
                                                        pres_n, disp_true, velo_true,
                                                        acce_true, pres_true);

         mfem::out << "converged in " << iterations
                   << " iterations. Time taken: " << std::fixed << std::setprecision(2)
                   << step_timer.RealTime() << " sec. " << SystemTools::get_time()
                   << std::defaultfloat << std::setprecision(6) << '\n';

         save_step(step, disp_true, velo_true, acce_true, pres_true, disp, velo, acce, pres);
         if (mfem::Mpi::Root())
            time_file << step << ',' << time << ',' << dt << ',' << iterations << '\n';

         disp_n = disp_true;
         velo_n = velo_true;
         acce_n = acce_true;
         pres_n = pres_true;
         time_n = time;
      }
   }

private:
   // Sets the grid functions of a step from its true dofs, and saves them;
   // collective.
   void save_step(int step, const mfem::Vector &disp_true, const mfem::Vector &velo_true,
                  const mfem::Vector &acce_true, const mfem::Vector &pres_true,
                  mfem::ParGridFunction &disp, mfem::ParGridFunction &velo,
                  mfem::ParGridFunction &acce, mfem::ParGridFunction &pres)
   {
      disp.SetFromTrueDofs(disp_true);
      velo.SetFromTrueDofs(velo_true);
      acce.SetFromTrueDofs(acce_true);
      pres.SetFromTrueDofs(pres_true);
      SystemTools::save_gf(results_dir, "disp", step, disp, serial_mesh);
      SystemTools::save_gf(results_dir, "velo", step, velo, serial_mesh);
      SystemTools::save_gf(results_dir, "acce", step, acce, serial_mesh);
      SystemTools::save_gf(results_dir, "pres", step, pres, serial_mesh);
   }

   const std::unique_ptr<NonlinearSolver_Dynamic_Mixed> nonlinear_solver;  // one time step
   const double nominal_dt;                                                // dt of config.yaml
   const double final_time;                                                // end of the time steps
   const std::filesystem::path results_dir;                                // folder of the results
   mfem::Mesh serial_mesh;                                                 // all elements on rank 0, empty elsewhere
};

#endif

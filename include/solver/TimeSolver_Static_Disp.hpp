// ============================================================================
// TimeSolver_Static_Disp.hpp
//
// The loop over the load steps of the static displacement form: step
// n = 1, ..., N is solved by the nonlinear solver from the converged state of
// step n - 1, and the displacement of every step is saved as
// <results>/disp_XXXX.gf. Rank 0 gathers it from the ranks onto the serial
// mesh of ParMesh::GetSerialMesh, saved as <results>/mesh.mesh, on which the
// postprocessors read it. It owns the nonlinear solver.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_STATIC_DISP_HPP
#define TIME_SOLVER_STATIC_DISP_HPP

#include <filesystem>
#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>

#include "NonlinearSolver_Static_Disp.hpp"
#include "SystemTools.hpp"

class TimeSolver_Static_Disp
{
public:
   // Takes the ownership of the nonlinear solver; the results go to
   // results_dir, which is emptied first, with the serial mesh of pmesh.
   TimeSolver_Static_Disp(std::unique_ptr<NonlinearSolver_Static_Disp> input_nonlinear_solver,
                          int input_num_load_steps,
                          const std::filesystem::path &input_results_dir,
                          mfem::ParMesh &pmesh)
      : nonlinear_solver(std::move(input_nonlinear_solver)),
        num_load_steps(input_num_load_steps),
        results_dir(input_results_dir),
        serial_mesh(pmesh.GetSerialMesh(0))
   {
      SystemTools::make_empty_dir(results_dir);
      SystemTools::save_serial_mesh(results_dir, pmesh, serial_mesh);
   }

   // Solves the load steps 1, ..., N from disp, the state of step 0, and
   // saves disp at every step, step 0 included.
   void run(mfem::ParGridFunction &disp)
   {
      SystemTools::save_gf(results_dir, "disp", 0, disp, serial_mesh);

      // The unknowns of the nonlinear solver, the true dofs of this rank.
      mfem::Vector disp_true(disp.ParFESpace()->GetTrueVSize());
      disp.GetTrueDofs(disp_true);

      mfem::StopWatch step_timer;
      for (int step = 1; step <= num_load_steps; step++)
      {
         // Wall-clock time of the step, up to the convergence.
         step_timer.Restart();

         mfem::out << std::string(74, '=') << '\n'
                   << "Load step " << step << " / " << num_load_steps << '\n';

         // Load factor t = n / N, the time of the loading functions.
         const double load_factor = static_cast<double>(step) / num_load_steps;
         const int iterations = nonlinear_solver->solve(load_factor, disp_true);
         disp.SetFromTrueDofs(disp_true);

         mfem::out << "converged in " << iterations
                   << " iterations. Time taken: " << std::fixed << std::setprecision(2)
                   << step_timer.RealTime() << " sec. " << SystemTools::get_time()
                   << std::defaultfloat << std::setprecision(6) << '\n';

         SystemTools::save_gf(results_dir, "disp", step, disp, serial_mesh);
      }
   }

private:
   const std::unique_ptr<NonlinearSolver_Static_Disp> nonlinear_solver;   // one load step
   const int num_load_steps;                                              // N
   const std::filesystem::path results_dir;                               // folder of disp_XXXX.gf
   mfem::Mesh serial_mesh;                                                // all elements on rank 0, empty elsewhere
};

#endif

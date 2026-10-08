// ============================================================================
// TimeSolver_Static_Disp.hpp
//
// The loop over the load steps of the static displacement form: step
// n = 1, ..., N is solved by the nonlinear solver from the converged state of
// step n - 1, and the displacement of every step is saved as
// <results>/disp_XXXX.gf. Rank 0 gathers it from all ranks onto the serial
// mesh of ParMesh::GetSerialMesh, saved as <results>/mesh.mesh; its element
// order is that of the ranks, not that of the mesh file read by the driver.
// It owns the nonlinear solver.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_STATIC_DISP_HPP
#define TIME_SOLVER_STATIC_DISP_HPP

#include <filesystem>
#include <fstream>
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
   // results_dir, which rank 0 empties first and where it saves the serial
   // mesh of mesh, with its named faces.
   TimeSolver_Static_Disp(std::unique_ptr<NonlinearSolver_Static_Disp> input_nonlinear_solver,
                          int input_num_load_steps,
                          const std::filesystem::path &input_results_dir,
                          mfem::ParMesh &mesh)
      : nonlinear_solver(std::move(input_nonlinear_solver)),
        num_load_steps(input_num_load_steps),
        results_dir(input_results_dir),
        serial_mesh(mesh.GetSerialMesh(0))
   {
      if (mfem::Mpi::Root())
      {
         SystemTools::make_empty_dir(results_dir);
         mesh.bdr_attribute_sets.Copy(serial_mesh.bdr_attribute_sets);
         std::ofstream mesh_file(results_dir / "mesh.mesh");
         mesh_file.precision(16);
         serial_mesh.Print(mesh_file);
      }
      MPI_Barrier(mesh.GetComm());
   }

   // Solves the load steps 1, ..., N from disp, the state of step 0, and
   // saves disp at every step, step 0 included.
   void run(mfem::ParGridFunction &disp)
   {
      save_disp(0, disp);

      mfem::StopWatch step_timer;
      for (int step = 1; step <= num_load_steps; step++)
      {
         // Wall-clock time of the step, up to the convergence.
         step_timer.Restart();

         mfem::out << std::string(74, '=') << '\n'
                   << "Load step " << step << " / " << num_load_steps << '\n';

         // Load factor t = n / N, the time of the loading functions.
         const double load_factor = static_cast<double>(step) / num_load_steps;
         const int iterations = nonlinear_solver->solve(load_factor, disp);

         mfem::out << "converged in " << iterations
                   << " iterations. Time taken: " << std::fixed << std::setprecision(2)
                   << step_timer.RealTime() << " sec. " << SystemTools::get_time()
                   << std::defaultfloat << std::setprecision(6) << '\n';

         save_disp(step, disp);
      }
   }

private:
   // Gathers disp onto serial_mesh and saves it on rank 0; every rank takes
   // part.
   void save_disp(int step, const mfem::ParGridFunction &disp)
   {
      const mfem::GridFunction serial_disp = disp.GetSerialGridFunction(0, serial_mesh);
      if (mfem::Mpi::Root())
         SystemTools::save_gf(results_dir, "disp", step, serial_disp);
   }

   const std::unique_ptr<NonlinearSolver_Static_Disp> nonlinear_solver;   // one load step
   const int num_load_steps;                                              // N
   const std::filesystem::path results_dir;                               // folder of disp_XXXX.gf
   mfem::Mesh serial_mesh;                                                // all elements on rank 0, empty elsewhere
};

#endif

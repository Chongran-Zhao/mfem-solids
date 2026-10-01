// ============================================================================
// TimeSolver_Static_Mixed.hpp
//
// Mixed load-step loop and displacement, pressure and element-center stress
// output. Owns the nonlinear solver; the solution belongs to the caller.
// Field views are constructed inside run; stress calculation is delegated
// to assembly.
//
// Author: Chongran Zhao
// Date: Oct. 1, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_STATIC_MIXED_HPP
#define TIME_SOLVER_STATIC_MIXED_HPP

#include <filesystem>
#include <iomanip>
#include <memory>
#include <utility>
#include <mfem.hpp>
#include "NonlinearSolver_Static_Mixed.hpp"
#include "SystemTools.hpp"

class TimeSolver_Static_Mixed
{
public:
   TimeSolver_Static_Mixed(std::unique_ptr<NonlinearSolver_Static_Mixed> input_nonlinear_solver,
                           int input_num_load_steps,
                           const std::filesystem::path &input_results_dir)
      : nonlinear_solver(std::move(input_nonlinear_solver)),
        num_load_steps(input_num_load_steps), results_dir(input_results_dir)
   {
      SystemTools::make_empty_dir(results_dir);
   }

   void run(mfem::BlockVector &sol)
   {
      mfem::GridFunction disp, pres;
      nonlinear_solver->make_solution_views(sol, disp, pres);
      mfem::Mesh &mesh = *disp.FESpace()->GetMesh();
      mfem::L2_FECollection fec_stress(0, mesh.Dimension());
      mfem::FiniteElementSpace space_stress(&mesh, &fec_stress, 9, mfem::Ordering::byVDIM);
      mfem::GridFunction stress(&space_stress);
      save_results(0, disp, pres, stress);

      mfem::StopWatch step_timer;
      for (int step = 1; step <= num_load_steps; step++)
      {
         step_timer.Restart();
         mfem::out << std::string(74, '=') << '\n'
                   << "Load step " << step << " / " << num_load_steps << '\n';
         const double load_factor = static_cast<double>(step) / num_load_steps;
         const int iterations = nonlinear_solver->solve(load_factor, sol);
         mfem::out << "converged in " << iterations
                   << " iterations. Time taken: " << std::fixed << std::setprecision(2)
                   << step_timer.RealTime() << " sec. " << SystemTools::get_time()
                   << std::defaultfloat << '\n';
         save_results(step, disp, pres, stress);
      }
   }

private:
   void save_results(int step, const mfem::GridFunction &disp,
                     const mfem::GridFunction &pres, mfem::GridFunction &stress) const
   {
      SystemTools::save_gf(results_dir, "disp", step, disp);
      SystemTools::save_gf(results_dir, "pres", step, pres);
      nonlinear_solver->set_center_stress(disp, pres, stress);
      SystemTools::save_gf(results_dir, "stress", step, stress);
   }

   const std::unique_ptr<NonlinearSolver_Static_Mixed> nonlinear_solver;
   const int num_load_steps;
   const std::filesystem::path results_dir;
};

#endif

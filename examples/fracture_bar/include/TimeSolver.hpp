// ============================================================================
// TimeSolver.hpp
//
// The loop over the load steps of phase-field fracture by alternate
// minimization, as in the notes Phase-field-fracture.pdf (p. 6, 7) and
// Algorithm 1 of Bourdin, Francfort and Marigo (J. Elasticity 2008): at the
// load step of time t,
//    1. the crack set K_n, the nodes where alpha_n <= eta_lim, is fixed to
//       alpha = 0 for the whole step;
//    2. repeat: solve the displacement with alpha fixed, then the phase
//       field with the displacement fixed,
//       until ||alpha^(j+1) - alpha^(j)||_inf <= tol.
// The first iteration of each step starts from the alpha of the previous
// one, 1 at the first step. Each step prints one block per iteration: the
// Newton iterations of each problem, indented, then
// ||alpha - alpha_old||_inf and the energy E(u, alpha), as a check that it
// decreases. The displacement and the phase
// field of step 0, of every save_interval-th step and of the last are saved
// as <results>/disp_XXXX.gf and alpha_XXXX.gf, and the time of each saved
// step, with its number of iterations, in <results>/time.csv; there the
// iterations are the alternate iterations of the step, not the Newton ones.
// It owns the two nonlinear solvers.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TIME_SOLVER_HPP
#define TIME_SOLVER_HPP

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
#include <yaml-cpp/yaml.h>

#include "ITimeSolver.hpp"
#include "NonlinearSolver_Disp.hpp"
#include "NonlinearSolver_Fracture.hpp"
#include "TimeSteps.hpp"

class TimeSolver : public ITimeSolver
{
public:
   // Takes the ownership of the two nonlinear solvers; eta_lim, tol and
   // max_iter come from the fracture section of config.yaml; the results go
   // to results_dir, which rank 0 empties first and where it saves the
   // serial mesh of mesh, with its named faces; the fields are saved every
   // input_save_interval-th step.
   TimeSolver(std::unique_ptr<NonlinearSolver_Disp> input_solver_disp,
              std::unique_ptr<NonlinearSolver_Fracture> input_solver_fracture,
              const YAML::Node &fracture, const TimeSteps &input_time_steps,
              const std::filesystem::path &input_results_dir,
              mfem::ParMesh &mesh, int input_save_interval)
      : ITimeSolver(input_results_dir, mesh, input_save_interval),
        solver_disp(std::move(input_solver_disp)),
        solver_fracture(std::move(input_solver_fracture)),
        eta_lim(fracture["eta_lim"].as<double>()),
        tol(fracture["tol"].as<double>()),
        max_iter(fracture["max_iter"].as<int>()),
        time_steps(input_time_steps)
   {
   }

   // Solves the load steps from disp and alpha, the state at t = 0, and
   // saves the state of step 0, of every save_interval-th step and of the
   // last.
   void run(mfem::ParGridFunction &disp, mfem::ParGridFunction &alpha)
   {
      save_fields(0, {{"disp", &disp}, {"alpha", &alpha}});
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

         // 1. The crack set K_n, from the alpha of the previous step.
         const int crack_size = solver_fracture->set_crack_set(alpha, eta_lim);
         mfem::out << "crack set K_n: " << crack_size << " dofs with alpha <= " << eta_lim
                   << '\n';

         // 2. The alternate minimization.
         const int iterations = solve_alternate(time, disp, alpha);
         print_converged(iterations, step_timer);

         if (is_saved(step, num_steps))
         {
            save_fields(step, {{"disp", &disp}, {"alpha", &alpha}});
            write_time(time_file, step, time, dt, iterations);
         }

         time_n = time;
      }
   }

private:
   // Alternate minimization at time: disp and alpha are solved in turn until
   // the largest change of alpha is at most tol; aborts unless it converges
   // in max_iter iterations. Returns the number of iterations.
   int solve_alternate(double time, mfem::ParGridFunction &disp, mfem::ParGridFunction &alpha)
   {
      solver_disp->print_load(time, disp);

      mfem::ParGridFunction alpha_old(alpha.ParFESpace());
      for (int iter = 1; iter <= max_iter; iter++)
      {
         alpha_old = alpha;

         // One block per iteration: the Newton iterations of each problem,
         // then the change of alpha and E; a blank line before the first,
         // a dashed line between two.
         mfem::out << ((iter == 1) ? std::string("\n") : std::string(74, '-') + '\n')
                   << "alternate iteration " << iter << '\n';
         mfem::out << "  --> displacement u\n";
         solver_disp->solve(time, disp);
         mfem::out << "  --> phase field alpha\n";
         solver_fracture->solve(time, alpha);

         const double change = get_max_change(alpha, alpha_old);
         const double energy = solver_fracture->get_energy(alpha);
         mfem::out << "  max|Delta alpha| = " << std::scientific << std::setprecision(6)
                   << change << ", energy = " << std::setprecision(12) << energy
                   << std::defaultfloat << std::setprecision(6) << '\n';

         if (change <= tol)
            return iter;
      }
      MFEM_ABORT("Alternate minimization did not converge at t = " << time << ".");
      return max_iter;
   }

   // max |alpha - alpha_old| over the dofs of all ranks.
   static double get_max_change(const mfem::ParGridFunction &alpha,
                                const mfem::ParGridFunction &alpha_old)
   {
      double change = 0.0;
      for (int dof = 0; dof < alpha.Size(); dof++)
         change = std::max(change, std::fabs(alpha(dof) - alpha_old(dof)));
      MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX,
                    alpha.ParFESpace()->GetComm());
      return change;
   }

   const std::unique_ptr<NonlinearSolver_Disp> solver_disp;           // u with alpha fixed
   const std::unique_ptr<NonlinearSolver_Fracture> solver_fracture;   // alpha with u fixed
   const double eta_lim;     // alpha_n <= eta_lim puts a node in K_n
   const double tol;         // tolerance on ||alpha^(j+1) - alpha^(j)||_inf
   const int max_iter;       // maximum of alternate iterations per step
   const TimeSteps time_steps;   // t_n of the steps
};

#endif

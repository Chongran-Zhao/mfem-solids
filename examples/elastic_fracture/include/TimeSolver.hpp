// ============================================================================
// TimeSolver.hpp
//
// The loop over the load steps of phase-field fracture by alternate
// minimization, as in the notes Phase-field-fracture.pdf (p. 6, 7) and
// Algorithm 1 of Bourdin, Francfort and Marigo (J. Elasticity 2008): at the
// load step of time t, from step 0 at t = 0,
//    1. alpha is bounded by alpha_n, alpha <= alpha_n, at every node for
//       the whole step, and set to 0 on the crack set K_n, the nodes where
//       alpha_n <= eta_lim;
//    2. repeat: solve the displacement with alpha fixed, then the phase
//       field with the displacement fixed,
//       until ||alpha^(j+1) - alpha^(j)||_inf <= tol.
//       With omega != 1 each solve is over-relaxed, after Algorithm 2 (ORAM)
//       of Farrell and Maurini (IJNME 2017): u = u_old + omega (u~ - u_old)
//       on the free dofs, and alpha = alpha_old + omega (alpha~ - alpha_old)
//       projected onto the bounds of the step dof by dof; omega = 1 is the
//       alternate minimization above. Four choices are ours, not the
//       paper's: the projection, where the paper halves one omega for all
//       dofs; the change of the convergence test is that of the solve,
//       alpha~ - alpha_old, not of the relaxed alpha, which omega would
//       enlarge; the first iteration of a step is not relaxed, its change
//       being that of the load, which the relaxation would overshoot; and
//       the converged step keeps the u~ and alpha~ of the last solves, since
//       the relaxed u would keep the error (1 - omega)^k of the relaxation.
// The first iteration of each step starts from the alpha of the previous
// one; step 0, without load, starts from the alpha given, 1 but 0 on the
// initial crack, and forms the profile of alpha around the crack, as mef90
// solves its first time. Each step prints one block per iteration: the
// Newton iterations of each problem, indented, then
// ||alpha - alpha_old||_inf. Once a step converges it prints the energy
// E(u, alpha), elastic, surface and E = elastic - work + surface, with the
// work F_ext . u of the external force, and the energy balance: the work
// W_ext of the loads over the steps so far, by the trapezoidal rule
// 1/2 (f_n + f_n+1) . (u_n+1 - u_n) with the internal force f at every dof,
// which is the reaction on the constrained ones, so that W_ext counts the
// prescribed displacement, and the energy dissipated,
// W_ext - elastic - (surface - surface of step 0). It stays near zero while
// the crack grows smoothly and jumps when it grows brutally. The
// displacement and the phase
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
   // Takes the ownership of the two nonlinear solvers; eta_lim, tol,
   // max_iter and omega come from the fracture section of config.yaml; the results go
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
        omega(fracture["omega"].as<double>()),
        time_steps(input_time_steps)
   {
      MFEM_VERIFY(omega > 0.0 && omega < 2.0,
                  "The over-relaxation omega must lie in (0, 2), not " << omega << ".");
   }

   // Solves the load steps from disp and alpha, the initial state, step 0
   // at t = 0 first, and saves the state of step 0, of every
   // save_interval-th step and of the last.
   void run(mfem::ParGridFunction &disp, mfem::ParGridFunction &alpha)
   {
      // time.csv starts with the row of step 0.
      std::ofstream time_file = open_time_file();

      const int num_steps = time_steps.get_num_steps();

      // The internal force and the displacement of the previous step, on the
      // dofs this rank owns, zero before step 0, for the work W_ext of the
      // loads; the surface energy of step 0, that of the initial crack.
      const int num_dofs = disp.ParFESpace()->GetTrueVSize();
      mfem::Vector force_n(num_dofs), disp_n(num_dofs), disp_owned(num_dofs), force(num_dofs);
      force_n = 0.0;
      disp_n = 0.0;
      double external_work = 0.0;
      double surface_0 = 0.0;

      mfem::StopWatch step_timer;
      double time_n = 0.0;
      for (int step = 0; step <= num_steps; step++)
      {
         // Wall-clock time of the step, up to the convergence.
         step_timer.Restart();

         const double time = time_steps.get_time(step);
         const double dt = time - time_n;

         mfem::out << std::string(74, '=') << '\n'
                   << "Time step " << step << " / " << num_steps << ", t = " << time
                   << ", dt = " << dt << '\n';

         // 1. The bounds alpha <= alpha_n and the crack set K_n, from the
         //    alpha of the previous step.
         const int crack_size = solver_fracture->set_bounds(alpha, eta_lim);
         mfem::out << "crack set K_n: " << crack_size << " dofs with alpha <= " << eta_lim
                   << '\n';

         // 2. The alternate minimization.
         const int iterations = solve_alternate(step, num_steps, time, disp, alpha);
         print_converged(iterations, step_timer);

         // 3. The energy and the energy balance; the phase-field problem
         //    gives elastic + surface.
         solver_disp->get_internal_force(disp, force);
         const MPI_Comm comm = disp.ParFESpace()->GetComm();
         disp.GetTrueDofs(disp_owned);
         mfem::Vector disp_change(disp_owned), force_mean(force);
         disp_change -= disp_n;
         force_mean += force_n;
         force_mean *= 0.5;
         external_work += mfem::InnerProduct(comm, force_mean, disp_change);
         const double elastic_energy = solver_disp->get_energy(disp);
         const double surface_energy = solver_fracture->get_energy(alpha) - elastic_energy;
         const double work = solver_disp->get_external_work(disp);
         if (step == 0)
            surface_0 = surface_energy;
         mfem::out << std::scientific << std::setprecision(6)
                   << "  energy: elastic = " << elastic_energy << ", surface = "
                   << surface_energy << ", total = " << elastic_energy - work + surface_energy
                   << '\n' << "  balance: W_ext = " << external_work << ", dissipated = "
                   << external_work - elastic_energy - (surface_energy - surface_0)
                   << std::defaultfloat << '\n';
         force_n = force;
         disp_n = disp_owned;

         if (is_saved(step, num_steps))
         {
            save_fields(step, {{"disp", &disp}, {"alpha", &alpha}});
            if (step > 0)
               write_time(time_file, step, time, dt, iterations);
         }

         time_n = time;
      }
   }

private:
   // Alternate minimization at time: disp and alpha are solved in turn until
   // the largest change of alpha is at most tol; aborts unless it converges
   // in max_iter iterations. Returns the number of iterations. step of
   // num_steps is the load step of time, printed with each iteration.
   int solve_alternate(int step, int num_steps, double time, mfem::ParGridFunction &disp,
                       mfem::ParGridFunction &alpha)
   {
      solver_disp->print_load(time, disp);

      mfem::ParGridFunction alpha_old(alpha.ParFESpace()), disp_old(disp.ParFESpace());
      mfem::ParGridFunction disp_solved(disp.ParFESpace());
      for (int iter = 1; iter <= max_iter; iter++)
      {
         alpha_old = alpha;
         disp_old = disp;

         // One block per iteration: the Newton iterations of each problem,
         // then the change of alpha; a blank line before the first,
         // a dashed line between two. The header names the load step, lost
         // from sight when a step takes many iterations.
         mfem::out << ((iter == 1) ? std::string("\n") : std::string(74, '-') + '\n')
                   << "alternate iteration " << iter << " (time step " << step << " / "
                   << num_steps << ", t = " << time << ")\n";
         mfem::out << "  --> displacement u\n";
         solver_disp->solve(time, disp);
         const bool relaxed = omega != 1.0 && iter > 1;
         disp_solved = disp;
         if (relaxed)
            solver_disp->relax(disp_old, disp, omega);
         mfem::out << "  --> phase field alpha\n";
         solver_fracture->solve(time, alpha);

         // The change of the solve, before any relaxation.
         const double change = get_max_change(alpha, alpha_old);
         mfem::out << "  max|Delta alpha| = " << std::scientific << std::setprecision(6)
                   << change << std::defaultfloat << '\n';

         if (change <= tol)
         {
            disp = disp_solved;
            return iter;
         }
         if (relaxed)
         {
            const int projected = solver_fracture->relax(alpha_old, alpha, omega);
            mfem::out << "  relaxed by omega = " << omega << ", " << projected
                      << " dofs projected onto the bounds\n";
         }
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
   const double omega;       // over-relaxation of the alternate minimization, 1 none
   const TimeSteps time_steps;   // t_n of the steps
};

#endif

// ============================================================================
// TimeSolver.hpp
//
// The loop over the load steps of phase-field fracture by alternate
// minimization, as in the notes Phase-field-fracture.pdf (p. 6, 7) and
// Algorithm 1 of Bourdin, Francfort and Marigo (J. Elasticity 2008): at the
// load step of time t, from step 0 at t = 0,
//    1. alpha is bounded by alpha_n, alpha <= alpha_n, at every node for
//       the whole step;
//    2. repeat: solve the displacement with alpha fixed, then the phase
//       field with the displacement fixed,
//       until ||alpha^(j+1) - alpha^(j)||_inf <= tol.
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
// With backtracking: true in the fracture section of config.yaml, the
// backtracking of Bourdin, Francfort and Marigo (J. Elasticity 2008),
// Section 8.2.2, Algorithm 2, also Algorithm 2 of Bourdin (2007): the loads
// being proportional to t and the bulk energy quadratic in u, the state of
// step i scaled to an earlier step j, ((t_j / t_i) u_i, alpha_i), is
// admissible at step j, of energy (t_j / t_i)^2 (elastic_i - work_i)
// + surface_i. After each step i, the first step j = 1, ..., i - 1 where
//    E_j - (t_j / t_i)^2 (elastic_i - work_i) - surface_i >= backtracking_tol,
// condition (8.15) violated, is solved again: from u = (t_j / t_i) u_i and
// alpha = alpha_i, with the bound alpha <= alpha_(j-1), not alpha_i; the
// steps from j on are dropped, their fields and rows of time.csv too. The
// state of every step is kept in memory for it.
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
#include <sstream>
#include <utility>
#include <vector>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "ITimeSolver.hpp"
#include "NonlinearSolver_Disp.hpp"
#include "NonlinearSolver_Fracture.hpp"
#include "TimeSteps.hpp"

class TimeSolver : public ITimeSolver
{
public:
   // Takes the ownership of the two nonlinear solvers; tol and max_iter
   // come from the fracture section of config.yaml; the results go
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
        tol(fracture["tol"].as<double>()),
        max_iter(fracture["max_iter"].as<int>()),
        backtracking(fracture["backtracking"] ? fracture["backtracking"].as<bool>() : false),
        backtracking_tol(backtracking ? fracture["backtracking_tol"].as<double>() : 0.0),
        time_steps(input_time_steps)
   {
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

      // The converged steps so far, step k at history[k], kept only with
      // backtracking.
      std::vector<step_state> history;

      mfem::StopWatch step_timer;
      double time_n = 0.0;
      int step = 0;
      while (step <= num_steps)
      {
         // Wall-clock time of the step, up to the convergence.
         step_timer.Restart();

         const double time = time_steps.get_time(step);
         const double dt = time - time_n;

         mfem::out << std::string(74, '=') << '\n'
                   << "Time step " << step << " / " << num_steps << ", t = " << time
                   << ", dt = " << dt << '\n';

         // 1. The bounds alpha <= alpha_n, from the alpha of the previous step;
         //    with backtracking, the one kept, alpha_(j-1) after going back
         //    to step j, while alpha starts from alpha_i.
         if (backtracking && step > 0)
         {
            mfem::ParGridFunction alpha_n(alpha.ParFESpace());
            alpha_n.SetFromTrueDofs(history[step - 1].alpha);
            solver_fracture->set_bounds(alpha_n);
         }
         else
            solver_fracture->set_bounds(alpha);

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

         // 4. The backtracking: back to the first earlier step whose energy
         //    the scaled state of this one beats.
         if (backtracking)
         {
            step_state state{time, dt, iterations, elastic_energy, work, surface_energy,
                             external_work, mfem::Vector(), disp_owned, force};
            state.alpha.SetSize(alpha.ParFESpace()->GetTrueVSize());
            alpha.GetTrueDofs(state.alpha);
            history.push_back(std::move(state));

            const int target = find_backtrack(history);
            if (target > 0)
            {
               // u = (t_j / t_i) u_i, alpha = alpha_i; the state of step
               // j - 1 for the energy balance; the steps from j on dropped.
               disp *= history[target].time / time;
               const step_state &previous = history[target - 1];
               force_n = previous.force;
               disp_n = previous.disp;
               external_work = previous.external_work;
               time_n = previous.time;
               drop_steps(target, step, num_steps, history, time_file);
               step = target;
               continue;
            }
         }
         step++;
      }
   }

private:
   // The converged state of a step, kept for the backtracking: energies, the
   // work W_ext so far, and alpha, u and the internal force on the dofs this
   // rank owns.
   struct step_state
   {
      double time;            // t of the step
      double dt;              // t - t of the previous step
      int iterations;         // alternate iterations
      double elastic;         // elastic energy
      double work;            // work F_ext . u of the external force
      double surface;         // surface energy
      double external_work;   // W_ext of the loads up to this step
      mfem::Vector alpha;     // alpha, owned dofs
      mfem::Vector disp;      // u, owned dofs
      mfem::Vector force;     // internal force, owned dofs
   };

   // The first step j = 1, ..., i - 1 at which the state of the last step i
   // of history, scaled to t_j, has an energy lower by backtracking_tol or
   // more, condition (8.15) of Bourdin, Francfort and Marigo (2008)
   // violated; 0 if none. Prints the step found.
   int find_backtrack(const std::vector<step_state> &history) const
   {
      const int last = static_cast<int>(history.size()) - 1;
      const step_state &current = history[last];
      for (int earlier = 1; earlier < last; earlier++)
      {
         const step_state &state = history[earlier];
         const double ratio = state.time / current.time;
         const double scaled = ratio * ratio * (current.elastic - current.work) + current.surface;
         const double energy = state.elastic - state.work + state.surface;
         if (energy - scaled >= backtracking_tol)
         {
            mfem::out << std::scientific << std::setprecision(6)
                      << "  backtracking: step " << last << " -> step " << earlier
                      << ", E = " << energy << ", scaled E of step " << last << " = "
                      << scaled << ", difference = " << energy - scaled
                      << std::defaultfloat << '\n';
            return earlier;
         }
      }
      return 0;
   }

   // Drops the steps from target to last, solved again: their states, the
   // fields saved of them, and time.csv rewritten with the saved steps
   // before target.
   void drop_steps(int target, int last, int num_steps, std::vector<step_state> &history,
                   std::ofstream &time_file)
   {
      history.resize(target);
      if (mfem::Mpi::Root())
      {
         for (int step = target; step <= last; step++)
            for (const char *name : {"disp", "alpha"})
            {
               std::ostringstream file_name;
               file_name << name << '_' << std::setw(4) << std::setfill('0') << step << ".gf";
               std::filesystem::remove(results_dir / file_name.str());
            }
         time_file.close();
      }
      time_file = open_time_file();
      for (int step = 1; step < target; step++)
         if (is_saved(step, num_steps))
            write_time(time_file, step, history[step].time, history[step].dt,
                       history[step].iterations);
   }

   // Alternate minimization at time: disp and alpha are solved in turn until
   // the largest change of alpha is at most tol; aborts unless it converges
   // in max_iter iterations. Returns the number of iterations. step of
   // num_steps is the load step of time, printed with each iteration.
   int solve_alternate(int step, int num_steps, double time, mfem::ParGridFunction &disp,
                       mfem::ParGridFunction &alpha)
   {
      solver_disp->print_load(time, disp);

      mfem::ParGridFunction alpha_old(alpha.ParFESpace());
      for (int iter = 1; iter <= max_iter; iter++)
      {
         alpha_old = alpha;

         // One block per iteration: the Newton iterations of each problem,
         // then the change of alpha; a blank line before the first,
         // a dashed line between two. The header names the load step, lost
         // from sight when a step takes many iterations.
         mfem::out << ((iter == 1) ? std::string("\n") : std::string(74, '-') + '\n')
                   << "alternate iteration " << iter << " (time step " << step << " / "
                   << num_steps << ", t = " << time << ")\n";
         mfem::out << "  --> displacement u\n";
         solver_disp->solve(time, disp);
         mfem::out << "  --> phase field alpha\n";
         solver_fracture->solve(time, alpha);

         const double change = get_max_change(alpha, alpha_old);
         mfem::out << "  max|Delta alpha| = " << std::scientific << std::setprecision(6)
                   << change << std::defaultfloat << '\n';

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
   const double tol;         // tolerance on ||alpha^(j+1) - alpha^(j)||_inf
   const int max_iter;       // maximum of alternate iterations per step
   const bool backtracking;          // whether to backtrack
   const double backtracking_tol;    // delta of the backtracking, in J
   const TimeSteps time_steps;   // t_n of the steps
};

#endif

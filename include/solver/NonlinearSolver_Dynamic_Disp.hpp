// ============================================================================
// NonlinearSolver_Dynamic_Disp.hpp
//
// Solves one time step of the displacement dynamics,
//    M a_{n+1} + F_int(u_{n+1}) = F_ext(t_{n+1}),
// with the generalized-alpha method: the equation holds at the intermediate
// states u_alpha, a_alpha and the time t_alpha, and Newmark's formulas give
// a_{n+1} and v_{n+1} from u_{n+1}, the unknown of Newton's method. It also
// computes the initial acceleration. It owns the global assembly, the mass
// matrix, the time method and the solvers; the loop over the time steps is
// left to its caller. It is also the mfem::Operator that its NewtonSolver
// solves, through Mult and GetGradient.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_DYNAMIC_DISP_HPP
#define NONLINEAR_SOLVER_DYNAMIC_DISP_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha.hpp"

class NonlinearSolver_Dynamic_Disp : public mfem::Operator
{
public:
   // Takes the ownership of the global assembly and of the time method, and
   // assembles the mass matrix with the reference density; the Newton
   // settings come from the solver section of config.yaml.
   NonlinearSolver_Dynamic_Disp(std::unique_ptr<GlobalAssembly_Disp> input_global_assembly,
                                double density,
                                std::unique_ptr<TimeMethod_GenAlpha> input_time_method,
                                const YAML::Node &solver)
      : mfem::Operator(input_global_assembly->get_num_dofs()),
        global_assembly(std::move(input_global_assembly)),
        time_method(std::move(input_time_method)),
        mass(global_assembly->assemble_mass(density))
   {
      newton_solver.SetOperator(*this);
      newton_solver.SetSolver(linear_solver);
      newton_solver.SetRelTol(solver["newton_rel_tol"].as<double>());
      newton_solver.SetAbsTol(solver["newton_abs_tol"].as<double>());
      newton_solver.SetMaxIter(solver["newton_max_iter"].as<int>());
      newton_solver.SetPrintLevel(-1);
      newton_solver.iterative_mode = true;
      newton_solver.SetMonitor(newton_monitor);
   }

   // The initial state at time tt: sets the boundary values of disp and
   // velo, and solves the equation of motion for the initial acceleration,
   //    M a_0 = F_ext(t_0) - F_int(u_0),
   // with the prescribed acceleration on the constrained dofs.
   void initialize(double tt, mfem::GridFunction &disp, mfem::GridFunction &velo,
                   mfem::GridFunction &acce)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();

      dirichlet.apply_fixed_bc(disp);
      dirichlet.apply_fixed_bc(velo);
      acce = 0.0;
      if (dirichlet.is_disp_load())
      {
         dirichlet.apply_disp_load_bc(tt, disp);
         dirichlet.apply_velo_load_bc(tt, velo);
         dirichlet.apply_acce_load_bc(tt, acce);
      }
      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);

      // rhs = -R(u_0); the prescribed acceleration in acce goes to the
      // right-hand side as the prescribed increment of the static predictor.
      mfem::Vector rhs(global_assembly->get_num_dofs());
      global_assembly->assemble_residual(disp, rhs);
      rhs.Neg();
      mfem::SparseMatrix constrained_mass(*mass);
      global_assembly->set_essential_bdr(constrained_mass, acce, rhs);

      linear_solver.SetOperator(constrained_mass);
      linear_solver.Mult(rhs, acce);
   }

   // Solves the step from time tt to tt + input_dt, from the state of time tt
   // into disp, velo and acce of time tt + input_dt, and returns the number
   // of Newton iterations.
   int solve(double tt, double input_dt, const mfem::GridFunction &input_disp_old,
             const mfem::GridFunction &input_velo_old, const mfem::GridFunction &input_acce_old,
             mfem::GridFunction &disp, mfem::GridFunction &velo, mfem::GridFunction &acce)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();
      const double gamma = time_method->get_gamma();
      const double beta = time_method->get_beta();

      set_step(tt, input_dt, input_disp_old, input_velo_old, input_acce_old);

      // Newton's method starts from u_pred, the displacement with
      // a_{n+1} = 0, corrected by the consistent predictor.
      disp = disp_predict;
      initial_guess(tt + dt, disp);

      // The load value: the prescribed displacement, or the traction faces.
      if (dirichlet.is_disp_load())
         dirichlet.print_disp_load_by_step(disp);
      if (neumann.is_traction_load())
         neumann.print_traction_load_by_step();
      SystemTools::print_newton_header();

      // Newton iterations for R_dyn(u_{n+1}) = 0; the empty right-hand side
      // means zero.
      newton_solver.Mult(mfem::Vector(), disp);
      MFEM_VERIFY(newton_solver.GetConverged(),
                  "Newton did not converge at t = " << tt + dt << ".");

      // a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2),
      // v_{n+1} = v_n + dt ( (1 - gamma) a_n + gamma a_{n+1} ).
      acce = disp;
      acce -= disp_predict;
      acce /= beta * dt * dt;
      velo = input_velo_old;
      velo.Add(dt * (1.0 - gamma), acce_old);
      velo.Add(dt * gamma, acce);
      return newton_solver.GetNumIterations();
   }

   // Required by MFEM: overrides mfem::Operator::Mult, which NewtonSolver
   // calls at every iteration for the residual.
   // R_dyn(u_{n+1}), zero on the constrained dofs.
   void Mult(const mfem::Vector &disp, mfem::Vector &residual) const override
   {
      assemble_residual(disp, residual);
      global_assembly->set_essential_bdr(residual);
   }

   // Required by MFEM: overrides mfem::Operator::GetGradient, which
   // NewtonSolver calls at every iteration for the tangent.
   // K_eff(u_{n+1}), the identity on the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &disp) const override
   {
      tangent = assemble_tangent(disp);
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

protected:
   // Sets the step from time tt to tt + input_dt: copies into members what
   // Mult and GetGradient need besides u_{n+1}, the known part of Newmark's
   // formula for u_{n+1},
   //    u_pred = u_n + dt v_n + dt^2 (1/2 - beta) a_n,
   // so that a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2), and the traction at
   // t_alpha = t_n + alpha_f dt.
   void set_step(double tt, double input_dt, const mfem::Vector &input_disp_old,
                 const mfem::Vector &input_velo_old, const mfem::Vector &input_acce_old)
   {
      dt = input_dt;
      disp_old = input_disp_old;
      acce_old = input_acce_old;

      disp_predict = disp_old;
      disp_predict.Add(dt, input_velo_old);
      disp_predict.Add(dt * dt * (0.5 - time_method->get_beta()), acce_old);

      if (global_assembly->get_neumann().is_traction_load())
         global_assembly->set_traction_load(tt + time_method->get_alpha_f() * dt);
   }

   // R_dyn(u_{n+1}) = R(u_alpha, t_alpha) + M a_alpha at every dof, with
   //    u_alpha = (1 - alpha_f) u_n + alpha_f u_{n+1},
   //    a_alpha = (1 - alpha_m) a_n + alpha_m (u_{n+1} - u_pred) / (beta dt^2).
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      const double alpha_m = time_method->get_alpha_m();
      const double alpha_f = time_method->get_alpha_f();
      const double beta = time_method->get_beta();

      mfem::Vector disp_alpha(disp_old);
      disp_alpha *= 1.0 - alpha_f;
      disp_alpha.Add(alpha_f, disp);

      mfem::Vector acce_alpha(disp);
      acce_alpha -= disp_predict;
      acce_alpha *= alpha_m / (beta * dt * dt);
      acce_alpha.Add(1.0 - alpha_m, acce_old);

      global_assembly->assemble_residual(disp_alpha, residual);
      mass->AddMult(acce_alpha, residual);
   }

   // K_eff(u_{n+1}) = dR_dyn/du_{n+1} = alpha_m / (beta dt^2) M + alpha_f K(u_alpha)
   // at every dof, in a new SparseMatrix, which the caller owns.
   std::unique_ptr<mfem::SparseMatrix> assemble_tangent(const mfem::Vector &disp) const
   {
      const double alpha_m = time_method->get_alpha_m();
      const double alpha_f = time_method->get_alpha_f();
      const double beta = time_method->get_beta();

      mfem::Vector disp_alpha(disp_old);
      disp_alpha *= 1.0 - alpha_f;
      disp_alpha.Add(alpha_f, disp);

      return std::unique_ptr<mfem::SparseMatrix>(
         mfem::Add(alpha_m / (beta * dt * dt), *mass,
                   alpha_f, global_assembly->assemble_tangent(disp_alpha)));
   }

private:
   // Consistent predictor, as in NonlinearSolver_Static_Disp: one linear step
   // of R_dyn = 0 from disp, so that the interior follows the prescribed
   // boundary increment g at time tt instead of only the boundary nodes
   // moving,
   //    K_eff,ff du_f = -R_dyn,f(u) - K_eff,fe g,
   // with K_eff,fe g moved to the right-hand side by set_essential_bdr.
   void initial_guess(double tt, mfem::GridFunction &disp)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();

      // Zero on the fixed faces, the prescribed values at time tt on the
      // displacement-driven ones; g is their difference from disp.
      mfem::GridFunction disp_target(disp);
      dirichlet.apply_fixed_bc(disp_target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp_target);
      mfem::Vector prescribed_increment(disp_target);
      prescribed_increment -= disp;

      mfem::Vector rhs(global_assembly->get_num_dofs());
      assemble_residual(disp, rhs);
      rhs.Neg();
      tangent = assemble_tangent(disp);
      global_assembly->set_essential_bdr(*tangent, prescribed_increment, rhs);

      mfem::Vector predicted_increment(global_assembly->get_num_dofs());
      linear_solver.SetOperator(*tangent);
      linear_solver.Mult(rhs, predicted_increment);
      disp += predicted_increment;

      // Set the prescribed values exactly, free of round-off.
      dirichlet.apply_fixed_bc(disp);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(tt, disp);
   }

   // newton_solver points to this operator, the linear solver and the
   // monitor, so it is declared last and goes first.
   const std::unique_ptr<GlobalAssembly_Disp> global_assembly;   // R, K and M
   const std::unique_ptr<TimeMethod_GenAlpha> time_method;       // alpha_m, alpha_f, gamma, beta
   const std::unique_ptr<mfem::SparseMatrix> mass;               // M, assembled once
   mfem::UMFPackSolver linear_solver;                            // direct solver of the tangent
   SystemTools::NewtonMonitor newton_monitor;                    // prints the residual norms
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;          // tangent of the predictor and of Newton's method

   // The step being solved, set by set_step.
   double dt = 0.0;                                              // time step
   mfem::Vector disp_old;                                        // u_n
   mfem::Vector acce_old;                                        // a_n
   mfem::Vector disp_predict;                                    // u_pred

   mfem::NewtonSolver newton_solver;
};

#endif

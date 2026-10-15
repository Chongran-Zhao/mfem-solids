// ============================================================================
// NonlinearSolver.hpp
//
// Solves one time step of the displacement dynamics,
//    M a + F_int(u) = F_ext(t),
// with the generalized-alpha method: the equation holds at the intermediate
// states u_alpha, a_alpha and the time t_alpha, and Newmark's formulas give
// a_{n+1} and v_{n+1} from u_{n+1}, the unknown of Newton's method. It also
// computes the initial acceleration. It owns the global assembly, the mass
// matrix, the time method and the solvers; the loop over the time steps is
// left to its caller. It is also the mfem::Operator that its NewtonSolver
// solves, through Mult and GetGradient. The fields are ParGridFunctions, the
// boundary values set on them; Newton's method works on the vector of the
// dofs this rank owns, with the norms over all ranks, and MUMPS solves the
// tangent, analyzing its sparsity once.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_HPP
#define NONLINEAR_SOLVER_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "INonlinearSolver.hpp"
#include "LoadData.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha_2nd_Order.hpp"

class NonlinearSolver : public INonlinearSolver
{
public:
   // Takes the ownership of the global assembly and of the time method, and
   // assembles the mass matrix; the Newton settings come from the solver
   // section of config.yaml.
   NonlinearSolver(std::unique_ptr<GlobalAssembly_Disp> input_global_assembly,
                   std::unique_ptr<TimeMethod_GenAlpha_2nd_Order> input_time_method,
                   const YAML::Node &solver)
      : INonlinearSolver(input_global_assembly->get_num_dofs(), input_global_assembly->get_comm()),
        global_assembly(std::move(input_global_assembly)),
        time_method(std::move(input_time_method)),
        mass(global_assembly->assemble_mass()),
        linear_solver(global_assembly->get_comm())
   {
      set_direct_solver(linear_solver);
      set_newton_solver(solver, linear_solver, newton_monitor);
   }

   // The initial state at time tt: sets the boundary values of disp and
   // velo, and solves the equation of motion for the initial acceleration,
   //    M a_0 = F_ext(t_0) - F_int(u_0),
   // on the free dofs; on the constrained ones a_0 = 0, as in MixPERIGEE. A
   // wrong a_0 there only makes their acceleration alternate about its
   // value, whereas a wrong v_0 would disturb the whole solution.
   void initialize(double tt, mfem::ParGridFunction &disp, mfem::ParGridFunction &velo,
                   mfem::ParGridFunction &acce)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();

      dirichlet.apply_fixed_bc(disp);
      dirichlet.apply_fixed_bc(velo);
      if (dirichlet.is_disp_load())
      {
         dirichlet.apply_load_bc(tt, LoadData::disp_loading, disp);
         dirichlet.apply_load_bc(tt, LoadData::velo_loading, velo);
      }
      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);

      // rhs = -R(u_0), and M the identity on the constrained dofs, where rhs
      // is zero.
      const int num_dofs = global_assembly->get_num_dofs();
      mfem::Vector disp_owned(num_dofs), rhs(num_dofs), acce_owned(num_dofs);
      disp.GetTrueDofs(disp_owned);
      global_assembly->assemble_residual(disp_owned, rhs);
      rhs.Neg();
      global_assembly->set_essential_bdr(rhs);
      mfem::HypreParMatrix constrained_mass(*mass);
      global_assembly->set_essential_bdr(constrained_mass);

      // M has another sparsity than the tangents, whose analysis
      // linear_solver keeps, so it has a solver of its own.
      mfem::MUMPSSolver mass_solver(global_assembly->get_comm());
      mass_solver.SetPrintLevel(0);
      mass_solver.SetOperator(constrained_mass);
      mass_solver.Mult(rhs, acce_owned);
      acce.SetFromTrueDofs(acce_owned);
   }

   // Solves the step from time_n to time_n + input_dt: disp, velo and acce,
   // the state of time_n, become that of time_n + input_dt. Returns the
   // number of Newton iterations.
   int solve(double time_n, double input_dt, mfem::ParGridFunction &disp,
             mfem::ParGridFunction &velo, mfem::ParGridFunction &acce)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();

      // The values on the dofs this rank owns, those of time_n first.
      const int num_dofs = global_assembly->get_num_dofs();
      mfem::Vector disp_owned(num_dofs), velo_owned(num_dofs), acce_owned(num_dofs);
      disp.GetTrueDofs(disp_owned);
      velo.GetTrueDofs(velo_owned);
      acce.GetTrueDofs(acce_owned);
      set_step(time_n, input_dt, disp_owned, velo_owned, acce_owned);

      // Newton's method starts from u_pred, the displacement with
      // a_{n+1} = 0, with the prescribed values of time time_n + dt on the
      // constrained dofs, as in MixPERIGEE; the Newton increments are zero
      // there.
      disp.SetFromTrueDofs(disp_predict);
      dirichlet.apply_fixed_bc(disp);
      if (dirichlet.is_disp_load())
         dirichlet.apply_load_bc(time_n + dt, LoadData::disp_loading, disp);
      disp.GetTrueDofs(disp_owned);

      SystemTools::print_newton_header();

      // Newton iterations for R_dyn(u_{n+1}) = 0.
      const int iterations = solve_newton(time_n + dt, disp_owned);

      // a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2),
      // v_{n+1} = v_n + dt ( (1 - gamma) a_n + gamma a_{n+1} ).
      acce_owned = time_method->get_acce(disp_owned, disp_predict, dt);
      velo_owned = time_method->get_velo(velo_owned, acce_n, acce_owned, dt);

      disp.SetFromTrueDofs(disp_owned);
      velo.SetFromTrueDofs(velo_owned);
      acce.SetFromTrueDofs(acce_owned);
      return iterations;
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

private:
   // Sets the step from time_n to time_n + input_dt: copies into members what
   // Mult and GetGradient need besides u_{n+1}, the known part of Newmark's
   // formula for u_{n+1},
   //    u_pred = u_n + dt v_n + dt^2 (1/2 - beta) a_n,
   // so that a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2), and the traction at
   // t_alpha = t_n + alpha_f dt.
   void set_step(double time_n, double input_dt, const mfem::Vector &input_disp_n,
                 const mfem::Vector &input_velo_n, const mfem::Vector &input_acce_n)
   {
      dt = input_dt;
      disp_n = input_disp_n;
      acce_n = input_acce_n;

      disp_predict = time_method->get_disp_predict(disp_n, input_velo_n, acce_n, dt);

      if (global_assembly->get_neumann().is_traction_load())
         global_assembly->set_traction_load(time_method->get_time_alpha(time_n, dt));
   }

   // R_dyn(u_{n+1}) = R(u_alpha, t_alpha) + M a_alpha at every dof this rank
   // owns, with
   //    u_alpha = (1 - alpha_f) u_n + alpha_f u_{n+1},
   //    a_alpha = (1 - alpha_m) a_n + alpha_m (u_{n+1} - u_pred) / (beta dt^2).
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      const mfem::Vector disp_alpha = time_method->get_state_alpha(disp_n, disp);
      const mfem::Vector acce_alpha = time_method->get_acce_alpha(disp, disp_predict, acce_n, dt);

      global_assembly->assemble_residual(disp_alpha, residual);
      mass->AddMult(acce_alpha, residual);
   }

   // K_eff(u_{n+1}) = dR_dyn/du_{n+1} = alpha_m / (beta dt^2) M + alpha_f K(u_alpha)
   // on the dofs this rank owns, in a new HypreParMatrix, which the caller
   // owns.
   std::unique_ptr<mfem::HypreParMatrix> assemble_tangent(const mfem::Vector &disp) const
   {
      const double alpha_f = time_method->get_alpha_f();
      const mfem::Vector disp_alpha = time_method->get_state_alpha(disp_n, disp);

      return std::unique_ptr<mfem::HypreParMatrix>(
         mfem::Add(time_method->get_acce_alpha_factor(dt), *mass,
                   alpha_f, global_assembly->assemble_tangent(disp_alpha)));
   }

   const std::unique_ptr<GlobalAssembly_Disp> global_assembly;   // R, K and M
   const std::unique_ptr<TimeMethod_GenAlpha_2nd_Order> time_method; // alpha_m, alpha_f, gamma, beta
   const std::unique_ptr<mfem::HypreParMatrix> mass;             // M, assembled once
   mfem::MUMPSSolver linear_solver;                              // parallel direct solver of the tangent
   SystemTools::NewtonMonitor newton_monitor;                    // prints the residual norms
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent;        // tangent of Newton's method

   // The step being solved, set by set_step.
   double dt = 0.0;                                              // time step
   mfem::Vector disp_n;                                          // u_n on the dofs this rank owns
   mfem::Vector acce_n;                                          // a_n on the dofs this rank owns
   mfem::Vector disp_predict;                                    // u_pred on the dofs this rank owns
};

#endif

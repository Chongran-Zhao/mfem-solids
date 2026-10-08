// ============================================================================
// NonlinearSolver_Dynamic_Mixed.hpp
//
// Solves one time step of the mixed displacement-pressure dynamics,
//    M a + F_int(u,p) = F_ext(t),   J(u) = J(p) weakly,
// with the generalized-alpha method of second order: both equations hold at
// the intermediate states u_alpha, p_alpha, a_alpha and the time t_alpha,
// and Newmark's formulas give a_{n+1} and v_{n+1} from u_{n+1}; u_{n+1} and
// p_{n+1} are the unknowns of Newton's method. The pressure has no inertia.
// It also computes the initial acceleration. It owns the global assembly, the
// mass matrix, the time method and the solvers; the loop over the time steps
// is left to its caller. It is also the mfem::Operator that its NewtonSolver
// solves, through Mult and GetGradient. The fields are ParGridFunctions, the
// boundary values set on the displacement and the velocity; Newton's method
// works on the vector of the dofs this rank owns, with the norms over all
// ranks, and MUMPS solves the tangent, analyzing its sparsity once.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_DYNAMIC_MIXED_HPP
#define NONLINEAR_SOLVER_DYNAMIC_MIXED_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Mixed.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha.hpp"

class NonlinearSolver_Dynamic_Mixed : public mfem::Operator
{
public:
   // Takes the ownership of the global assembly and of the time method, and
   // assembles the mass matrix; the Newton settings come from the solver
   // section of config.yaml.
   NonlinearSolver_Dynamic_Mixed(std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
                                 std::unique_ptr<TimeMethod_GenAlpha> input_time_method,
                                 const YAML::Node &solver)
      : mfem::Operator(input_global_assembly->get_num_dofs()),
        global_assembly(std::move(input_global_assembly)),
        time_method(std::move(input_time_method)),
        mass(global_assembly->assemble_mass()),
        linear_solver(global_assembly->get_comm()),
        newton_monitor(global_assembly->get_offsets(), global_assembly->get_comm()),
        sol_n(global_assembly->get_offsets()),
        newton_solver(global_assembly->get_comm())
   {
      linear_solver.SetPrintLevel(0);
      // The tangents keep the sparsity of the first one, so MUMPS orders and
      // analyzes it once and only factorizes the later ones.
      linear_solver.SetReorderingReuse(true);
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
   //    M a_0 = F_ext(t_0) - F_int(u_0,p_0),
   // on the free dofs; on the constrained ones a_0 = 0, as in MixPERIGEE.
   // pres is p_0, given with u_0.
   void initialize(double tt, mfem::ParGridFunction &disp, mfem::ParGridFunction &velo,
                   mfem::ParGridFunction &acce, const mfem::ParGridFunction &pres)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const NeumannBoundary &neumann = global_assembly->get_neumann();

      dirichlet.apply_fixed_bc(disp);
      dirichlet.apply_fixed_bc(velo);
      if (dirichlet.is_disp_load())
      {
         dirichlet.apply_disp_load_bc(tt, disp);
         dirichlet.apply_velo_load_bc(tt, velo);
      }
      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);

      // rhs = -R_u(u_0,p_0), the displacement block of the residual, and M
      // the identity on the constrained dofs, where rhs is zero.
      mfem::BlockVector sol(global_assembly->get_offsets());
      disp.GetTrueDofs(sol.GetBlock(0));
      pres.GetTrueDofs(sol.GetBlock(1));
      mfem::BlockVector residual(global_assembly->get_offsets());
      global_assembly->assemble_residual(sol, residual);
      mfem::Vector rhs(residual.GetBlock(0));
      rhs.Neg();
      global_assembly->set_essential_bdr(rhs);
      mfem::HypreParMatrix constrained_mass(*mass);
      global_assembly->set_essential_bdr(constrained_mass);

      // M has another sparsity than the tangents, whose analysis
      // linear_solver keeps, so it has a solver of its own.
      mfem::MUMPSSolver mass_solver(global_assembly->get_comm());
      mass_solver.SetPrintLevel(0);
      mass_solver.SetOperator(constrained_mass);
      mfem::Vector acce_owned(rhs.Size());
      mass_solver.Mult(rhs, acce_owned);
      acce.SetFromTrueDofs(acce_owned);
   }

   // Solves the step from time_n to time_n + input_dt: disp, velo, acce and
   // pres, the state of time_n, become that of time_n + input_dt. Returns
   // the number of Newton iterations.
   int solve(double time_n, double input_dt, mfem::ParGridFunction &disp,
             mfem::ParGridFunction &velo, mfem::ParGridFunction &acce,
             mfem::ParGridFunction &pres)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();
      const double gamma = time_method->get_gamma();
      const double beta = time_method->get_beta();

      // The values on the dofs this rank owns, those of time_n first;
      // sol = [u; p] holds both fields.
      mfem::BlockVector sol(global_assembly->get_offsets());
      mfem::Vector velo_owned(sol.GetBlock(0).Size()), acce_owned(sol.GetBlock(0).Size());
      disp.GetTrueDofs(sol.GetBlock(0));
      pres.GetTrueDofs(sol.GetBlock(1));
      velo.GetTrueDofs(velo_owned);
      acce.GetTrueDofs(acce_owned);
      set_step(time_n, input_dt, sol, velo_owned, acce_owned);

      // Newton's method starts from [u_pred; p_n], with the prescribed
      // values of time time_n + dt on the constrained dofs, as in MixPERIGEE;
      // the Newton increments are zero there.
      disp.SetFromTrueDofs(disp_predict);
      dirichlet.apply_fixed_bc(disp);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(time_n + dt, disp);
      disp.GetTrueDofs(sol.GetBlock(0));

      SystemTools::print_block_newton_header();

      // Newton iterations for R_dyn(u_{n+1},p_{n+1}) = 0; the empty
      // right-hand side means zero.
      newton_solver.Mult(mfem::Vector(), sol);
      MFEM_VERIFY(newton_solver.GetConverged(),
                  "Newton did not converge at t = " << time_n + dt << ".");

      // a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2),
      // v_{n+1} = v_n + dt ( (1 - gamma) a_n + gamma a_{n+1} ).
      acce_owned = sol.GetBlock(0);
      acce_owned -= disp_predict;
      acce_owned /= beta * dt * dt;
      velo_owned.Add(dt * (1.0 - gamma), acce_n);
      velo_owned.Add(dt * gamma, acce_owned);

      disp.SetFromTrueDofs(sol.GetBlock(0));
      pres.SetFromTrueDofs(sol.GetBlock(1));
      velo.SetFromTrueDofs(velo_owned);
      acce.SetFromTrueDofs(acce_owned);
      return newton_solver.GetNumIterations();
   }

   // Required by MFEM: overrides mfem::Operator::Mult, which NewtonSolver
   // calls at every iteration for the residual.
   // R_dyn(u_{n+1},p_{n+1}), zero on the constrained dofs.
   void Mult(const mfem::Vector &sol, mfem::Vector &residual) const override
   {
      assemble_residual(sol, residual);
      global_assembly->set_essential_bdr(residual);
   }

   // Required by MFEM: overrides mfem::Operator::GetGradient, which
   // NewtonSolver calls at every iteration for the tangent.
   // K_eff(u_{n+1},p_{n+1}), the identity on the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      tangent = assemble_tangent(sol);
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

private:
   // Sets the step from time_n to time_n + input_dt: copies into members what
   // Mult and GetGradient need besides u_{n+1} and p_{n+1}, the known part of
   // Newmark's formula for u_{n+1},
   //    u_pred = u_n + dt v_n + dt^2 (1/2 - beta) a_n,
   // so that a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2), and the traction at
   // t_alpha = t_n + alpha_f dt.
   void set_step(double time_n, double input_dt, const mfem::BlockVector &input_sol_n,
                 const mfem::Vector &input_velo_n, const mfem::Vector &input_acce_n)
   {
      dt = input_dt;
      sol_n = input_sol_n;
      acce_n = input_acce_n;

      disp_predict = input_sol_n.GetBlock(0);
      disp_predict.Add(dt, input_velo_n);
      disp_predict.Add(dt * dt * (0.5 - time_method->get_beta()), acce_n);

      if (global_assembly->get_neumann().is_traction_load())
         global_assembly->set_traction_load(time_n + time_method->get_alpha_f() * dt);
   }

   // R_dyn(u_{n+1},p_{n+1}) = R(u_alpha,p_alpha,t_alpha) + [M a_alpha; 0] at
   // every dof this rank owns, with
   //    [u_alpha; p_alpha] = (1 - alpha_f) [u_n; p_n] + alpha_f [u_{n+1}; p_{n+1}],
   //    a_alpha = (1 - alpha_m) a_n + alpha_m (u_{n+1} - u_pred) / (beta dt^2).
   void assemble_residual(const mfem::Vector &sol, mfem::Vector &residual) const
   {
      const double alpha_m = time_method->get_alpha_m();
      const double alpha_f = time_method->get_alpha_f();
      const double beta = time_method->get_beta();

      mfem::Vector sol_alpha(sol_n);
      sol_alpha *= 1.0 - alpha_f;
      sol_alpha.Add(alpha_f, sol);

      // From u_{n+1}, the displacement block of sol, copied.
      mfem::Vector acce_alpha(mass->Height());
      acce_alpha = sol.GetData();
      acce_alpha -= disp_predict;
      acce_alpha *= alpha_m / (beta * dt * dt);
      acce_alpha.Add(1.0 - alpha_m, acce_n);

      global_assembly->assemble_residual(sol_alpha, residual);
      mfem::BlockVector residual_blocks(residual, global_assembly->get_offsets());
      mass->AddMult(acce_alpha, residual_blocks.GetBlock(0));
   }

   // K_eff(u_{n+1},p_{n+1}) = dR_dyn/d(u_{n+1},p_{n+1})
   //    = alpha_m / (beta dt^2) [M 0; 0 0] + alpha_f K(u_alpha,p_alpha)
   // on the dofs this rank owns, in a new HypreParMatrix, which the caller
   // owns: the mass is added to the displacement block, and the blocks are
   // joined with alpha_f on the other three.
   std::unique_ptr<mfem::HypreParMatrix> assemble_tangent(const mfem::Vector &sol) const
   {
      const double alpha_m = time_method->get_alpha_m();
      const double alpha_f = time_method->get_alpha_f();
      const double beta = time_method->get_beta();

      mfem::Vector sol_alpha(sol_n);
      sol_alpha *= 1.0 - alpha_f;
      sol_alpha.Add(alpha_f, sol);

      mfem::Array2D<const mfem::HypreParMatrix *> blocks =
         global_assembly->assemble_tangent_blocks(sol_alpha);
      const std::unique_ptr<mfem::HypreParMatrix> tangent_uu(
         mfem::Add(alpha_m / (beta * dt * dt), *mass, alpha_f, *blocks(0, 0)));
      blocks(0, 0) = tangent_uu.get();
      mfem::Array2D<mfem::real_t> coefficients(2, 2);
      coefficients = alpha_f;
      coefficients(0, 0) = 1.0;
      return std::unique_ptr<mfem::HypreParMatrix>(
         mfem::HypreParMatrixFromBlocks(blocks, &coefficients));
   }

   // newton_solver points to this operator, the linear solver and the
   // monitor, so it is declared last and goes first.
   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly;  // R, K and M
   const std::unique_ptr<TimeMethod_GenAlpha> time_method;       // alpha_m, alpha_f, gamma, beta
   const std::unique_ptr<mfem::HypreParMatrix> mass;             // M of the displacement, assembled once
   mfem::MUMPSSolver linear_solver;                              // parallel direct solver of the tangent
   SystemTools::BlockNewtonMonitor newton_monitor;               // prints the residual norms of u and p
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent;        // tangent of Newton's method

   // The step being solved, set by set_step.
   double dt = 0.0;                                              // time step
   mfem::BlockVector sol_n;                                      // [u_n; p_n] on the dofs this rank owns
   mfem::Vector acce_n;                                          // a_n on the dofs this rank owns
   mfem::Vector disp_predict;                                    // u_pred on the dofs this rank owns

   mfem::NewtonSolver newton_solver;
};

#endif

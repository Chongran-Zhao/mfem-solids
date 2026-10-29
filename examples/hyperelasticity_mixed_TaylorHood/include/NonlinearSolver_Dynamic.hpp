// ============================================================================
// NonlinearSolver_Dynamic.hpp
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
// ranks.
//
// The linear systems are solved iteratively. The tangent
//    K = [A  B^T; B  C],   A = c M + alpha_f K_uu,   B = alpha_f K_pu,
//    C = alpha_f K_pp = -M_p[alpha_f / kappa],
// with c = alpha_m / (beta dt^2) and M_p[w] = int w N_c N_d dV, is solved by
// GMRES, preconditioned by its block lower triangle,
//    y_u = A^-1 x_u,   y_p = -S^-1 (x_p - B y_u),   S = B A^-1 B^T - C,
// with BoomerAMG for A and, for S, the approximation of Cahouet and Chabard
// of the unsteady Stokes problem, with the compressibility added to its
// inertial part,
//    S^-1 ~ M_p[1 / (alpha_f mu)]^-1 + (L_p[1 / rho_0] / c + M_p[alpha_f / kappa])^-1,
// where L_p[w] = int w grad N_c . grad N_d dV; both pressure operators are
// of the reference configuration. For waves of number k it gives
// alpha_f mu + 1 / (k^2 / (c rho_0) + alpha_f / kappa), against the exact
// 1 / (k^2 / (c rho_0 + alpha_f mu k^2) + alpha_f / kappa): equal without
// compressibility, and close with it. The first inverse is a Chebyshev
// smoother of order 4 on the diagonal, with about 35 % fewer GMRES
// iterations than one Jacobi sweep, the second one V-cycle of BoomerAMG,
// set up again when dt changes.
// The initial acceleration solves M a_0 = r by CG, preconditioned by the
// diagonal of M.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_DYNAMIC_HPP
#define NONLINEAR_SOLVER_DYNAMIC_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Mixed.hpp"
#include "LoadData.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha_2nd_Order.hpp"

class NonlinearSolver_Dynamic : public NonlinearSolver
{
public:
   // Takes the ownership of the global assembly and of the time method, and
   // assembles the mass matrix and, on space_p, the pressure operators of the
   // preconditioner; the Newton settings come from the solver
   // section of config.yaml.
   NonlinearSolver_Dynamic(mfem::ParFiniteElementSpace &space_p,
                           std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
                           std::unique_ptr<TimeMethod_GenAlpha_2nd_Order> input_time_method,
                           const YAML::Node &solver)
      : NonlinearSolver(input_global_assembly->get_num_dofs(), input_global_assembly->get_comm()),
        global_assembly(std::move(input_global_assembly)),
        time_method(std::move(input_time_method)),
        mass(global_assembly->assemble_mass()),
        schur_inverse(*this, space_p.GetTrueVSize()),
        preconditioner(global_assembly->get_offsets()),
        linear_solver(global_assembly->get_comm()),
        newton_monitor(global_assembly->get_offsets(), global_assembly->get_comm()),
        sol_n(global_assembly->get_offsets())
   {
      // M_p[1 / (alpha_f mu)], L_p[1 / rho_0] and M_p[alpha_f / kappa]; the
      // sum of the last two is formed by set_step.
      const double alpha_f = time_method->get_alpha_f();
      pres_viscous = assemble_pres_operator(space_p, 1.0 / (alpha_f * shear_modulus), false);
      pres_laplace = assemble_pres_operator(space_p, 1.0 / density, true);
      pres_compress = assemble_pres_operator(space_p, alpha_f / bulk_modulus, false);
      viscous_diag.SetSize(pres_viscous->Height());
      pres_viscous->GetDiag(viscous_diag);
      viscous_inverse = std::make_unique<mfem::OperatorChebyshevSmoother>(
         *pres_viscous, viscous_diag, mfem::Array<int>(), 4, pres_viscous->GetComm());
      amg_p.SetPrintLevel(0);

      // A is a vector Laplacian-like operator of 3 components per node.
      amg_u.SetSystemsOptions(3, false);
      amg_u.SetPrintLevel(0);
      preconditioner.SetDiagonalBlock(1, &schur_inverse);

      set_linear_solver(linear_solver, preconditioner);
      set_newton_solver(solver, linear_solver, newton_monitor);
   }

   // The initial state at time tt: sets the boundary values of disp and
   // velo, and solves the equation of motion for the initial acceleration,
   //    M a_0 = F_ext(t_0) - F_int(u_0,p_0),
   // on the free dofs; on the constrained ones a_0 = 0, as in MixPERIGEE.
   // pres is p_0, given with u_0.
   void initialize(double tt, mfem::ParGridFunction &disp, mfem::ParGridFunction &velo,
                   mfem::ParGridFunction &acce, mfem::ParGridFunction &pres) override
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

      mfem::Vector acce_owned(rhs.Size());
      solve_mass(constrained_mass, rhs, acce_owned);
      acce.SetFromTrueDofs(acce_owned);
   }

   // Solves the step from time_n to time, of input_dt = time - time_n: disp,
   // velo, acce and pres, the state of time_n, become that of time. Returns
   // the number of Newton iterations.
   int solve(double time_n, double time, mfem::ParGridFunction &disp,
             mfem::ParGridFunction &velo, mfem::ParGridFunction &acce,
             mfem::ParGridFunction &pres) override
   {
      const double input_dt = time - time_n;
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();

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
         dirichlet.apply_load_bc(time_n + dt, LoadData::disp_loading, disp);
      disp.GetTrueDofs(sol.GetBlock(0));

      SystemTools::print_block_newton_header();

      // Newton iterations for R_dyn(u_{n+1},p_{n+1}) = 0.
      const int iterations = solve_newton(time_n + dt, sol);

      // a_{n+1} = (u_{n+1} - u_pred) / (beta dt^2),
      // v_{n+1} = v_n + dt ( (1 - gamma) a_n + gamma a_{n+1} ).
      acce_owned = time_method->get_acce(sol.GetBlock(0), disp_predict, dt);
      velo_owned = time_method->get_velo(velo_owned, acce_n, acce_owned, dt);

      disp.SetFromTrueDofs(sol.GetBlock(0));
      pres.SetFromTrueDofs(sol.GetBlock(1));
      velo.SetFromTrueDofs(velo_owned);
      acce.SetFromTrueDofs(acce_owned);
      return iterations;
   }

   bool is_dynamic() const override { return true; }

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
   // K_eff(u_{n+1},p_{n+1}), the identity on the constrained dofs; also sets
   // the blocks A and B of the preconditioner.
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      tangent = assemble_tangent(sol);
      global_assembly->set_essential_bdr(*tangent);

      // A and B with the displacement constraints, A the identity on them and
      // B without their columns; with MUMPS, none.
      if (!is_direct())
      {
         const mfem::Array<int> ess_tdof_list = global_assembly->get_dirichlet().get_ess_tdof_list();
         tangent_uu->EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
         delete tangent_pu->EliminateCols(ess_tdof_list);
         amg_u.SetOperator(*tangent_uu);
         preconditioner.SetDiagonalBlock(0, &amg_u);
         preconditioner.SetBlock(1, 0, tangent_pu.get());
      }
      return *tangent;
   }

private:
   // M_p[w] or, with is_laplace, L_p[w], with w constant.
   static std::unique_ptr<mfem::HypreParMatrix> assemble_pres_operator(
      mfem::ParFiniteElementSpace &space_p, double weight, bool is_laplace)
   {
      mfem::ConstantCoefficient coeff(weight);
      mfem::ParBilinearForm form(&space_p);
      if (is_laplace)
         form.AddDomainIntegrator(new mfem::DiffusionIntegrator(coeff));
      else
         form.AddDomainIntegrator(new mfem::MassIntegrator(coeff));
      form.Assemble();
      form.Finalize();
      return std::unique_ptr<mfem::HypreParMatrix>(form.ParallelAssemble());
   }

   // -S^-1 ~ -(M_p[1 / (alpha_f mu)]^-1
   //          + (L_p[1 / rho_0] / c + M_p[alpha_f / kappa])^-1), with one
   // Chebyshev smoother for the first and one V-cycle of BoomerAMG for the
   // second.
   class SchurInverse : public mfem::Solver
   {
   public:
      // size: the pressure dofs this rank owns.
      SchurInverse(const NonlinearSolver_Dynamic &input_owner, int size)
         : mfem::Solver(size), owner(input_owner), work(size) {}

      void SetOperator(const mfem::Operator &op) override {}

      void Mult(const mfem::Vector &xx, mfem::Vector &yy) const override
      {
         owner.viscous_inverse->Mult(xx, yy);
         owner.amg_p.Mult(xx, work);
         yy += work;
         yy.Neg();
      }

   private:
      const NonlinearSolver_Dynamic &owner;
      mutable mfem::Vector work;
   };

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

      disp_predict = time_method->get_disp_predict(input_sol_n.GetBlock(0), input_velo_n, acce_n, dt);

      if (global_assembly->get_neumann().is_traction_load())
         global_assembly->set_traction_load(time_method->get_time_alpha(time_n, dt));

      // L_p[1 / rho_0] / c + M_p[alpha_f / kappa], for the inertial part of
      // S^-1, when dt changes.
      if (dt != inertia_dt)
      {
         const double cc = time_method->get_acce_alpha_factor(dt);
         pres_inertia.reset(mfem::Add(1.0 / cc, *pres_laplace, 1.0, *pres_compress));
         amg_p.SetOperator(*pres_inertia);
         inertia_dt = dt;
      }
   }

   // R_dyn(u_{n+1},p_{n+1}) = R(u_alpha,p_alpha,t_alpha) + [M a_alpha; 0] at
   // every dof this rank owns, with
   //    [u_alpha; p_alpha] = (1 - alpha_f) [u_n; p_n] + alpha_f [u_{n+1}; p_{n+1}],
   //    a_alpha = (1 - alpha_m) a_n + alpha_m (u_{n+1} - u_pred) / (beta dt^2).
   void assemble_residual(const mfem::Vector &sol, mfem::Vector &residual) const
   {
      const mfem::Vector sol_alpha = time_method->get_state_alpha(sol_n, sol);

      // u_{n+1}, the displacement block of sol, not copied.
      const mfem::Vector disp(sol.GetData(), mass->Height());
      const mfem::Vector acce_alpha = time_method->get_acce_alpha(disp, disp_predict, acce_n, dt);

      global_assembly->assemble_residual(sol_alpha, residual);
      mfem::BlockVector residual_blocks(residual, global_assembly->get_offsets());
      mass->AddMult(acce_alpha, residual_blocks.GetBlock(0));
   }

   // K_eff(u_{n+1},p_{n+1}) = dR_dyn/d(u_{n+1},p_{n+1})
   //    = alpha_m / (beta dt^2) [M 0; 0 0] + alpha_f K(u_alpha,p_alpha)
   // on the dofs this rank owns, in a new HypreParMatrix, which the caller
   // owns: the mass is added to the displacement block, and the blocks are
   // joined with alpha_f on the other three. Keeps A = tangent_uu and
   // B = tangent_pu for the preconditioner.
   std::unique_ptr<mfem::HypreParMatrix> assemble_tangent(const mfem::Vector &sol) const
   {
      const double alpha_f = time_method->get_alpha_f();
      const mfem::Vector sol_alpha = time_method->get_state_alpha(sol_n, sol);

      mfem::Array2D<const mfem::HypreParMatrix *> blocks =
         global_assembly->assemble_tangent_blocks(sol_alpha);
      tangent_uu.reset(mfem::Add(time_method->get_acce_alpha_factor(dt), *mass, alpha_f, *blocks(0, 0)));
      tangent_pu = std::make_unique<mfem::HypreParMatrix>(*blocks(1, 0));
      *tangent_pu *= alpha_f;
      blocks(0, 0) = tangent_uu.get();
      mfem::Array2D<mfem::real_t> coefficients(2, 2);
      coefficients = alpha_f;
      coefficients(0, 0) = 1.0;
      return std::unique_ptr<mfem::HypreParMatrix>(
         mfem::HypreParMatrixFromBlocks(blocks, &coefficients));
   }

   // The preconditioner points to the AMG solvers and to schur_inverse,
   // which reads dt, the time method and the pressure operators.
   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly;  // R, K and M
   const std::unique_ptr<TimeMethod_GenAlpha_2nd_Order> time_method; // alpha_m, alpha_f, gamma, beta
   const std::unique_ptr<mfem::HypreParMatrix> mass;             // M of the displacement, assembled once
   std::unique_ptr<mfem::HypreParMatrix> pres_viscous;           // M_p[1 / (alpha_f mu)]
   std::unique_ptr<mfem::HypreParMatrix> pres_laplace;           // L_p[1 / rho_0]
   std::unique_ptr<mfem::HypreParMatrix> pres_compress;          // M_p[alpha_f / kappa]
   std::unique_ptr<mfem::HypreParMatrix> pres_inertia;           // L_p[1 / rho_0] / c + M_p[alpha_f / kappa]
   double inertia_dt = 0.0;                                      // the dt of pres_inertia
   mfem::Vector viscous_diag;                                    // diagonal of pres_viscous
   std::unique_ptr<mfem::OperatorChebyshevSmoother> viscous_inverse; // pres_viscous^-1, order 4
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent_uu;     // A, constrained
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent_pu;     // B, constrained columns removed
   mutable mfem::HypreBoomerAMG amg_u;                           // A^-1
   mfem::HypreBoomerAMG amg_p;                                   // pres_inertia^-1
   SchurInverse schur_inverse;                                   // -S^-1
   mutable mfem::BlockLowerTriangularPreconditioner preconditioner;
   mfem::GMRESSolver linear_solver;                              // solver of the tangent
   SystemTools::BlockNewtonMonitor newton_monitor;               // prints the residual norms of u and p
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent;        // tangent of Newton's method

   // The step being solved, set by set_step.
   double dt = 0.0;                                              // time step
   mfem::BlockVector sol_n;                                      // [u_n; p_n] on the dofs this rank owns
   mfem::Vector acce_n;                                          // a_n on the dofs this rank owns
   mfem::Vector disp_predict;                                    // u_pred on the dofs this rank owns
};

#endif

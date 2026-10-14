// ============================================================================
// NonlinearSolver.hpp
//
// Solves one time step of the rate form of the mixed displacement-pressure
// dynamics, as in PERIGEE solids,
//    du/dt = v,   M dv/dt + F_int(u,p) = F_ext(t),   J div v = J'(p) dp/dt weakly,
// with the generalized-alpha method of first order, TimeMethod_GenAlpha_1st:
// the equations hold at the intermediate states and the time t_alpha, and
// v_{n+1} and p_{n+1} are the unknowns of Newton's method; u_{n+1} follows
// from the kinematics du_alpha = v_alpha. The pressure has no inertia. The
// mass equation, with its VMS stabilization, is that of
// LocalAssembly_Mixed_VMS, to which this solver gives, at every step, u,
// dv/dt and dp/dt as affine functions of v_alpha and p_alpha. It also
// computes the initial acceleration. It owns the global assembly, the mass
// matrix, the time method and the solvers, and keeps the rates du/dt and
// dp/dt between the steps; the loop over the time steps is left to its
// caller. It is also the mfem::Operator that its NewtonSolver solves, through
// Mult and GetGradient. The fields are ParGridFunctions, the boundary values
// set on the displacement and the velocity; Newton's method works on the
// vector of the dofs this rank owns, with the norms over all ranks, and MUMPS
// solves the tangent, analyzing its sparsity once.
//
// Author: Chongran Zhao
// Date: Oct. 14, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_HPP
#define NONLINEAR_SOLVER_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Mixed.hpp"
#include "INonlinearSolver.hpp"
#include "LoadData.hpp"
#include "LocalAssembly_Mixed_VMS.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha_1st.hpp"

class NonlinearSolver : public INonlinearSolver
{
public:
   // Takes the ownership of the global assembly and of the time method, and
   // assembles the mass matrix; the Newton settings come from the solver
   // section of config.yaml. local_assembly is that of the global assembly,
   // which owns it.
   NonlinearSolver(std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
                   LocalAssembly_Mixed_VMS &input_local_assembly,
                   std::unique_ptr<TimeMethod_GenAlpha_1st> input_time_method,
                   const YAML::Node &solver)
      : INonlinearSolver(input_global_assembly->get_num_dofs(), input_global_assembly->get_comm()),
        global_assembly(std::move(input_global_assembly)),
        local_assembly(input_local_assembly),
        time_method(std::move(input_time_method)),
        mass(global_assembly->assemble_mass()),
        linear_solver(global_assembly->get_comm()),
        newton_monitor(global_assembly->get_offsets(), global_assembly->get_comm()),
        sol_n(global_assembly->get_offsets())
   {
      set_direct_solver(linear_solver);
      set_newton_solver(solver, linear_solver, newton_monitor);
   }

   // The initial state at time tt: sets the boundary values of disp and
   // velo, du/dt = v and dp/dt = 0, and solves the equation of motion for the
   // initial acceleration,
   //    M dv_0/dt = F_ext(t_0) - F_int(u_0,p_0),
   // on the free dofs; on the constrained ones dv_0/dt = 0, as in MixPERIGEE.
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
         dirichlet.apply_load_bc(tt, LoadData::disp_loading, disp);
         dirichlet.apply_load_bc(tt, LoadData::velo_loading, velo);
      }
      if (neumann.is_traction_load())
         global_assembly->set_traction_load(tt);

      mfem::BlockVector sol(global_assembly->get_offsets());
      velo.GetTrueDofs(sol.GetBlock(0));
      pres.GetTrueDofs(sol.GetBlock(1));
      dot_disp_n = sol.GetBlock(0);
      dot_pres_n.SetSize(sol.GetBlock(1).Size());
      dot_pres_n = 0.0;

      // F_int(u_0,p_0) from the local assembly with u = u_0 for any v.
      mfem::ParFiniteElementSpace &space_u = *disp.ParFESpace();
      mfem::ParFiniteElementSpace &space_p = *pres.ParFESpace();
      disp_offset.SetSpace(&space_u);
      disp_offset = disp;
      dot_velo_offset.SetSpace(&space_u);
      dot_velo_offset = 0.0;
      dot_pres_offset.SetSpace(&space_p);
      dot_pres_offset = 0.0;
      local_assembly.set_rates(disp_offset, 0.0, dot_velo_offset, dot_pres_offset, 0.0);

      // rhs = -R_u(u_0,p_0), the velocity block of the residual, and M the
      // identity on the constrained dofs, where rhs is zero.
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

   // Solves the step from time_n to time_n + input_dt: disp, velo, acce
   // (dv/dt) and pres, the state of time_n, become that of time_n + input_dt.
   // Returns the number of Newton iterations.
   int solve(double time_n, double input_dt, mfem::ParGridFunction &disp,
             mfem::ParGridFunction &velo, mfem::ParGridFunction &acce,
             mfem::ParGridFunction &pres)
   {
      const DirichletBoundary &dirichlet = global_assembly->get_dirichlet();

      // The values on the dofs this rank owns, those of time_n first;
      // sol = [v; p] holds the unknowns.
      mfem::BlockVector sol(global_assembly->get_offsets());
      mfem::Vector disp_owned(sol.GetBlock(0).Size()), acce_owned(sol.GetBlock(0).Size());
      velo.GetTrueDofs(sol.GetBlock(0));
      pres.GetTrueDofs(sol.GetBlock(1));
      disp.GetTrueDofs(disp_owned);
      acce.GetTrueDofs(acce_owned);
      set_step(time_n, input_dt, sol, disp_owned, acce_owned, *disp.ParFESpace(),
               *pres.ParFESpace());

      // Newton's method starts from [v_n; p_n], with the prescribed velocity
      // of time time_n + dt on the constrained dofs, as in PERIGEE; the
      // Newton increments are zero there.
      dirichlet.apply_fixed_bc(velo);
      if (dirichlet.is_disp_load())
         dirichlet.apply_load_bc(time_n + dt, LoadData::velo_loading, velo);
      velo.GetTrueDofs(sol.GetBlock(0));

      SystemTools::print_block_newton_header();

      // Newton iterations for R_dyn(v_{n+1},p_{n+1}) = 0.
      const int iterations = solve_newton(time_n + dt, sol);

      // The rates of time_n + dt, and u_{n+1} from the kinematics.
      const mfem::Vector velo_alpha = time_method->get_state_alpha(sol_n.GetBlock(0),
                                                                   sol.GetBlock(0));
      acce_owned = time_method->get_dot(sol_n.GetBlock(0), dot_velo_n, sol.GetBlock(0), dt);
      const mfem::Vector dot_pres = time_method->get_dot(sol_n.GetBlock(1), dot_pres_n,
                                                         sol.GetBlock(1), dt);
      disp_owned = time_method->get_disp(disp_n, dot_disp_n, velo_alpha, dt);
      dot_disp_n = time_method->get_dot_disp(dot_disp_n, velo_alpha);
      dot_pres_n = dot_pres;

      disp.SetFromTrueDofs(disp_owned);
      if (dirichlet.is_disp_load())
         dirichlet.apply_load_bc(time_n + dt, LoadData::disp_loading, disp);
      velo.SetFromTrueDofs(sol.GetBlock(0));
      pres.SetFromTrueDofs(sol.GetBlock(1));
      acce.SetFromTrueDofs(acce_owned);
      return iterations;
   }

   // Required by MFEM: overrides mfem::Operator::Mult, which NewtonSolver
   // calls at every iteration for the residual.
   // R_dyn(v_{n+1},p_{n+1}), zero on the constrained dofs.
   void Mult(const mfem::Vector &sol, mfem::Vector &residual) const override
   {
      assemble_residual(sol, residual);
      global_assembly->set_essential_bdr(residual);
   }

   // Required by MFEM: overrides mfem::Operator::GetGradient, which
   // NewtonSolver calls at every iteration for the tangent.
   // K_eff(v_{n+1},p_{n+1}), the identity on the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      tangent = assemble_tangent(sol);
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

private:
   // Sets the step from time_n to time_n + input_dt: copies into members the
   // state of time_n, the traction at t_alpha = t_n + alpha_f dt, and gives
   // the local assembly u, dv/dt and dp/dt as affine functions of v_alpha
   // and p_alpha, their values at v_alpha = 0 and p_alpha = 0 and their
   // factors:
   //    u_alpha  = (1 - alpha_f) u_n + alpha_f u_{n+1}(v_alpha),
   //    dv_alpha = (1 - alpha_m) dv_n + alpha_m dv_{n+1}(v_{n+1}(v_alpha)),
   //    dp_alpha likewise. space_u and space_p are the spaces of the
   // velocity and the pressure.
   void set_step(double time_n, double input_dt, const mfem::BlockVector &input_sol_n,
                 const mfem::Vector &input_disp_n, const mfem::Vector &input_dot_velo_n,
                 mfem::ParFiniteElementSpace &space_u, mfem::ParFiniteElementSpace &space_p)
   {
      dt = input_dt;
      sol_n = input_sol_n;
      disp_n = input_disp_n;
      dot_velo_n = input_dot_velo_n;

      if (global_assembly->get_neumann().is_traction_load())
         global_assembly->set_traction_load(time_method->get_time_alpha(time_n, dt));

      mfem::Vector zero_u(disp_n.Size()), zero_p(dot_pres_n.Size());
      zero_u = 0.0;
      zero_p = 0.0;
      const mfem::Vector &velo_n = sol_n.GetBlock(0);
      const mfem::Vector &pres_n = sol_n.GetBlock(1);

      disp_offset.SetSpace(&space_u);
      disp_offset.SetFromTrueDofs(time_method->get_state_alpha(
         disp_n, time_method->get_disp(disp_n, dot_disp_n, zero_u, dt)));
      dot_velo_offset.SetSpace(&space_u);
      dot_velo_offset.SetFromTrueDofs(time_method->get_dot_alpha(
         velo_n, dot_velo_n, time_method->get_state(velo_n, zero_u), dt));
      dot_pres_offset.SetSpace(&space_p);
      dot_pres_offset.SetFromTrueDofs(time_method->get_dot_alpha(
         pres_n, dot_pres_n, time_method->get_state(pres_n, zero_p), dt));
      local_assembly.set_rates(disp_offset, time_method->get_disp_alpha_factor(dt),
                               dot_velo_offset, dot_pres_offset,
                               time_method->get_dot_alpha_factor(dt));
   }

   // R_dyn(v_{n+1},p_{n+1}) = R(v_alpha,p_alpha,t_alpha) + [M dv_alpha; 0] at
   // every dof this rank owns, with
   //    [v_alpha; p_alpha] = (1 - alpha_f) [v_n; p_n] + alpha_f [v_{n+1}; p_{n+1}],
   //    dv_alpha = (1 - alpha_m) dv_n + alpha_m dv_{n+1}.
   void assemble_residual(const mfem::Vector &sol, mfem::Vector &residual) const
   {
      const mfem::Vector sol_alpha = time_method->get_state_alpha(sol_n, sol);

      // v_{n+1}, the velocity block of sol, not copied.
      const mfem::Vector velo(sol.GetData(), mass->Height());
      const mfem::Vector dot_velo_alpha =
         time_method->get_dot_alpha(sol_n.GetBlock(0), dot_velo_n, velo, dt);

      global_assembly->assemble_residual(sol_alpha, residual);
      mfem::BlockVector residual_blocks(residual, global_assembly->get_offsets());
      mass->AddMult(dot_velo_alpha, residual_blocks.GetBlock(0));
   }

   // K_eff(v_{n+1},p_{n+1}) = dR_dyn/d(v_{n+1},p_{n+1})
   //    = alpha_m / (gamma dt) [M 0; 0 0] + alpha_f K(v_alpha,p_alpha)
   // on the dofs this rank owns, in a new HypreParMatrix, which the caller
   // owns; K is the tangent of the local assembly with respect to v_alpha and
   // p_alpha. The mass is added to the velocity block, and the blocks are
   // joined with alpha_f on the other three.
   std::unique_ptr<mfem::HypreParMatrix> assemble_tangent(const mfem::Vector &sol) const
   {
      const double alpha_f = time_method->get_alpha_f();
      const mfem::Vector sol_alpha = time_method->get_state_alpha(sol_n, sol);

      mfem::Array2D<const mfem::HypreParMatrix *> blocks =
         global_assembly->assemble_tangent_blocks(sol_alpha);
      const std::unique_ptr<mfem::HypreParMatrix> tangent_uu(
         mfem::Add(alpha_f * time_method->get_dot_alpha_factor(dt), *mass, alpha_f, *blocks(0, 0)));
      blocks(0, 0) = tangent_uu.get();
      mfem::Array2D<mfem::real_t> coefficients(2, 2);
      coefficients = alpha_f;
      coefficients(0, 0) = 1.0;
      return std::unique_ptr<mfem::HypreParMatrix>(
         mfem::HypreParMatrixFromBlocks(blocks, &coefficients));
   }

   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly;   // R, K and M
   LocalAssembly_Mixed_VMS &local_assembly;                       // that of global_assembly, for set_rates
   const std::unique_ptr<TimeMethod_GenAlpha_1st> time_method;    // alpha_m, alpha_f, gamma
   const std::unique_ptr<mfem::HypreParMatrix> mass;              // M of the velocity, assembled once
   mfem::MUMPSSolver linear_solver;                               // parallel direct solver of the tangent
   SystemTools::BlockNewtonMonitor newton_monitor;                // prints the residual norms of v and p
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent;         // tangent of Newton's method

   // The rates kept between the steps, on the dofs this rank owns.
   mfem::Vector dot_disp_n;                                       // du_n/dt
   mfem::Vector dot_pres_n;                                       // dp_n/dt

   // The step being solved, set by set_step.
   double dt = 0.0;                                               // time step
   mfem::BlockVector sol_n;                                       // [v_n; p_n] on the dofs this rank owns
   mfem::Vector disp_n;                                           // u_n on the dofs this rank owns
   mfem::Vector dot_velo_n;                                       // dv_n/dt on the dofs this rank owns
   mfem::ParGridFunction disp_offset;                             // u_alpha at v_alpha = 0
   mfem::ParGridFunction dot_velo_offset;                         // dv_alpha at v_alpha = 0
   mfem::ParGridFunction dot_pres_offset;                         // dp_alpha at p_alpha = 0
};

#endif

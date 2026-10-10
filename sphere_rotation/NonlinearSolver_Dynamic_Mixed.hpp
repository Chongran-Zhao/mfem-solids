// ============================================================================
// NonlinearSolver_Dynamic_Mixed.hpp
//
// include/solver/NonlinearSolver_Dynamic_Mixed.hpp for sphere_rotation, with
// two changes; driver.cpp includes it before TimeSolver_Dynamic_Mixed.hpp,
// so that its include guard keeps the other one out.
//
// 1. The pressure is fixed at the center of the ball. The material is fully
//    incompressible and the displacement is prescribed on the whole surface,
//    so the pressure is defined only up to a constant: a constant pressure
//    p_0 does no work on any admissible displacement,
//    int p_0 div du dV = p_0 int du . n dA = 0, and the tangent is singular.
//    The pressure dof at the vertex nearest to the origin keeps its initial
//    value 0, as a constrained dof; the pressure elsewhere is relative to the
//    center, and the motion is not affected.
// 2. The linear systems are solved iteratively, which, unlike MUMPS, gets
//    faster with more ranks. The tangent
//       K = [A  B^T; B  0],   A = c M + alpha_f K_uu,   B = alpha_f K_pu,
//    with c = alpha_m / (beta dt^2), is solved by GMRES, preconditioned by
//    its block lower triangle,
//       y_u = A^-1 x_u,   y_p = -S^-1 (x_p - B y_u),   S = B A^-1 B^T,
//    with BoomerAMG for A and, for S, the approximation of Cahouet and
//    Chabard of the unsteady Stokes problem,
//       S^-1 ~ alpha_f mu M_p^-1 + c rho_0 L_p^-1,
//    where M_p and L_p are the mass and the Laplacian of the pressure space,
//    in the reference configuration; mu and rho_0 come from
//    MaterialModelData. The initial acceleration solves M a_0 = r by CG,
//    preconditioned by the diagonal of M.
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
// Author: Chongran Zhao
// Date: Oct. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_DYNAMIC_MIXED_HPP
#define NONLINEAR_SOLVER_DYNAMIC_MIXED_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Mixed.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha.hpp"

class NonlinearSolver_Dynamic_Mixed : public mfem::Operator
{
public:
   // Takes the ownership of the global assembly and of the time method, and
   // assembles the mass matrix and the pressure operators of the
   // preconditioner on space_p; the Newton and the GMRES settings come from
   // the solver section of config.yaml.
   NonlinearSolver_Dynamic_Mixed(mfem::ParFiniteElementSpace &space_p,
                                 std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
                                 std::unique_ptr<TimeMethod_GenAlpha> input_time_method,
                                 const YAML::Node &solver)
      : mfem::Operator(input_global_assembly->get_num_dofs()),
        global_assembly(std::move(input_global_assembly)),
        time_method(std::move(input_time_method)),
        mass(global_assembly->assemble_mass()),
        schur_inverse(*this, space_p.GetTrueVSize()),
        preconditioner(global_assembly->get_offsets()),
        linear_solver(global_assembly->get_comm()),
        newton_monitor(global_assembly->get_offsets(), global_assembly->get_comm()),
        sol_n(global_assembly->get_offsets()),
        newton_solver(global_assembly->get_comm())
   {
      set_fixed_pressure(space_p);

      // M_p and L_p, with the fixed pressure dof constrained in L_p, whose
      // constants would otherwise be its null space.
      mfem::ParBilinearForm pres_mass_form(&space_p), pres_laplace_form(&space_p);
      pres_mass_form.AddDomainIntegrator(new mfem::MassIntegrator);
      pres_laplace_form.AddDomainIntegrator(new mfem::DiffusionIntegrator);
      pres_mass_form.Assemble();
      pres_mass_form.Finalize();
      pres_laplace_form.Assemble();
      pres_laplace_form.Finalize();
      pres_mass.reset(pres_mass_form.ParallelAssemble());
      pres_laplace.reset(pres_laplace_form.ParallelAssemble());
      pres_laplace->EliminateBC(fixed_pres_dof, mfem::Operator::DIAG_ONE);
      pres_mass_jacobi = std::make_unique<mfem::HypreDiagScale>(*pres_mass);
      amg_p.SetOperator(*pres_laplace);
      amg_p.SetPrintLevel(0);

      // A is a vector Laplacian-like operator of 3 components per node.
      amg_u.SetSystemsOptions(3, false);
      amg_u.SetPrintLevel(0);
      preconditioner.SetDiagonalBlock(1, &schur_inverse);

      linear_solver.SetRelTol(solver["linear_rel_tol"].as<double>());
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(solver["linear_max_iter"].as<int>());
      linear_solver.SetKDim(solver["linear_max_iter"].as<int>());
      linear_solver.SetPrintLevel(mfem::IterativeSolver::PrintLevel().Warnings().Errors());
      linear_solver.SetPreconditioner(preconditioner);
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

      // M is well conditioned once scaled by its diagonal.
      mfem::HypreDiagScale mass_jacobi(constrained_mass);
      mfem::CGSolver mass_solver(global_assembly->get_comm());
      mass_solver.SetRelTol(1.0e-12);
      mass_solver.SetAbsTol(0.0);
      mass_solver.SetMaxIter(1000);
      mass_solver.SetPrintLevel(mfem::IterativeSolver::PrintLevel().Warnings().Errors());
      mass_solver.SetPreconditioner(mass_jacobi);
      mass_solver.SetOperator(constrained_mass);
      mfem::Vector acce_owned(rhs.Size());
      acce_owned = 0.0;
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
   // R_dyn(u_{n+1},p_{n+1}), zero on the constrained dofs and at the fixed
   // pressure.
   void Mult(const mfem::Vector &sol, mfem::Vector &residual) const override
   {
      assemble_residual(sol, residual);
      global_assembly->set_essential_bdr(residual);
      for (int dof : fixed_pres_dof)
         residual(num_dofs_u + dof) = 0.0;
   }

   // Required by MFEM: overrides mfem::Operator::GetGradient, which
   // NewtonSolver calls at every iteration for the tangent.
   // K_eff(u_{n+1},p_{n+1}), the identity on the constrained dofs and at the
   // fixed pressure; also sets the blocks A and B of the preconditioner.
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      tangent = assemble_tangent(sol);
      global_assembly->set_essential_bdr(*tangent);
      mfem::Array<int> fixed_dof(fixed_pres_dof);
      for (int &dof : fixed_dof)
         dof += num_dofs_u;
      tangent->EliminateBC(fixed_dof, mfem::Operator::DIAG_ONE);

      // A and B with the displacement constraints: A the identity on them,
      // B without their columns.
      const mfem::Array<int> ess_tdof_list = global_assembly->get_dirichlet().get_ess_tdof_list();
      tangent_uu->EliminateBC(ess_tdof_list, mfem::Operator::DIAG_ONE);
      delete tangent_pu->EliminateCols(ess_tdof_list);
      amg_u.SetOperator(*tangent_uu);
      preconditioner.SetDiagonalBlock(0, &amg_u);
      preconditioner.SetBlock(1, 0, tangent_pu.get());
      return *tangent;
   }

private:
   // -S^-1 ~ -(alpha_f mu M_p^-1 + c rho_0 L_p^-1), with one Jacobi sweep
   // for M_p and one V-cycle of BoomerAMG for L_p.
   class SchurInverse : public mfem::Solver
   {
   public:
      // size: the pressure dofs this rank owns.
      SchurInverse(const NonlinearSolver_Dynamic_Mixed &input_owner, int size)
         : mfem::Solver(size), owner(input_owner), work(size) {}

      void SetOperator(const mfem::Operator &op) override {}

      void Mult(const mfem::Vector &xx, mfem::Vector &yy) const override
      {
         const double alpha_m = owner.time_method->get_alpha_m();
         const double alpha_f = owner.time_method->get_alpha_f();
         const double beta = owner.time_method->get_beta();
         const double cc = alpha_m / (beta * owner.dt * owner.dt);

         owner.pres_mass_jacobi->Mult(xx, yy);
         yy *= -alpha_f * mu;
         owner.amg_p.Mult(xx, work);
         yy.Add(-cc * density, work);
      }

   private:
      const NonlinearSolver_Dynamic_Mixed &owner;
      mutable mfem::Vector work;
   };

   // The pressure vertex nearest to the origin: each rank finds the nearest
   // of the vertices whose dof it owns, MPI_MINLOC picks the nearest of all,
   // and its rank fixes that dof. In H1, the dof of vertex vv is the vv-th
   // one.
   void set_fixed_pressure(const mfem::ParFiniteElementSpace &space_p)
   {
      const mfem::ParMesh &mesh = *space_p.GetParMesh();
      struct { double distance; int rank; } nearest, global_nearest;
      nearest = {std::numeric_limits<double>::max(), space_p.GetMyRank()};
      int nearest_dof = -1;
      for (int vv = 0; vv < mesh.GetNV(); vv++)
      {
         const int dof = space_p.GetLocalTDofNumber(vv);
         if (dof < 0)
            continue;
         const double *coord = mesh.GetVertex(vv);
         const double distance = std::sqrt(coord[0] * coord[0] + coord[1] * coord[1]
                                           + coord[2] * coord[2]);
         if (distance < nearest.distance)
         {
            nearest.distance = distance;
            nearest_dof = dof;
         }
      }
      MPI_Allreduce(&nearest, &global_nearest, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                    space_p.GetComm());
      if (global_nearest.rank == space_p.GetMyRank())
         fixed_pres_dof.Append(nearest_dof);
   }

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
   // joined with alpha_f on the other three. Keeps A = tangent_uu and
   // B = tangent_pu for the preconditioner.
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
      tangent_uu.reset(mfem::Add(alpha_m / (beta * dt * dt), *mass, alpha_f, *blocks(0, 0)));
      tangent_pu = std::make_unique<mfem::HypreParMatrix>(*blocks(1, 0));
      *tangent_pu *= alpha_f;
      blocks(0, 0) = tangent_uu.get();
      mfem::Array2D<mfem::real_t> coefficients(2, 2);
      coefficients = alpha_f;
      coefficients(0, 0) = 1.0;
      return std::unique_ptr<mfem::HypreParMatrix>(
         mfem::HypreParMatrixFromBlocks(blocks, &coefficients));
   }

   // newton_solver points to this operator, the linear solver and the
   // monitor, so it is declared last and goes first; the preconditioner
   // points to the AMG solvers and to schur_inverse, which reads dt, the
   // time method and the pressure operators.
   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly;  // R, K and M
   const std::unique_ptr<TimeMethod_GenAlpha> time_method;       // alpha_m, alpha_f, gamma, beta
   const std::unique_ptr<mfem::HypreParMatrix> mass;             // M of the displacement, assembled once
   const int num_dofs_u = mass->Height();                        // displacement dofs this rank owns
   mfem::Array<int> fixed_pres_dof;                              // the fixed pressure dof, if this rank owns it
   std::unique_ptr<mfem::HypreParMatrix> pres_mass;              // M_p
   std::unique_ptr<mfem::HypreParMatrix> pres_laplace;           // L_p, fixed dof constrained
   std::unique_ptr<mfem::HypreDiagScale> pres_mass_jacobi;       // diagonal of M_p, inverted
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent_uu;     // A, constrained
   mutable std::unique_ptr<mfem::HypreParMatrix> tangent_pu;     // B, constrained columns removed
   mutable mfem::HypreBoomerAMG amg_u;                           // A^-1
   mfem::HypreBoomerAMG amg_p;                                   // L_p^-1
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

   mfem::NewtonSolver newton_solver;
};

#endif

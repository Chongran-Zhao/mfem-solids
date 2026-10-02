// Assembles and solves a dynamic step at the times requested by the time solver.
#ifndef NONLINEAR_SOLVER_DYNAMIC_DISP_HPP
#define NONLINEAR_SOLVER_DYNAMIC_DISP_HPP

#include <cmath>
#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha.hpp"

class NonlinearSolver_Dynamic_Disp : public mfem::Operator
{
public:
   NonlinearSolver_Dynamic_Disp(std::unique_ptr<GlobalAssembly_Disp> input_assembly,
                                double density, std::unique_ptr<TimeMethod_GenAlpha> input_time_method,
                                const YAML::Node &solver)
      : mfem::Operator(input_assembly->get_num_dofs()),
        global_assembly(std::move(input_assembly)),
        mass(global_assembly->assemble_mass(density)),
        time_method(std::move(input_time_method))
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

   // Enforce initial motion and solve M a_0 = F_ext(t_0) - F_int(u_0),
   // including prescribed acceleration through the full mass coupling.
   void initialize(double time, mfem::GridFunction &disp,
                   mfem::GridFunction &velo, mfem::GridFunction &acce)
   {
      check_fields(disp, velo, acce);
      const auto &dirichlet = global_assembly->get_dirichlet();
      dirichlet.apply_fixed_bc(disp);
      dirichlet.apply_fixed_bc(velo);
      acce = 0.0;
      if (dirichlet.is_disp_load())
      {
         dirichlet.apply_disp_load_bc(time, disp);
         dirichlet.apply_velo_load_bc(time, velo);
         dirichlet.apply_acce_load_bc(time, acce);
      }
      global_assembly->set_traction_load(time);
      mfem::Vector u, prescribed_acce, rhs(mass->Height()), a(mass->Height());
      disp.GetTrueDofs(u);
      acce.GetTrueDofs(prescribed_acce);
      global_assembly->assemble_residual(u, rhs);
      rhs.Neg();
      mfem::SparseMatrix constrained_mass(*mass);
      global_assembly->set_essential_bdr(constrained_mass, prescribed_acce, rhs);
      linear_solver.SetOperator(constrained_mass);
      linear_solver.Mult(rhs, a);
      acce.SetFromTrueDofs(a);
   }

   // Required by MFEM: Newton's residual at the intermediate states.
   void Mult(const mfem::Vector &disp, mfem::Vector &residual) const override
   {
      assemble_residual(disp, residual);
      global_assembly->set_essential_bdr(residual);
   }

   // R_dyn = R(disp_alpha, t_alpha_f) + M acce_alpha.
   // pre_* is the previous state; acce is acceleration.
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      MFEM_VERIFY(pre_disp && pre_acce && disp_predictor && step_dt > 0.0, "No active dynamic step.");
      const double alpha_m = time_method->get_alpha_m();
      const double alpha_f = time_method->get_alpha_f();
      const double acce_factor = 1.0 / (time_method->get_beta() * step_dt * step_dt);
      residual.SetSize(Height());
      mfem::Vector disp_alpha(*pre_disp), acce_alpha(disp);
      disp_alpha *= 1.0 - alpha_f;
      disp_alpha.Add(alpha_f, disp);
      acce_alpha -= *disp_predictor;
      acce_alpha *= alpha_m * acce_factor;
      acce_alpha.Add(1.0 - alpha_m, *pre_acce);
      global_assembly->assemble_residual(disp_alpha, residual);
      mass->AddMult(acce_alpha, residual);
   }

   std::unique_ptr<mfem::SparseMatrix> assemble_tangent(const mfem::Vector &disp) const
   {
      MFEM_VERIFY(pre_disp && step_dt > 0.0, "No active dynamic step.");
      const double alpha_m = time_method->get_alpha_m();
      const double alpha_f = time_method->get_alpha_f();
      const double acce_factor = 1.0 / (time_method->get_beta() * step_dt * step_dt);
      mfem::Vector disp_alpha(*pre_disp);
      disp_alpha *= 1.0 - alpha_f;
      disp_alpha.Add(alpha_f, disp);
      return std::unique_ptr<mfem::SparseMatrix>(mfem::Add(
         alpha_m * acce_factor, *mass,
         alpha_f, global_assembly->assemble_tangent(disp_alpha)));
   }

   // Required by MFEM: effective tangent after essential elimination.
   mfem::Operator &GetGradient(const mfem::Vector &disp) const override
   {
      tangent = assemble_tangent(disp);
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

   // Solve one physical time step with the owned generalized-alpha method.
   int solve(double time, double dt, mfem::GridFunction &disp,
             mfem::GridFunction &velo, mfem::GridFunction &acce)
   {
      MFEM_VERIFY(std::isfinite(dt) && dt > 0.0, "The time step must be positive.");
      check_fields(disp, velo, acce);
      const double end_time = time + dt;
      global_assembly->set_traction_load(time + time_method->get_alpha_f() * dt);
      mfem::Vector previous_disp, previous_velo, previous_acce;
      disp.GetTrueDofs(previous_disp);
      velo.GetTrueDofs(previous_velo);
      acce.GetTrueDofs(previous_acce);
      mfem::Vector predictor(previous_disp);
      predictor.Add(dt, previous_velo);
      predictor.Add(dt * dt * (0.5 - time_method->get_beta()), previous_acce);
      step_dt = dt;
      pre_disp = &previous_disp;
      pre_acce = &previous_acce;
      disp_predictor = &predictor;

      mfem::Vector u(previous_disp);
      u.Add(dt, previous_velo); u.Add(0.5 * dt * dt, previous_acce);
      mfem::GridFunction next_disp(disp.FESpace());
      next_disp.SetFromTrueDofs(u);
      initial_guess(end_time, next_disp);
      next_disp.GetTrueDofs(u);
      SystemTools::print_newton_header();
      newton_solver.Mult(mfem::Vector(), u);
      MFEM_VERIFY(newton_solver.GetConverged(),
                  "Dynamic Newton did not converge at t = " << end_time << ".");

      mfem::Vector a(u);
      a -= previous_disp; a.Add(-dt, previous_velo);
      a.Add(-dt * dt * (0.5 - time_method->get_beta()), previous_acce);
      a /= time_method->get_beta() * dt * dt;
      mfem::Vector v(previous_velo);
      v.Add(dt * (1.0 - time_method->get_gamma()), previous_acce);
      v.Add(dt * time_method->get_gamma(), a);
      disp.SetFromTrueDofs(u); velo.SetFromTrueDofs(v); acce.SetFromTrueDofs(a);
      pre_disp = pre_acce = disp_predictor = nullptr;
      step_dt = 0.0;
      return newton_solver.GetNumIterations();
   }

   double get_kinetic_energy(const mfem::GridFunction &velo) const
   {
      mfem::Vector v, mv(mass->Height());
      velo.GetTrueDofs(v); mass->Mult(v, mv);
      return 0.5 * (v * mv);
   }

private:
   // One effective-tangent correction from the time solver's predictor.
   // Include prescribed displacement increments through the full mass/stiffness
   // coupling, then impose their end-time values exactly before Newton.
   void initial_guess(double end_time, mfem::GridFunction &disp)
   {
      mfem::Vector u;
      disp.GetTrueDofs(u);
      mfem::GridFunction target(disp.FESpace());
      target.SetFromTrueDofs(u);
      const auto &dirichlet = global_assembly->get_dirichlet();
      dirichlet.apply_fixed_bc(target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(end_time, target);
      mfem::Vector prescribed_increment, rhs, increment(u.Size());
      target.GetTrueDofs(prescribed_increment);
      prescribed_increment -= u;
      assemble_residual(u, rhs); rhs.Neg();
      tangent = assemble_tangent(u);
      global_assembly->set_essential_bdr(*tangent, prescribed_increment, rhs);
      linear_solver.SetOperator(*tangent);
      linear_solver.Mult(rhs, increment);
      u += increment;

      target.SetFromTrueDofs(u);
      dirichlet.apply_fixed_bc(target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(end_time, target);
      target.GetTrueDofs(u);
      disp.SetFromTrueDofs(u);
   }

   void check_fields(const mfem::GridFunction &disp, const mfem::GridFunction &velo,
                     const mfem::GridFunction &acce) const
   {
      MFEM_VERIFY(disp.FESpace() == velo.FESpace() && disp.FESpace() == acce.FESpace() &&
                  disp.Size() == mass->Height(),
                  "Dynamics currently requires one conforming displacement space for all fields.");
   }

   const std::unique_ptr<GlobalAssembly_Disp> global_assembly;
   const std::unique_ptr<mfem::SparseMatrix> mass;
   const std::unique_ptr<TimeMethod_GenAlpha> time_method;
   mfem::UMFPackSolver linear_solver;
   SystemTools::NewtonMonitor newton_monitor;
   // Known vectors are local to solve(); callbacks borrow them only while it runs.
   double step_dt = 0.0;
   const mfem::Vector *pre_disp = nullptr, *pre_acce = nullptr, *disp_predictor = nullptr;
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;
   mfem::NewtonSolver newton_solver;
};

#endif

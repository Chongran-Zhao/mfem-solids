// One generalized-alpha time step. Owns assembly, mass and Newton solvers;
// the time loop and committed displacement/velocity/acceleration are external.
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

class NonlinearSolver_Dynamic_Disp
{
public:
   NonlinearSolver_Dynamic_Disp(std::unique_ptr<GlobalAssembly_Disp> input_assembly,
                                double density, const YAML::Node &solver)
      : global_assembly(std::move(input_assembly)),
        mass(global_assembly->assemble_mass(density))
   {
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

   // Local MFEM operator for a fixed time step. It borrows the previous state
   // only during solve(); its tangent includes mass before boundary elimination.
   class StepOperator : public mfem::Operator
   {
   public:
      StepOperator(GlobalAssembly_Disp &assembly, const mfem::SparseMatrix &input_mass,
                   const TimeMethod_GenAlpha &method, double input_dt,
                   const mfem::Vector &input_disp_n, const mfem::Vector &velo_n,
                   const mfem::Vector &input_acce_n)
         : mfem::Operator(input_mass.Height()), global_assembly(assembly),
           mass(input_mass), disp_n(input_disp_n), acce_n(input_acce_n),
           alpha_m(method.get_alpha_m()), alpha_f(method.get_alpha_f()),
           acce_factor(1.0 / (method.get_beta() * input_dt * input_dt)),
           disp_predictor(input_disp_n)
      {
         disp_predictor.Add(input_dt, velo_n);
         disp_predictor.Add(input_dt * input_dt * (0.5 - method.get_beta()), acce_n);
      }

      // Required by MFEM: Newton's residual at the intermediate states.
      void Mult(const mfem::Vector &disp, mfem::Vector &residual) const override
      {
         assemble_residual(disp, residual);
         global_assembly.set_essential_bdr(residual);
      }

      // Full residual, also used for a consistent predictor with nonzero
      // prescribed displacement increments and for verification.
      void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
      {
         residual.SetSize(Height());
         mfem::Vector disp_alpha(disp_n), acce_alpha(disp);
         disp_alpha *= 1.0 - alpha_f;
         disp_alpha.Add(alpha_f, disp);
         acce_alpha -= disp_predictor;
         acce_alpha *= alpha_m * acce_factor;
         acce_alpha.Add(1.0 - alpha_m, acce_n);
         global_assembly.assemble_residual(disp_alpha, residual);
         mass.AddMult(acce_alpha, residual);
      }

      std::unique_ptr<mfem::SparseMatrix> assemble_tangent(const mfem::Vector &disp) const
      {
         mfem::Vector disp_alpha(disp_n);
         disp_alpha *= 1.0 - alpha_f;
         disp_alpha.Add(alpha_f, disp);
         return std::unique_ptr<mfem::SparseMatrix>(mfem::Add(
            alpha_m * acce_factor, mass,
            alpha_f, global_assembly.assemble_tangent(disp_alpha)));
      }

      // Required by MFEM: effective tangent after essential elimination.
      mfem::Operator &GetGradient(const mfem::Vector &disp) const override
      {
         tangent = assemble_tangent(disp);
         global_assembly.set_essential_bdr(*tangent);
         return *tangent;
      }

   private:
      GlobalAssembly_Disp &global_assembly;
      const mfem::SparseMatrix &mass;
      const mfem::Vector &disp_n, &acce_n;
      const double alpha_m, alpha_f, acce_factor;
      mfem::Vector disp_predictor;
      mutable std::unique_ptr<mfem::SparseMatrix> tangent;
   };

   int solve(double time, double dt, const TimeMethod_GenAlpha &method,
             mfem::GridFunction &disp, mfem::GridFunction &velo, mfem::GridFunction &acce)
   {
      MFEM_VERIFY(std::isfinite(dt) && dt > 0.0, "The time step must be positive.");
      check_fields(disp, velo, acce);
      mfem::Vector u_n, v_n, a_n;
      disp.GetTrueDofs(u_n); velo.GetTrueDofs(v_n); acce.GetTrueDofs(a_n);
      global_assembly->set_traction_load(time + method.get_alpha_f() * dt);
      StepOperator step(*global_assembly, *mass, method, dt, u_n, v_n, a_n);

      // Constant-acceleration predictor, followed by one consistent linear
      // correction that includes coupling from the new boundary values.
      mfem::Vector u(u_n);
      u.Add(dt, v_n); u.Add(0.5 * dt * dt, a_n);
      mfem::GridFunction target(disp.FESpace());
      target.SetFromTrueDofs(u);
      const auto &dirichlet = global_assembly->get_dirichlet();
      dirichlet.apply_fixed_bc(target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(time + dt, target);
      mfem::Vector prescribed_increment, rhs, increment(u.Size());
      target.GetTrueDofs(prescribed_increment);
      prescribed_increment -= u;
      step.assemble_residual(u, rhs); rhs.Neg();
      auto tangent = step.assemble_tangent(u);
      global_assembly->set_essential_bdr(*tangent, prescribed_increment, rhs);
      linear_solver.SetOperator(*tangent);
      linear_solver.Mult(rhs, increment);
      u += increment;

      target.SetFromTrueDofs(u);
      dirichlet.apply_fixed_bc(target);
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(time + dt, target);
      target.GetTrueDofs(u);
      newton_solver.SetOperator(step);
      SystemTools::print_newton_header();
      newton_solver.Mult(mfem::Vector(), u);
      MFEM_VERIFY(newton_solver.GetConverged(),
                  "Dynamic Newton did not converge at t = " << time + dt << ".");

      mfem::Vector a(u);
      a -= u_n; a.Add(-dt, v_n);
      a.Add(-dt * dt * (0.5 - method.get_beta()), a_n);
      a /= method.get_beta() * dt * dt;
      mfem::Vector v(v_n);
      v.Add(dt * (1.0 - method.get_gamma()), a_n);
      v.Add(dt * method.get_gamma(), a);
      disp.SetFromTrueDofs(u); velo.SetFromTrueDofs(v); acce.SetFromTrueDofs(a);
      return newton_solver.GetNumIterations();
   }

   double get_kinetic_energy(const mfem::GridFunction &velo) const
   {
      mfem::Vector v, mv(mass->Height());
      velo.GetTrueDofs(v); mass->Mult(v, mv);
      return 0.5 * (v * mv);
   }

private:
   void check_fields(const mfem::GridFunction &disp, const mfem::GridFunction &velo,
                     const mfem::GridFunction &acce) const
   {
      MFEM_VERIFY(disp.FESpace() == velo.FESpace() && disp.FESpace() == acce.FESpace() &&
                  disp.Size() == mass->Height(),
                  "Dynamics currently requires one conforming displacement space for all fields.");
   }

   const std::unique_ptr<GlobalAssembly_Disp> global_assembly;
   const std::unique_ptr<mfem::SparseMatrix> mass;
   mfem::UMFPackSolver linear_solver;
   SystemTools::NewtonMonitor newton_monitor;
   mfem::NewtonSolver newton_solver;
};

#endif

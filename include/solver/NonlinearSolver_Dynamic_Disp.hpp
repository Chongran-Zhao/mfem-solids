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
   // *_old is the previous state; acce is acceleration.
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      MFEM_VERIFY(disp_old && acce_old && disp_predict && dt > 0.0, "No active dynamic step.");
      const double alpha_m = time_method->get_alpha_m();
      const double alpha_f = time_method->get_alpha_f();
      residual.SetSize(Height());
      mfem::Vector disp_alpha(*disp_old), acce_alpha(disp);
      disp_alpha *= 1.0 - alpha_f;
      disp_alpha.Add(alpha_f, disp);
      acce_alpha -= *disp_predict;
      acce_alpha *= alpha_m / (time_method->get_beta() * dt * dt);
      acce_alpha.Add(1.0 - alpha_m, *acce_old);
      global_assembly->assemble_residual(disp_alpha, residual);
      mass->AddMult(acce_alpha, residual);
   }

   std::unique_ptr<mfem::SparseMatrix> assemble_tangent(const mfem::Vector &disp) const
   {
      MFEM_VERIFY(disp_old && dt > 0.0, "No active dynamic step.");
      const double alpha_m = time_method->get_alpha_m();
      const double alpha_f = time_method->get_alpha_f();
      mfem::Vector disp_alpha(*disp_old);
      disp_alpha *= 1.0 - alpha_f;
      disp_alpha.Add(alpha_f, disp);
      return std::unique_ptr<mfem::SparseMatrix>(mfem::Add(
         alpha_m / (time_method->get_beta() * dt * dt), *mass,
         alpha_f, global_assembly->assemble_tangent(disp_alpha)));
   }

   // Required by MFEM: effective tangent after essential elimination.
   mfem::Operator &GetGradient(const mfem::Vector &disp) const override
   {
      tangent = assemble_tangent(disp);
      global_assembly->set_essential_bdr(*tangent);
      return *tangent;
   }

   // Read the state at time and solve into separate fields at time + dt.
   int solve(double time, double input_dt,
             const mfem::GridFunction &disp_old, const mfem::GridFunction &velo_old,
             const mfem::GridFunction &acce_old, mfem::GridFunction &disp_new,
             mfem::GridFunction &velo_new, mfem::GridFunction &acce_new)
   {
      MFEM_VERIFY(std::isfinite(input_dt) && input_dt > 0.0, "The time step must be positive.");
      check_fields(disp_old, velo_old, acce_old);
      check_fields(disp_new, velo_new, acce_new);
      MFEM_VERIFY(disp_old.FESpace() == disp_new.FESpace() &&
                  &disp_old != &disp_new && &velo_old != &velo_new && &acce_old != &acce_new,
                  "Old and new states must be separate fields on the same space.");
      dt = input_dt;
      const double end_time = time + dt;
      global_assembly->set_traction_load(time + time_method->get_alpha_f() * dt);
      mfem::Vector disp_old_true, velo_old_true, acce_old_true;
      disp_old.GetTrueDofs(disp_old_true);
      velo_old.GetTrueDofs(velo_old_true);
      acce_old.GetTrueDofs(acce_old_true);
      mfem::Vector predictor(disp_old_true);
      predictor.Add(dt, velo_old_true);
      predictor.Add(dt * dt * (0.5 - time_method->get_beta()), acce_old_true);
      this->disp_old = &disp_old_true;
      this->acce_old = &acce_old_true;
      disp_predict = &predictor;

      mfem::Vector u(disp_old_true);
      u.Add(dt, velo_old_true); u.Add(0.5 * dt * dt, acce_old_true);
      mfem::GridFunction next_disp(disp_new.FESpace());
      next_disp.SetFromTrueDofs(u);
      initial_guess(end_time, next_disp);
      next_disp.GetTrueDofs(u);
      SystemTools::print_newton_header();
      newton_solver.Mult(mfem::Vector(), u);
      MFEM_VERIFY(newton_solver.GetConverged(),
                  "Dynamic Newton did not converge at t = " << end_time << ".");

      mfem::Vector a(u);
      a -= disp_old_true; a.Add(-dt, velo_old_true);
      a.Add(-dt * dt * (0.5 - time_method->get_beta()), acce_old_true);
      a /= time_method->get_beta() * dt * dt;
      mfem::Vector v(velo_old_true);
      v.Add(dt * (1.0 - time_method->get_gamma()), acce_old_true);
      v.Add(dt * time_method->get_gamma(), a);
      disp_new.SetFromTrueDofs(u); velo_new.SetFromTrueDofs(v); acce_new.SetFromTrueDofs(a);
      this->disp_old = this->acce_old = disp_predict = nullptr;
      dt = 0.0;
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
   double dt = 0.0;
   const mfem::Vector *disp_old = nullptr, *acce_old = nullptr, *disp_predict = nullptr;
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;
   mfem::NewtonSolver newton_solver;
};

#endif

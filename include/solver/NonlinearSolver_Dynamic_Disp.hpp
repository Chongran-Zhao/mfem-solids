// Assembles and solves a dynamic step at the times requested by the time solver.
#ifndef NONLINEAR_SOLVER_DYNAMIC_DISP_HPP
#define NONLINEAR_SOLVER_DYNAMIC_DISP_HPP

#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "SystemTools.hpp"

class NonlinearSolver_Dynamic_Disp : public mfem::Operator
{
public:
   NonlinearSolver_Dynamic_Disp(std::unique_ptr<GlobalAssembly_Disp> input_assembly,
                                double density, const YAML::Node &solver)
      : mfem::Operator(input_assembly->get_num_dofs()),
        global_assembly(std::move(input_assembly)),
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

   // Full residual, also used for a consistent predictor with nonzero
   // prescribed displacement increments and for verification.
   void assemble_residual(const mfem::Vector &disp, mfem::Vector &residual) const
   {
      MFEM_VERIFY(disp_n && acce_n && disp_predictor, "No active dynamic step.");
      residual.SetSize(Height());
      mfem::Vector disp_alpha(*disp_n), acce_alpha(disp);
      disp_alpha *= 1.0 - alpha_f;
      disp_alpha.Add(alpha_f, disp);
      acce_alpha -= *disp_predictor;
      acce_alpha *= alpha_m * acce_factor;
      acce_alpha.Add(1.0 - alpha_m, *acce_n);
      global_assembly->assemble_residual(disp_alpha, residual);
      mass->AddMult(acce_alpha, residual);
   }

   std::unique_ptr<mfem::SparseMatrix> assemble_tangent(const mfem::Vector &disp) const
   {
      MFEM_VERIFY(disp_n, "No active dynamic step.");
      mfem::Vector disp_alpha(*disp_n);
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

   // Load and assemble at stage_time; impose the boundary at end_time.
   // Time integration supplies scalar weights and known vectors, not an operator.
   int solve(double stage_time, double end_time,
             double input_alpha_m, double input_alpha_f, double input_acce_factor,
             const mfem::Vector &input_disp_n, const mfem::Vector &input_acce_n,
             const mfem::Vector &input_predictor, mfem::GridFunction &disp)
   {
      MFEM_VERIFY(disp.Size() == mass->Height(), "Dynamics requires a conforming space.");
      global_assembly->set_traction_load(stage_time);
      alpha_m = input_alpha_m;
      alpha_f = input_alpha_f;
      acce_factor = input_acce_factor;
      disp_n = &input_disp_n;
      acce_n = &input_acce_n;
      disp_predictor = &input_predictor;
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
      SystemTools::print_newton_header();
      newton_solver.Mult(mfem::Vector(), u);
      MFEM_VERIFY(newton_solver.GetConverged(),
                  "Dynamic Newton did not converge at t = " << end_time << ".");

      disp.SetFromTrueDofs(u);
      disp_n = acce_n = disp_predictor = nullptr;
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
   // Known vectors are borrowed only for the active solve; the unknown stays external.
   double alpha_m = 0.0, alpha_f = 0.0, acce_factor = 0.0;
   const mfem::Vector *disp_n = nullptr, *acce_n = nullptr, *disp_predictor = nullptr;
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;
   mfem::NewtonSolver newton_solver;
};

#endif

// ============================================================================
// Operator_Dynamic_Displacement.hpp
//
// The nonlinear problem of one time step of the generalized-alpha method
// (Chung & Hulbert, 1993) in the displacement form, for NewtonSolver. With
// the state d_n, v_n, a_n at t_n, the unknown is d_{n+1}, and
//    a_{n+1} = (d_{n+1} - d_n - dt v_n) / (beta dt^2) - (1 / (2 beta) - 1) a_n,
//    v_{n+1} = v_n + dt ((1 - gamma) a_n + gamma a_{n+1}),
// the Newmark update. The balance is taken at two intermediate points,
//    R(d_{n+1}) = M a_{n+1-alpha_m} + R_int(d_{n+1-alpha_f})
//                 - F_ext(t_{n+1-alpha_f}) = 0,
// with x_{n+1-alpha} = (1 - alpha) x_{n+1} + alpha x_n, and the tangent is
//    K_eff = (1 - alpha_m) / (beta dt^2) M + (1 - alpha_f) K(d_{n+1-alpha_f}).
//
// Author: Chongran Zhao
// Date: Sep. 30, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef OPERATOR_DYNAMIC_DISPLACEMENT_HPP
#define OPERATOR_DYNAMIC_DISPLACEMENT_HPP

#include <memory>
#include <mfem.hpp>

class Operator_Dynamic_Displacement : public mfem::Operator
{
public:
   // internal_force_form gives R_int and K and has no essential dofs; mass is
   // the assembled M; the rows and columns of ess_tdof_list are eliminated.
   // The parameters follow from rho_inf, the spectral radius at infinite
   // frequency, in [0, 1].
   Operator_Dynamic_Displacement(const mfem::NonlinearForm &input_internal_force_form,
                                 const mfem::SparseMatrix &input_mass,
                                 const mfem::Array<int> &input_ess_tdof_list,
                                 double rho_inf, double input_dt)
      : mfem::Operator(input_mass.Height()),
        internal_force_form(input_internal_force_form),
        mass(input_mass),
        ess_tdof_list(input_ess_tdof_list),
        dt(input_dt),
        alpha_m((2.0 * rho_inf - 1.0) / (rho_inf + 1.0)),
        alpha_f(rho_inf / (rho_inf + 1.0)),
        gamma(0.5 - alpha_m + alpha_f),
        beta(0.25 * (1.0 - alpha_m + alpha_f) * (1.0 - alpha_m + alpha_f)),
        disp_old(height), velo_old(height), acce_old(height), external_force(height),
        disp_mid(height), acce_new(height), acce_mid(height), internal_force(height)
   {
      MFEM_VERIFY(rho_inf >= 0.0 && rho_inf <= 1.0, "rho_inf must be in [0, 1].");
   }

   // Start a step from the state at t_n, with F_ext(t_{n+1-alpha_f}).
   void set_state(const mfem::Vector &disp, const mfem::Vector &velo,
                  const mfem::Vector &acce, const mfem::Vector &input_external_force)
   {
      disp_old = disp;
      velo_old = velo;
      acce_old = acce;
      external_force = input_external_force;
   }

   // t_{n+1-alpha_f}, where the external force is taken.
   double get_mid_time(double time_old) const
   {
      return time_old + (1.0 - alpha_f) * dt;
   }

   // Initial guess of Newton's method, d_n + dt v_n + dt^2 / 2 a_n.
   void predict(mfem::Vector &disp_new) const
   {
      disp_new = disp_old;
      disp_new.Add(dt, velo_old);
      disp_new.Add(0.5 * dt * dt, acce_old);
   }

   // a_{n+1} from d_{n+1}, by the Newmark update of d.
   void get_acceleration(const mfem::Vector &disp_new, mfem::Vector &acce) const
   {
      acce = disp_new;
      acce -= disp_old;
      acce.Add(-dt, velo_old);
      acce *= 1.0 / (beta * dt * dt);
      acce.Add(-(0.5 / beta - 1.0), acce_old);
   }

   // v_{n+1} from a_{n+1}, by the Newmark update of v.
   void get_velocity(const mfem::Vector &acce, mfem::Vector &velo) const
   {
      velo = velo_old;
      velo.Add(dt * (1.0 - gamma), acce_old);
      velo.Add(dt * gamma, acce);
   }

   // R(d_{n+1}), zero on the constrained dofs.
   void Mult(const mfem::Vector &disp_new, mfem::Vector &residual) const override
   {
      get_acceleration(disp_new, acce_new);
      add(1.0 - alpha_m, acce_new, alpha_m, acce_old, acce_mid);
      add(1.0 - alpha_f, disp_new, alpha_f, disp_old, disp_mid);

      mass.Mult(acce_mid, residual);
      internal_force_form.Mult(disp_mid, internal_force);
      residual += internal_force;
      residual -= external_force;

      for (int dof : ess_tdof_list)
         residual(dof) = 0.0;
   }

   // K_eff at d_{n+1}, the identity on the constrained dofs.
   mfem::Operator &GetGradient(const mfem::Vector &disp_new) const override
   {
      add(1.0 - alpha_f, disp_new, alpha_f, disp_old, disp_mid);
      const auto &stiffness =
         dynamic_cast<const mfem::SparseMatrix &>(internal_force_form.GetGradient(disp_mid));

      tangent.reset(mfem::Add((1.0 - alpha_m) / (beta * dt * dt), mass,
                              1.0 - alpha_f, stiffness));
      for (int dof : ess_tdof_list)
         tangent->EliminateRowCol(dof, mfem::Operator::DIAG_ONE);
      return *tangent;
   }

   // a_0 from the balance at t = 0, M a_0 = F_ext(0) - R_int(d_0); zero on the
   // constrained dofs.
   void get_initial_acceleration(const mfem::Vector &disp,
                                 const mfem::Vector &input_external_force,
                                 mfem::Vector &acce) const
   {
      mfem::Vector rhs(height);
      internal_force_form.Mult(disp, rhs);
      rhs.Neg();
      rhs += input_external_force;

      // M with the constrained rows and columns eliminated.
      mfem::SparseMatrix mass_free(mass);
      for (int dof : ess_tdof_list)
      {
         mass_free.EliminateRowCol(dof, mfem::Operator::DIAG_ONE);
         rhs(dof) = 0.0;
      }

      mfem::CGSolver mass_solver;
      mass_solver.SetOperator(mass_free);
      mass_solver.SetRelTol(1.0e-12);
      mass_solver.SetAbsTol(0.0);
      mass_solver.SetMaxIter(1000);
      mass_solver.SetPrintLevel(0);
      acce = 0.0;
      mass_solver.Mult(rhs, acce);
      MFEM_VERIFY(mass_solver.GetConverged(), "The initial acceleration did not converge.");
   }

private:
   const mfem::NonlinearForm &internal_force_form;   // R_int(d) and K(d)
   const mfem::SparseMatrix &mass;                   // M
   const mfem::Array<int> &ess_tdof_list;            // constrained dofs
   const double dt;                                  // time step
   const double alpha_m, alpha_f, gamma, beta;       // generalized-alpha parameters

   mfem::Vector disp_old, velo_old, acce_old;        // d_n, v_n, a_n
   mfem::Vector external_force;                      // F_ext(t_{n+1-alpha_f})

   // Work vectors of Mult and GetGradient.
   mutable mfem::Vector disp_mid, acce_new, acce_mid, internal_force;
   mutable std::unique_ptr<mfem::SparseMatrix> tangent;   // K_eff
};

#endif

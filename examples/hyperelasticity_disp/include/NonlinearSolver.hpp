// ============================================================================
// NonlinearSolver.hpp
//
// Interface of the two nonlinear solvers of hyperelasticity_disp, that of
// driver_static and that of driver_dynamic, which TimeSolver calls:
//    NonlinearSolver_Static    static, no inertia: R(u_{n+1}, t_{n+1}) = 0;
//    NonlinearSolver_Dynamic   dynamics by the generalized-alpha method of
//                              second order.
// Both solve one step from the state of time_n to that of time, on disp,
// velo and acce; the static one leaves velo and acce untouched, and only
// the displacement is saved.
//
// Author: Chongran Zhao
// Date: Oct. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_HPP
#define NONLINEAR_SOLVER_HPP

#include <mfem.hpp>

#include "INonlinearSolver.hpp"

class NonlinearSolver : public INonlinearSolver
{
public:
   // Completes disp, velo and acce, the initial state at time tt.
   virtual void initialize(double tt, mfem::ParGridFunction &disp, mfem::ParGridFunction &velo,
                           mfem::ParGridFunction &acce) = 0;

   // Solves the step from time_n to time: disp, velo and acce, the state of
   // time_n, become that of time. Returns the number of Newton iterations.
   virtual int solve(double time_n, double time, mfem::ParGridFunction &disp,
                     mfem::ParGridFunction &velo, mfem::ParGridFunction &acce) = 0;

   // Whether velo and acce are solved, and saved with disp.
   virtual bool is_dynamic() const = 0;

protected:
   NonlinearSolver(int num_dofs, MPI_Comm comm) : INonlinearSolver(num_dofs, comm) {}
};

#endif

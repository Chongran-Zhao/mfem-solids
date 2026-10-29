// ============================================================================
// NonlinearSolver.hpp
//
// Interface of the two nonlinear solvers of sphere_rotation, one per driver,
// which TimeSolver calls:
//    NonlinearSolver_TaylorHood_Dynamic   driver_TaylorHood_dynamic: Taylor-Hood
//                                         elements, generalized-alpha of second
//                                         order;
//    NonlinearSolver_VMS_Dynamic          driver_vms_dynamic: equal-order
//                                         elements with VMS, the rate form by
//                                         generalized-alpha of first order.
// Each solves one step from the state of time_n to that of time, on disp,
// velo, acce and pres.
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
   // Completes disp, velo, acce and pres, the initial state at time tt.
   virtual void initialize(double tt, mfem::ParGridFunction &disp, mfem::ParGridFunction &velo,
                           mfem::ParGridFunction &acce, mfem::ParGridFunction &pres) = 0;

   // Solves the step from time_n to time: disp, velo, acce and pres, the
   // state of time_n, become that of time. Returns the number of Newton
   // iterations.
   virtual int solve(double time_n, double time, mfem::ParGridFunction &disp,
                     mfem::ParGridFunction &velo, mfem::ParGridFunction &acce,
                     mfem::ParGridFunction &pres) = 0;

protected:
   NonlinearSolver(int num_dofs, MPI_Comm comm) : INonlinearSolver(num_dofs, comm) {}
};

#endif

// ============================================================================
// NonlinearSolver_Dynamic_Mixed_FixedPressure.hpp
//
// NonlinearSolver_Dynamic_Mixed with the pressure fixed at the center of the
// ball. The material is fully incompressible and the displacement is
// prescribed on the whole surface, so the pressure is defined only up to a
// constant: a constant pressure p_0 does no work on any admissible
// displacement, int p_0 div du dV = p_0 int du . n dA = 0, and the tangent
// is singular. Newton's method therefore keeps the pressure dof at the
// vertex nearest to the origin at its initial value 0, as a constrained dof:
// its residual is zero and its row and column of the tangent the identity.
// The pressure elsewhere is relative to the center; the motion is not
// affected. The initial acceleration needs nothing more, since it solves
// only for the displacement.
//
// Author: Chongran Zhao
// Date: Oct. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NONLINEAR_SOLVER_DYNAMIC_MIXED_FIXED_PRESSURE_HPP
#define NONLINEAR_SOLVER_DYNAMIC_MIXED_FIXED_PRESSURE_HPP

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "GlobalAssembly_Mixed.hpp"
#include "NonlinearSolver_Dynamic_Mixed.hpp"
#include "TimeMethod_GenAlpha.hpp"

class NonlinearSolver_Dynamic_Mixed_FixedPressure : public NonlinearSolver_Dynamic_Mixed
{
public:
   // As NonlinearSolver_Dynamic_Mixed, with the spaces of the global
   // assembly, from which the fixed pressure dof is found.
   NonlinearSolver_Dynamic_Mixed_FixedPressure(
      mfem::ParFiniteElementSpace &space_u, mfem::ParFiniteElementSpace &space_p,
      std::unique_ptr<GlobalAssembly_Mixed> input_global_assembly,
      std::unique_ptr<TimeMethod_GenAlpha> input_time_method, const YAML::Node &solver)
      : NonlinearSolver_Dynamic_Mixed(std::move(input_global_assembly),
                                      std::move(input_time_method), solver)
   {
      // Each rank finds the nearest to the origin of the vertices whose
      // pressure dof it owns, MPI_MINLOC picks the nearest of all, and its
      // rank fixes that dof, numbered after the displacement ones. In H1,
      // the dof of vertex vv is the vv-th one.
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
         fixed_dof.Append(space_u.GetTrueVSize() + nearest_dof);
   }

   // R_dyn of NonlinearSolver_Dynamic_Mixed, also zero at the fixed dof.
   void Mult(const mfem::Vector &sol, mfem::Vector &residual) const override
   {
      NonlinearSolver_Dynamic_Mixed::Mult(sol, residual);
      for (int dof : fixed_dof)
         residual(dof) = 0.0;
   }

   // K_eff of NonlinearSolver_Dynamic_Mixed, also the identity at the fixed
   // dof.
   mfem::Operator &GetGradient(const mfem::Vector &sol) const override
   {
      auto &tangent = static_cast<mfem::HypreParMatrix &>(
         NonlinearSolver_Dynamic_Mixed::GetGradient(sol));
      tangent.EliminateBC(fixed_dof, mfem::Operator::DIAG_ONE);
      return tangent;
   }

private:
   mfem::Array<int> fixed_dof;   // the fixed pressure dof, if this rank owns it, of [u; p]
};

#endif

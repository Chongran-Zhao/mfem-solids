// ============================================================================
// ILoadData.hpp
//
// Interface of the loading of a case: the body force, the prescribed
// displacement and its velocity, and the traction, on the faces named in
// config.yaml, and, for the dynamics, the initial velocity. Each case
// derives its own, LoadData, in its folder. The time tt is the load
// factor t = n / N in the static programs, and the physical time in seconds
// in the dynamic ones.
//
// Author: Chongran Zhao
// Date: Oct. 13, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef ILOAD_DATA_HPP
#define ILOAD_DATA_HPP

#include <string>

#include <mfem.hpp>

#include "Vector_3D.hpp"

class ILoadData
{
public:
   virtual ~ILoadData() = default;

   // Initial velocity v_0(pt) of the dynamics, at the reference position pt;
   // zero unless the case gives one.
   virtual Vector_3D initial_velo(const mfem::Vector &pt) const
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // Body force per unit reference volume, rho_0 b(pt, tt); zero unless the
   // case gives one.
   virtual Vector_3D body_force(const mfem::Vector &pt, double tt) const
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // Nominal traction (force per reference area) on the named face.
   virtual Vector_3D surface_traction(const mfem::Vector &pt, double tt,
                                      const std::string &face) const = 0;

   // Prescribed displacement on the named face.
   virtual Vector_3D disp_loading(const mfem::Vector &pt, double tt,
                                  const std::string &face) const = 0;

   // Velocity of the prescribed displacement, the time derivative of
   // disp_loading, for the initial state of the dynamics; keep the two
   // consistent.
   virtual Vector_3D velo_loading(const mfem::Vector &pt, double tt,
                                  const std::string &face) const = 0;
};

#endif

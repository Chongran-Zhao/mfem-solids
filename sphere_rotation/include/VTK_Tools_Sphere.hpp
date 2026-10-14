// ============================================================================
// VTK_Tools_Sphere.hpp
//
// VTK_Tools in spherical components (r, theta, phi), with theta from the z
// axis and phi about it: the displacement, the velocity, the acceleration,
// the Cauchy and the second Piola-Kirchhoff stresses and the Green-Lagrange
// strain are written in the basis Q = [e_r e_theta e_phi] of the point, in
// the arrays <name>_sph; the displacement is also written in Cartesian
// components. On the z axis phi is taken as 0, at the center also theta.
//
// Author: Chongran Zhao
// Date: Oct. 13, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef VTK_TOOLS_SPHERE_HPP
#define VTK_TOOLS_SPHERE_HPP

#include <cmath>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "Tensor2_3D.hpp"
#include "VTK_Tools.hpp"

class VTK_Tools_Sphere : public VTK_Tools
{
public:
   using VTK_Tools::VTK_Tools;

protected:
   // Q = [e_r e_theta e_phi] at the point pt, the basis vectors as columns,
   //    e_r     = (sin theta cos phi, sin theta sin phi, cos theta),
   //    e_theta = (cos theta cos phi, cos theta sin phi, -sin theta),
   //    e_phi   = (-sin phi, cos phi, 0);
   // on the z axis phi = 0, at the center also theta = 0.
   Tensor2_3D get_basis(const mfem::Vector &pt) const override
   {
      const double radius = pt.Norml2();
      const double theta = (radius > 0.0) ? std::acos(pt(2) / radius) : 0.0;
      const double phi = std::atan2(pt(1), pt(0));
      const double sin_theta = std::sin(theta), cos_theta = std::cos(theta);
      const double sin_phi = std::sin(phi), cos_phi = std::cos(phi);
      return Tensor2_3D(sin_theta * cos_phi, cos_theta * cos_phi, -sin_phi,
                        sin_theta * sin_phi, cos_theta * sin_phi, cos_phi,
                        cos_theta, -sin_theta, 0.0);
   }

   // r, theta, phi for x, y, z.
   std::vector<std::string> get_vector_names() const override { return {"r", "theta", "phi"}; }
   std::vector<std::string> get_tensor_names() const override
   {
      return {"rr", "theta_theta", "phi_phi", "r_theta", "theta_phi", "r_phi"};
   }

   std::string get_suffix() const override { return "_sph"; }
};

#endif

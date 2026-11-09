// ============================================================================
// LoadData.hpp
//
// The loading of the helical crack of Bourdin (2007, Section 3): a body force
// along +z on the top cap, element attribute 4, as in the paper, of .968 t
// per unit volume as in the mef90 runs of ~/runs/HelicalCrack2007L016
// (cs0004, bodyForce 0,0,.968, linear in time), so that the force on the cap,
// of volume pi R^2 tc = pi / 2, is F = 1.52 t; the factor is a fit to Fig. 2
// of the paper there, not given in the paper. The faces have no prescribed
// displacement other than zero, and no traction. It passes its functions to
// DirichletBoundary, NeumannBoundary and the driver.
//
// Author: Chongran Zhao
// Date: Nov. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

#include <map>
#include <string>

#include <mfem.hpp>

#include "Vector_3D.hpp"

class LoadData
{
public:
   // Element attribute of the region the body force acts on, the top cap.
   static constexpr int body_force_region = 4;

   // Body force per unit reference volume, rho_0 b(pt, tt), on that region.
   static Vector_3D body_force(const mfem::Vector &pt, double tt)
   {
      return Vector_3D(0.0, 0.0, 0.968 * tt);
   }

   // Nominal traction (force per reference area) on the named face: none.
   static Vector_3D surface_traction(const mfem::Vector &pt, double tt,
                                     const std::string &face)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // Prescribed displacement on the named face, along the selected direction:
   // zero on every face.
   static Vector_3D disp_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      switch (get_face(face))
      {
         case faces::clamped:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::top:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::crack:
            return Vector_3D(0.0, 0.0, 0.0);
      }
      return Vector_3D(0.0, 0.0, 0.0);
   }

private:

   enum class faces { clamped, top, crack };

   // Face name -> faces.
   static faces get_face(const std::string &face)
   {
      static const std::map<std::string, faces> face_map = {
         {"clamped", faces::clamped}, {"top", faces::top}, {"crack", faces::crack}};
      MFEM_VERIFY(face_map.count(face), "Unknown face \"" << face << "\".");
      return face_map.at(face);
   }
};

#endif

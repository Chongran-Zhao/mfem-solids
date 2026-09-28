// ============================================================================
// LoadData.hpp
//
// Defines the loading (body force, prescribed displacement and traction).
//
// Author: Chongran Zhao
// Date: Sep. 27, 2026
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
   // Body force per unit reference volume, rho_0 b(pt, tt).
   // Examples:
   //   gravity along -z, rho_0 = 1000:
   //     return Vector_3D(0.0, 0.0, -1000.0 * 9.81 * tt);
   //   growing linearly with x:
   //     return Vector_3D(0.0, 0.0, -100.0 * pt(0) * tt);
   static Vector_3D body_force(const mfem::Vector &pt, double tt)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // Nominal traction (force per reference area) on the named face.
   // Examples:
   //   uniform, along -z:
   //     return Vector_3D(0.0, 0.0, -2275.0 * tt);
   //   linear in y over the face y in [0, 0.1], bending about x:
   //     return Vector_3D(0.0, 0.0, -2275.0 * (pt(1) - 0.05) / 0.05 * tt);
   static Vector_3D surface_traction(const mfem::Vector &pt, double tt,
                                     const std::string &face)
   {
      switch (get_face(face))
      {
         case faces::left:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::right:
            return Vector_3D(0.0, 0.0, -2275.0 * tt);
         case faces::front:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::back:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::bottom:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::top:
            return Vector_3D(0.0, 0.0, 0.0);
      }
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // Prescribed displacement on the named face, along the selected direction.
   // Examples:
   //   uniform, along -z:
   //     return Vector_3D(0.0, 0.0, -0.5 * tt);
   //   rotation by angle a about the x axis through (y, z) = (0.05, 0.05):
   //     const double a = 0.3 * tt;
   //     const double y = pt(1) - 0.05, z = pt(2) - 0.05;
   //     return Vector_3D(0.0, std::cos(a) * y - std::sin(a) * z - y,
   //                           std::sin(a) * y + std::cos(a) * z - z);
   static Vector_3D disp_driven(const mfem::Vector &pt, double tt,
                                    const std::string &face)
   {
      switch (get_face(face))
      {
         case faces::left:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::right:
            return Vector_3D(0.0, 0.0, -0.5 * tt);
         case faces::front:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::back:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::bottom:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::top:
            return Vector_3D(0.0, 0.0, 0.0);
      }
      return Vector_3D(0.0, 0.0, 0.0);
   }

private:

  enum class faces { left, right, front, back, bottom, top };

   // Face name -> faces.
   static faces get_face(const std::string &face)
   {
      static const std::map<std::string, faces> face_map = {
         {"left", faces::left},     {"right", faces::right},
         {"front", faces::front},   {"back", faces::back},
         {"bottom", faces::bottom}, {"top", faces::top}};
      MFEM_VERIFY(face_map.count(face), "Unknown face \"" << face << "\".");
      return face_map.at(face);
   }
};

#endif

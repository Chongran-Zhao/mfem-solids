// ============================================================================
// LoadData.hpp
//
// The loading of the bar of Bourdin, Francfort and Marigo (J. Elasticity
// 2008, Section 8.3.1), x in [0, L] with L = 10, clamped at x = 0 and pulled
// by the displacement t L at x = L, so that the strain of the uncracked bar
// is t, the time of the load steps. It passes its functions to
// DirichletBoundary and NeumannBoundary.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
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
   // Length of the bar.
   static constexpr double length = 10.0;

   // Nominal traction (force per reference area) on the named face: none.
   static Vector_3D surface_traction(const mfem::Vector &pt, double tt,
                                     const std::string &face)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // Prescribed displacement on the named face, along the selected direction:
   // t L along x at x = L.
   static Vector_3D disp_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      switch (get_face(face))
      {
         case faces::left:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::right:
            return Vector_3D(tt * length, 0.0, 0.0);
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

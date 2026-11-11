// ============================================================================
// LoadData.hpp
//
// The loading of the helical crack of Bourdin, Francfort and Marigo (2008,
// Section 6, Fig. 30): the tube is clamped at its left cross-section and
// pulled by a displacement delta along e1 at its right one, monotonically
// increasing; delta = t here, from 0 to 50 as in the mef90 runs of
// ~/runs/ATTEMPTS.md. The paper does not say whether the right cross-section
// may move across; config.yaml fixes it, u = (delta, 0, 0) there. No
// traction and no body force. It passes its functions to DirichletBoundary
// and NeumannBoundary.
//
// Author: Chongran Zhao
// Date: Nov. 10, 2026
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
   // Nominal traction (force per reference area) on the named face: none.
   static Vector_3D surface_traction(const mfem::Vector &pt, double tt,
                                     const std::string &face)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // Prescribed displacement on the named face, along the selected direction:
   // delta = tt along e1 on the right cross-section, zero elsewhere.
   static Vector_3D disp_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      switch (get_face(face))
      {
         case faces::left:
            return Vector_3D(0.0, 0.0, 0.0);
         case faces::right:
            return Vector_3D(tt, 0.0, 0.0);
         case faces::crack:
            return Vector_3D(0.0, 0.0, 0.0);
      }
      return Vector_3D(0.0, 0.0, 0.0);
   }

private:

   enum class faces { left, right, crack };

   // Face name -> faces.
   static faces get_face(const std::string &face)
   {
      static const std::map<std::string, faces> face_map = {
         {"left", faces::left}, {"right", faces::right}, {"crack", faces::crack}};
      MFEM_VERIFY(face_map.count(face), "Unknown face \"" << face << "\".");
      return face_map.at(face);
   }
};

#endif

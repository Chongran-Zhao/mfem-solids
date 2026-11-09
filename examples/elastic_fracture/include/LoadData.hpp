// ============================================================================
// LoadData.hpp
//
// The loading of the helical crack of Bourdin (2007, Section 3). In the
// mef90 runs of ~/runs/HelicalCrack2007L016 the cylinder is loaded by a body
// force .968 t along z on the top cap, which is not yet applied here; the
// faces have no prescribed displacement other than zero, and no traction.
// It passes its functions to DirichletBoundary and NeumannBoundary.
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

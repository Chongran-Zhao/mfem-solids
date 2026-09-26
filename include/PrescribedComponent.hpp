// ============================================================================
// PrescribedComponent.hpp
//
// One prescribed displacement component on one named face, as read from the
// Dirichlet conditions of config.yaml.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef PRESCRIBED_COMPONENT_HPP
#define PRESCRIBED_COMPONENT_HPP

#include "mfem.hpp"
#include <string>

struct PrescribedComponent
{
   std::string face;               // face name written by read_mesh
   mfem::Array<int> face_marker;   // boundary attributes of the face
   int component;                  // 0, 1, 2 for x, y, z
   double value;                   // final value, reached at the last step
   int num_dofs;                   // number of constrained dofs
};

#endif

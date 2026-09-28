// ============================================================================
// Vector_3D.hpp
//
// First-order tensor (vector) in 3D, a_i with i = 0, 1, 2.
//
// Author: Chongran Zhao
// Date: Sep. 25, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef VECTOR_3D_HPP
#define VECTOR_3D_HPP

#include <array>

class Vector_3D
{
public:
   // Zero vector.
   Vector_3D() { vec.fill(0.0); }

   // Vector with the given components.
   Vector_3D(double a0, double a1, double a2) : vec{a0, a1, a2} {}

   double &operator()(int ii) { return vec[ii]; }
   double operator()(int ii) const { return vec[ii]; }

private:
   std::array<double, 3> vec;
};

#endif

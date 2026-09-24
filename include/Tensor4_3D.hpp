// ============================================================================
// Tensor4_3D.hpp
//
// Fourth-order tensor in 3D, C_ijkl with i, j, k, l = 0, 1, 2.
// 
// Author: Chongran Zhao
// Date: Sep. 24, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TENSOR4_3D_HPP
#define TENSOR4_3D_HPP

#include <array>

class Tensor4_3D
{
public:
   // Zero tensor.
   Tensor4_3D() { ten.fill(0.0); }

   double &operator()(int ii, int jj, int kk, int ll)
   {
      return ten[27 * ii + 9 * jj + 3 * kk + ll];
   }

   double operator()(int ii, int jj, int kk, int ll) const
   {
      return ten[27 * ii + 9 * jj + 3 * kk + ll];
   }

private:
   std::array<double, 81> ten;
};

#endif

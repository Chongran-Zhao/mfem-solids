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

#include "mfem.hpp"

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

   // 6 x 6 Voigt form, out(P, Q) = C_ijkl, with P from (i,j) and Q from (k,l)
   // in the order
   mfem::DenseMatrix to_voigt() const
   {
      mfem::DenseMatrix out(6);
      for (int nn = 0; nn < 9; nn++)
         for (int mm = 0; mm < 9; mm++)
            out(voigt_map[nn], voigt_map[mm]) = ten[9 * nn + mm];
      return out;
   }

private:
   std::array<double, 81> ten;

   // Voigt index P of the pair (i,j), stored at 3i + j.
   static constexpr std::array<int, 9> voigt_map{{0, 5, 4,
                                                  5, 1, 3,
                                                  4, 3, 2}};
};

#endif

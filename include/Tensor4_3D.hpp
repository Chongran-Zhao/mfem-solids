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

#include "Tensor2_3D.hpp"
#include "mfem.hpp"
#include <array>

class Tensor4_3D
{
public:
   // Zero tensor.
   Tensor4_3D() { ten.fill(0.0); }

   double &operator()(int ii, int jj, int kk, int ll)
   { return ten[27 * ii + 9 * jj + 3 * kk + ll]; }

   double operator()(int ii, int jj, int kk, int ll) const
   { return ten[27 * ii + 9 * jj + 3 * kk + ll]; }

   Tensor4_3D &operator+=(const Tensor4_3D &BB)
   {
      for (int nn = 0; nn < 81; nn++)
         ten[nn] += BB.ten[nn];
      return *this;
   }

   Tensor4_3D &operator*=(double aa)
   {
      for (double &xx : ten)
         xx *= aa;
      return *this;
   }

   // 6 x 6 Voigt form, out(P, Q) = C_ijkl, with P from (i,j) and Q from (k,l)
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

inline Tensor4_3D operator+(Tensor4_3D AA, const Tensor4_3D &BB) { return AA += BB; }
inline Tensor4_3D operator*(double aa, Tensor4_3D BB) { return BB *= aa; }

// Outer product of two second-order tensors (A otimes B)_ijkl = A_ij B_kl.
inline Tensor4_3D otimes(const Tensor2_3D &AA, const Tensor2_3D &BB)
{
   Tensor4_3D out;
   for (int ii = 0; ii < 3; ii++)
      for (int jj = 0; jj < 3; jj++)
         for (int kk = 0; kk < 3; kk++)
            for (int ll = 0; ll < 3; ll++)
               out(ii, jj, kk, ll) = AA(ii, jj) * BB(kk, ll);
   return out;
}

// Par C^-1 / Par C = - C^-1 odot C^-1
// Symmetrized product (A odot B)_ijkl = 1/2 (A_ik B_jl + A_il B_jk).
// Symmetric in k and l; also symmetric in i and j when A = B.
inline Tensor4_3D odot(const Tensor2_3D &AA, const Tensor2_3D &BB)
{
   Tensor4_3D out;
   for (int ii = 0; ii < 3; ii++)
      for (int jj = 0; jj < 3; jj++)
         for (int kk = 0; kk < 3; kk++)
            for (int ll = 0; ll < 3; ll++)
               out(ii, jj, kk, ll) = 0.5 * (AA(ii, kk) * BB(jj, ll) + AA(ii, ll) * BB(jj, kk));
   return out;
}

#endif

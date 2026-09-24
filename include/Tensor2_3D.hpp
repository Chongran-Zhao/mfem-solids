// ============================================================================
// Tensor2_3D.hpp
//
// Second-order tensor in 3D, A_ij with i, j = 0, 1, 2.
// 
// Author: Chongran Zhao
// Date: Sep. 24, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef TENSOR2_3D_HPP
#define TENSOR2_3D_HPP

#include <array>

class Tensor2_3D
{
public:
   // Zero tensor.
   Tensor2_3D() { ten.fill(0.0); }

   // Tensor with the given components, row by row.
   Tensor2_3D(double a00, double a01, double a02,
              double a10, double a11, double a12,
              double a20, double a21, double a22)
      : ten{a00, a01, a02, a10, a11, a12, a20, a21, a22} {}

   static Tensor2_3D identity()
   {
      return Tensor2_3D(1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0);
   }

   double &operator()(int ii, int jj) { return ten[3 * ii + jj]; }
   double operator()(int ii, int jj) const { return ten[3 * ii + jj]; }

   Tensor2_3D transpose() const
   {
      const Tensor2_3D &AA = *this;
      return Tensor2_3D(AA(0, 0), AA(1, 0), AA(2, 0),
                        AA(0, 1), AA(1, 1), AA(2, 1),
                        AA(0, 2), AA(1, 2), AA(2, 2));
   }

   double trace() const { return (*this)(0, 0) + (*this)(1, 1) + (*this)(2, 2); }

   double det() const
   {
      const Tensor2_3D &AA = *this;
      return AA(0, 0) * (AA(1, 1) * AA(2, 2) - AA(1, 2) * AA(2, 1))
             - AA(0, 1) * (AA(1, 0) * AA(2, 2) - AA(1, 2) * AA(2, 0))
             + AA(0, 2) * (AA(1, 0) * AA(2, 1) - AA(1, 1) * AA(2, 0));
   }

   // A^-1 = adj(A) / det A.
   Tensor2_3D inverse() const
   {
      const Tensor2_3D &AA = *this;
      const double det_A = det();

      Tensor2_3D out(AA(1, 1) * AA(2, 2) - AA(1, 2) * AA(2, 1),
                     AA(0, 2) * AA(2, 1) - AA(0, 1) * AA(2, 2),
                     AA(0, 1) * AA(1, 2) - AA(0, 2) * AA(1, 1),
                     AA(1, 2) * AA(2, 0) - AA(1, 0) * AA(2, 2),
                     AA(0, 0) * AA(2, 2) - AA(0, 2) * AA(2, 0),
                     AA(0, 2) * AA(1, 0) - AA(0, 0) * AA(1, 2),
                     AA(1, 0) * AA(2, 1) - AA(1, 1) * AA(2, 0),
                     AA(0, 1) * AA(2, 0) - AA(0, 0) * AA(2, 1),
                     AA(0, 0) * AA(1, 1) - AA(0, 1) * AA(1, 0));
      out *= 1.0 / det_A;
      return out;
   }

   Tensor2_3D &operator+=(const Tensor2_3D &BB)
   {
      for (int nn = 0; nn < 9; nn++)
         ten[nn] += BB.ten[nn];
      return *this;
   }

   Tensor2_3D &operator-=(const Tensor2_3D &BB)
   {
      for (int nn = 0; nn < 9; nn++)
         ten[nn] -= BB.ten[nn];
      return *this;
   }

   Tensor2_3D &operator*=(double aa)
   {
      for (double &xx : ten)
         xx *= aa;
      return *this;
   }

private:
   std::array<double, 9> ten;
};

inline Tensor2_3D operator+(Tensor2_3D AA, const Tensor2_3D &BB) { return AA += BB; }
inline Tensor2_3D operator-(Tensor2_3D AA, const Tensor2_3D &BB) { return AA -= BB; }
inline Tensor2_3D operator*(double aa, Tensor2_3D BB) { return BB *= aa; }
inline Tensor2_3D operator*(Tensor2_3D BB, double aa) { return BB *= aa; }

// Matrix product (A B)_ij = A_ik B_kj.
inline Tensor2_3D operator*(const Tensor2_3D &AA, const Tensor2_3D &BB)
{
   Tensor2_3D out;
   for (int ii = 0; ii < 3; ii++)
      for (int jj = 0; jj < 3; jj++)
         for (int kk = 0; kk < 3; kk++)
            out(ii, jj) += AA(ii, kk) * BB(kk, jj);
   return out;
}

#endif

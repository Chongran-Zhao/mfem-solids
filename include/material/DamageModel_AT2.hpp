// ============================================================================
// DamageModel_AT2.hpp
//
// The AT2 damage model of mef90 (m_MEF90_DefMechAT2.F90):
//    a(alpha) = eta + (1 - eta) (1 - alpha)^2,
//    w(alpha) = alpha^2,   c_w = 1/2.
// Since w'(0) = 0, the damage grows from the start of the loading.
//
// Author: Chongran Zhao
// Date: Nov. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef DAMAGE_MODEL_AT2_HPP
#define DAMAGE_MODEL_AT2_HPP

#include "IDamageModel.hpp"

class DamageModel_AT2 : public IDamageModel
{
public:
   using IDamageModel::IDamageModel;

   // a = eta + (1 - eta) (1 - alpha)^2
   double get_a(double alpha) const override
   {
      return eta + (1.0 - eta) * (1.0 - alpha) * (1.0 - alpha);
   }

   // a' = -2 (1 - eta) (1 - alpha)
   double get_da(double alpha) const override
   {
      return -2.0 * (1.0 - eta) * (1.0 - alpha);
   }

   // a'' = 2 (1 - eta)
   double get_d2a(double alpha) const override { return 2.0 * (1.0 - eta); }

   // w = alpha^2
   double get_w(double alpha) const override { return alpha * alpha; }

   // w' = 2 alpha
   double get_dw(double alpha) const override { return 2.0 * alpha; }

   // w'' = 2
   double get_d2w(double alpha) const override { return 2.0; }

   double get_c_w() const override { return 0.5; }
};

#endif

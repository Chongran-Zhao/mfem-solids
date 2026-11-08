// ============================================================================
// ElasticEnergyCoefficient.hpp
//
// The elastic energy density psi_e(F) at a point, from the displacement: the
// psi_e of the phase-field problem. It reads the displacement at every
// evaluation, and neither the material nor the displacement are its own.
//
// Author: Chongran Zhao
// Date: Nov. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef ELASTIC_ENERGY_COEFFICIENT_HPP
#define ELASTIC_ENERGY_COEFFICIENT_HPP

#include <mfem.hpp>

#include "IMaterialModel.hpp"
#include "Tensor2_3D.hpp"

class ElasticEnergyCoefficient : public mfem::Coefficient
{
public:
   ElasticEnergyCoefficient(const IMaterialModel &input_material,
                            const mfem::GridFunction &input_disp)
      : material(input_material), disp(input_disp) {}

   // Required by MFEM: overrides mfem::Coefficient::Eval.
   // psi_e(F), with F_kJ = delta_kJ + u_k,J.
   double Eval(mfem::ElementTransformation &elem_map,
               const mfem::IntegrationPoint &quad_pt) override
   {
      elem_map.SetIntPoint(&quad_pt);
      disp.GetVectorGradient(elem_map, grad_u);
      Tensor2_3D F = Tensor2_3D::identity();
      for (int kk = 0; kk < 3; kk++)
         for (int JJ = 0; JJ < 3; JJ++)
            F(kk, JJ) += grad_u(kk, JJ);
      return material.get_strain_energy(F);
   }

private:
   const IMaterialModel &material;     // psi_e(F)
   const mfem::GridFunction &disp;     // displacement
   mfem::DenseMatrix grad_u;           // u_k,J at the point
};

#endif

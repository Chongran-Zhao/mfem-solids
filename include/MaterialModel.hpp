// ============================================================================
// MaterialModel.hpp
//
// The material of the problems in this project, shared by the driver,
// write_paraview and write_traction_disp. MaterialModel is the model class, get_material_model gives
// it with its parameters; changing the material means changing this file.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_HPP
#define MATERIAL_MODEL_HPP

#include "CompressibleNeoHookean.hpp"

using MaterialModel = CompressibleNeoHookean;

// Young's modulus and Poisson's ratio of the reference case, converted to
// the shear and bulk moduli. The moduli are in the stress unit of
// config.yaml.
inline MaterialModel get_material_model()
{
   const double young = 540.0e3;
   const double poisson = 0.324;
   const double mu = young / (2.0 * (1.0 + poisson));
   const double kappa = young / (3.0 * (1.0 - 2.0 * poisson));

   return MaterialModel(kappa, mu);
}

#endif

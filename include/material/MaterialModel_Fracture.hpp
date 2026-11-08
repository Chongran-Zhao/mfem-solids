// ============================================================================
// MaterialModel_Fracture.hpp
//
// The material of phase-field fracture: an elastic material, which gives
// psi_e(F), and a surface energy, which gives g(alpha), w(alpha) and c_grad,
// for the energy density
//    g(alpha) psi_e(F) + G_c ( w(alpha) + c_grad |Grad alpha|^2 ).
// An unbreakable material, as mef90's elastic: true (m_MEF90_DefMechATClass
// and m_MEF90_DefMechAssembly), keeps g = 1: its elastic energy is not
// degraded and does not drive the phase field, since g' = 0, while the
// surface energy still holds the phase field at 1.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef MATERIAL_MODEL_FRACTURE_HPP
#define MATERIAL_MODEL_FRACTURE_HPP

#include <memory>
#include <utility>

#include "IMaterialModel.hpp"
#include "ISurfaceEnergy.hpp"
#include "Tensor2_3D.hpp"

class MaterialModel_Fracture
{
public:
   // Takes the ownership of the elastic material and of the surface energy.
   MaterialModel_Fracture(std::unique_ptr<const IMaterialModel> input_elastic,
                          std::unique_ptr<const ISurfaceEnergy> input_surface_energy,
                          bool input_breakable = true)
      : elastic(std::move(input_elastic)),
        surface_energy(std::move(input_surface_energy)),
        breakable(input_breakable) {}

   // psi_e(F), the elastic energy per reference volume, not degraded.
   double get_elastic_energy(const Tensor2_3D &F) const
   {
      return elastic->get_strain_energy(F);
   }

   // g(alpha) and its derivatives; g = 1 if the material is unbreakable.
   double get_g(double alpha) const { return breakable ? surface_energy->get_g(alpha) : 1.0; }
   double get_dg(double alpha) const { return breakable ? surface_energy->get_dg(alpha) : 0.0; }
   double get_d2g(double alpha) const { return breakable ? surface_energy->get_d2g(alpha) : 0.0; }

   // w(alpha), its derivatives and c_grad of the surface energy.
   double get_w(double alpha) const { return surface_energy->get_w(alpha); }
   double get_dw(double alpha) const { return surface_energy->get_dw(alpha); }
   double get_d2w(double alpha) const { return surface_energy->get_d2w(alpha); }
   double get_c_grad() const { return surface_energy->get_c_grad(); }

   double get_G_c() const { return surface_energy->get_G_c(); }
   bool is_breakable() const { return breakable; }

private:
   const std::unique_ptr<const IMaterialModel> elastic;          // psi_e(F)
   const std::unique_ptr<const ISurfaceEnergy> surface_energy;   // g, w, c_grad
   const bool breakable;                                         // false: g = 1
};

#endif

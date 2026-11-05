// ============================================================================
// IMassEquation.hpp
//
// Interface of the mass equation of the mixed displacement-pressure (u/p)
// form, the part in which the formulations differ: LocalAssembly_Mixed
// assembles the momentum equation, the same in all of them, and calls this
// class for the pressure rows R^c, K_pu and K_pp of each element.
//    MassEquation_TaylorHood   the weak J = J(p) of Taylor-Hood elements;
//    MassEquation_VMS          its rate form with the VMS stabilization of
//                              equal-order elements.
// The material comes with every call, since it may change with the region of
// the element.
//
// Author: Chongran Zhao
// Date: Nov. 5, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef I_MASS_EQUATION_HPP
#define I_MASS_EQUATION_HPP

#include <mfem.hpp>

#include "MaterialModel.hpp"
#include "ShapeCache.hpp"
#include "Tensor2_3D.hpp"
#include "Tensor4_3D.hpp"

class IMassEquation
{
public:
   virtual ~IMassEquation() = default;

   // Whether the shapes of a quadrature point must hold N_a,JL.
   virtual bool needs_hessian() const = 0;

   // c_u = du/dsol_u, with sol_u the unknowns of block 0.
   virtual double get_disp_factor() const = 0;

   // Starts an element, from sol_u and pres, its unknowns of blocks 0 and 1,
   // which must outlive the calls for that element. Returns its displacement.
   virtual const mfem::Vector &set_element(const MaterialModel &material,
                                           const mfem::FiniteElement &elem_u,
                                           mfem::ElementTransformation &elem_map,
                                           const mfem::Vector &sol_u,
                                           const mfem::Vector &pres) = 0;

   // Adds to R^c the term of a quadrature point of the element, at F and p.
   virtual void add_residual(const MaterialModel &material,
                             const ShapeCache::PointShapes &point, const Tensor2_3D &F, double p,
                             mfem::Vector &residual_p) = 0;

   // Adds to K_pu and K_pp the terms of a quadrature point of the element;
   // CC_ich and AA_ich are the isochoric elasticity tensors at F.
   virtual void add_tangent(const MaterialModel &material,
                            const ShapeCache::PointShapes &point, const Tensor2_3D &F, double p,
                            const Tensor4_3D &CC_ich, const Tensor4_3D &AA_ich,
                            mfem::DenseMatrix &K_pu, mfem::DenseMatrix &K_pp) = 0;
};

#endif

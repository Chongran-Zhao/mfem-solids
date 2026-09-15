// ============================================================================
// 1d-elastostatics.cpp
//
// One-dimensional linear elastostatics: a bar fixed at x = 0 and pulled by a
// uniform axial traction t at x = L, with no body force.
//
//         u'' = 0    on 0 < x < L
//        u(0) = 0    fixed end        (essential condition)
//     E u'(L) = t    traction         (natural condition)
//
// Discretised with linear H1 elements, the equations solved are
//
//     sum_j ( integral of N_i' N_j' over [0, L] ) d_j = N_i(L) t / E,
//
// for every free degree of freedom i. The cross-sectional area A cancels.
// ============================================================================
#include "mfem.hpp"

int main()
{
   // 1. Problem data, SI units.
   const mfem::real_t length = 1.0;        // L [m]
   const mfem::real_t young = 200.0e9;     // E [Pa]
   const mfem::real_t traction = 100.0e6;  // t [Pa]
   const int num_elements = 4;
   const int order = 1;

   // 2. Mesh: num_elements segments of equal length on [0, L]. The two end
   //    points are boundary attribute 1 (x = 0) and 2 (x = L).
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(num_elements, length);

   // 3. Shape functions N_i: continuous and piecewise linear, one degree of
   //    freedom d_i per node, numbered from 0.
   mfem::H1_FECollection fec(order, mesh.Dimension());
   mfem::FiniteElementSpace fespace(&mesh, &fec);

   // 4. Essential boundary: the degree of freedom at x = 0 is not free.
   mfem::Array<int> fixed_bdr(mesh.bdr_attributes.Max());
   fixed_bdr = 0;
   fixed_bdr[0] = 1;                       // attribute 1, the end at x = 0
   mfem::Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(fixed_bdr, ess_tdof_list);

   // 5. Stiffness matrix: K_ij = integral of N_i' N_j' over [0, L].
   mfem::BilinearForm stiffness(&fespace);
   stiffness.AddDomainIntegrator(new mfem::DiffusionIntegrator);
   stiffness.Assemble();

   // 6. Load vector: F_i = N_i(L) t / E, applied only at x = L.
   mfem::ConstantCoefficient scaled_traction(traction/young);
   mfem::Array<int> end_bdr(mesh.bdr_attributes.Max());
   end_bdr = 0;
   end_bdr[1] = 1;                         // attribute 2, the end at x = L

   mfem::LinearForm load(&fespace);
   load.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(scaled_traction),
                              end_bdr);
   load.Assemble();

   // 7. Displacement u^h, whose coefficients are the d_i. Its value on the
   //    fixed node is the prescribed displacement, zero.
   mfem::GridFunction disp(&fespace);
   disp = 0.0;

   // 8. Keep only the equations of the free degrees of freedom, solve for d,
   //    and put the result back into disp. MFEM keeps the fixed row but
   //    decouples it from the rest, which gives the same d.
   mfem::SparseMatrix stiff_mat;
   mfem::Vector solution, rhs;
   stiffness.FormLinearSystem(ess_tdof_list, disp, load,
                              stiff_mat, solution, rhs);

   mfem::GSSmoother precond(stiff_mat);
   mfem::CGSolver solver;
   solver.SetRelTol(1e-12);
   solver.SetMaxIter(100);
   solver.SetPrintLevel(0);
   solver.SetPreconditioner(precond);
   solver.SetOperator(stiff_mat);
   solver.Mult(rhs, solution);

   stiffness.RecoverFEMSolution(solution, load, disp);

   // 9. Nodal displacements d_i. With linear elements node ii is unknown ii.
   for (int ii = 0; ii < mesh.GetNV(); ++ii)
      mfem::out << "x = " << mesh.GetVertex(ii)[0]
                << "   u = " << disp(ii) << '\n';

   return 0;
}

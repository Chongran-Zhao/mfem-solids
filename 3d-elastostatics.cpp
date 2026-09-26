// ============================================================================
// 3d-elastostatics.cpp
//
// Three-dimensional elastostatics with MFEM, built one step at a time.
// Step 1: read the reference mesh.
// Step 2: create the displacement finite element space.
// Step 3: identify displacement degrees of freedom on constrained boundaries.
// Step 4: assign the prescribed displacement on the right end.
// Step 5: define the material parameters.
// Step 6: build the nonlinear form and evaluate the residual.
// Step 7: solve R(d) = 0 with Newton's method for the first load step.
// Step 8: write the displacement for ParaView.
// ============================================================================
#include "CompressibleHyperelasticIntegrator.hpp"
#include "CompressibleNeoHookean.hpp"
#include "mfem.hpp"

int main()
{
   // 1. Read the reference mesh used by the Total Lagrangian example.
   const char *mesh_file = "/Users/chongran/MFEM/projects/NonlinearElasticity_mfem/input/meshes/Beam_coarse-hex.mesh";
   mfem::Mesh mesh(mesh_file);
   MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3,
               "Expected a three-dimensional mesh embedded in 3D.");

   mfem::out << "Elements: " << mesh.GetNE() << '\n'
             << "Vertices: " << mesh.GetNV() << '\n'
             << "Boundary elements: " << mesh.GetNBE() << '\n';

   // 2. Continuous first-order shape functions interpolate displacement.
   //    Each scalar degree of freedom has three components: u_x, u_y, u_z.
   //    byVDIM stores the components node by node: x_1, y_1, z_1, x_2, ...
   const int dim = mesh.Dimension();
   const int order = 1;
   mfem::H1_FECollection fec(order, dim);
   mfem::FiniteElementSpace fespace(&mesh, &fec, dim,
                                   mfem::Ordering::byVDIM);
   mfem::GridFunction disp(&fespace);
   disp = 0.0;

   mfem::out << "Displacement unknowns: "
             << fespace.GetTrueVSize() << '\n';

   // 3. Boundary attribute 1 is the left end (x = 0): fix u_x, u_y, u_z.
   //    Attribute 2 is the right end (x = 1): prescribe u_y there.
   mfem::Array<int> ess_bdr_x(mesh.bdr_attributes.Max());
   mfem::Array<int> ess_bdr_y(mesh.bdr_attributes.Max());
   mfem::Array<int> ess_bdr_z(mesh.bdr_attributes.Max());
   ess_bdr_x = 0;
   ess_bdr_y = 0;
   ess_bdr_z = 0;
   ess_bdr_x[0] = 1;
   ess_bdr_y[0] = 1;
   ess_bdr_y[1] = 1;
   ess_bdr_z[0] = 1;

   mfem::Array<int> ess_tdof_x, ess_tdof_y, ess_tdof_z, ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr_x, ess_tdof_x, 0);
   fespace.GetEssentialTrueDofs(ess_bdr_y, ess_tdof_y, 1);
   fespace.GetEssentialTrueDofs(ess_bdr_z, ess_tdof_z, 2);
   ess_tdof_list.Append(ess_tdof_x);
   ess_tdof_list.Append(ess_tdof_y);
   ess_tdof_list.Append(ess_tdof_z);

   mfem::out << "Constrained displacement unknowns: "
             << ess_tdof_list.Size() << '\n';

   // 4. The left-end displacement remains zero from initialization. Set only
   //    u_y on boundary 2; null coefficients leave u_x and u_z untouched.
   const double prescribed_uy = 0.02;
   mfem::ConstantCoefficient right_uy(prescribed_uy);
   mfem::Coefficient *right_disp[3] = {nullptr, &right_uy, nullptr};
   mfem::Array<int> right_bdr(mesh.bdr_attributes.Max());
   right_bdr = 0;
   right_bdr[1] = 1;
   disp.ProjectBdrCoefficient(right_disp, right_bdr);

   mfem::out << "Prescribed right-end u_y: " << prescribed_uy << '\n';

   // 5. Convert Young's modulus and Poisson's ratio from the reference case
   //    to the shear and bulk moduli used by our Neo-Hookean material.
   const double young = 540.0e3;
   const double poisson = 0.324;
   const double mu = young / (2.0 * (1.0 + poisson));
   const double kappa = young / (3.0 * (1.0 - 2.0 * poisson));
   const CompressibleNeoHookean material(kappa, mu);

   mfem::out << "Shear modulus: " << mu << " Pa\n"
             << "Bulk modulus: " << kappa << " Pa\n"
             << "Energy at F = I: "
             << material.get_strain_energy(Tensor2_3D::identity()) << '\n';

   // 6. The nonlinear form loops over elements, calls our integrator for the
   //    element residual and tangent, and assembles them. It owns the
   //    integrator; the material must outlive it. Residual entries on the
   //    constrained dofs are set to zero.
   mfem::NonlinearForm nonlinear_form(&fespace);
   nonlinear_form.AddDomainIntegrator(
      new CompressibleHyperelasticIntegrator(material));
   nonlinear_form.SetEssentialTrueDofs(ess_tdof_list);

   // On this serial, conforming mesh every dof is a true dof, so the
   // GridFunction disp can be passed to the solvers directly.
   mfem::Vector residual(fespace.GetTrueVSize());
   nonlinear_form.Mult(disp, residual);

   mfem::out << "Residual norm with the prescribed boundary displacement: "
             << residual.Norml2() << '\n';

   // 7. Newton's method: at each iteration assemble the tangent K(d), solve
   //    K dd = R(d) with CG, and update d -= dd. The tangent is symmetric, and
   //    NonlinearForm replaces the constrained rows and columns by identity,
   //    so CG applies and the boundary values stay fixed.
   mfem::GSSmoother preconditioner;
   mfem::CGSolver linear_solver;
   linear_solver.SetRelTol(1.0e-6);
   linear_solver.SetAbsTol(0.0);
   linear_solver.SetMaxIter(500);
   linear_solver.SetPrintLevel(0);
   linear_solver.SetPreconditioner(preconditioner);

   // Converged when ||R(d)|| <= 1e-8 ||R(d_0)||. iterative_mode = true starts
   // from disp, which carries the prescribed boundary displacement.
   mfem::NewtonSolver newton_solver;
   newton_solver.SetOperator(nonlinear_form);
   newton_solver.SetSolver(linear_solver);
   newton_solver.SetRelTol(1.0e-8);
   newton_solver.SetAbsTol(0.0);
   newton_solver.SetMaxIter(20);
   newton_solver.SetPrintLevel(1);
   newton_solver.iterative_mode = true;

   // An empty right-hand side means solving R(d) = 0.
   const mfem::Vector zero_rhs;
   newton_solver.Mult(zero_rhs, disp);
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge.");

   mfem::out << "Newton iterations: " << newton_solver.GetNumIterations() << '\n';

   // 8. Save the reference mesh with the displacement field as VTU files in
   //    ParaView/3d-elastostatics. The mesh itself is not moved; in ParaView,
   //    apply "Warp By Vector" with "displacement" to see the deformed beam.
   mfem::ParaViewDataCollection paraview("3d-elastostatics", &mesh);
   paraview.SetPrefixPath("ParaView");
   paraview.SetLevelsOfDetail(order);
   paraview.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview.RegisterField("displacement", &disp);
   paraview.SetCycle(1);
   paraview.SetTime(prescribed_uy);
   paraview.Save();

   return 0;
}

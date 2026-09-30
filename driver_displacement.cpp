// ============================================================================
// driver_displacement.cpp
//
// Hyperelastostatics in the displacement form, the displacement being the
// only unknown. Boundary conditions are read from config.yaml and refer to
// faces by name; the material is given by MaterialModelData.
//
// Author: Chongran Zhao
// Date: Sep. 28, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <filesystem>
#include <string>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "BoundaryManager.hpp"
#include "Integrator_Displacement.hpp"
#include "MaterialModelData.hpp"
#include "SystemTools.hpp"

int main(int argc, char *argv[])
{
   // 1. Read config.yaml.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path("config.yaml");
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Read the mesh file.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. Set up the finite element space of the displacement.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   mfem::H1_FECollection fec_u(order, dim);
   mfem::FiniteElementSpace space_u(&mesh, &fec_u, dim, mfem::Ordering::byVDIM);
   mfem::GridFunction disp(&space_u);
   disp = 0.0;
   SystemTools::print_space(space_u);

   // 4. Set up the boundary conditions.
   BoundaryManager boundaries(config, space_u);
   const int num_load_steps = boundaries.get_num_load_steps();
   boundaries.print_fixed_bc();
   boundaries.print_load();

   // 5. Set up the material model.
   const MaterialModel material = get_material_model();

   // 6. Construct the nonlinear form; its essential dofs are the constrained
   //    displacements.
   mfem::NonlinearForm nonlinear_form(&space_u);
   nonlinear_form.AddDomainIntegrator(new Integrator_Displacement(material));

   const mfem::Array<int> ess_u = boundaries.get_ess_tdof_list();
   nonlinear_form.SetEssentialTrueDofs(ess_u);

   // Internal force R^a_k = int N_a,J P_kJ dV at every node, from a second
   // form without essential dofs: the reaction on the constrained dofs, the
   // load on the others.
   mfem::NonlinearForm internal_force_form(&space_u);
   internal_force_form.AddDomainIntegrator(new Integrator_Displacement(material));
   mfem::GridFunction internal_force(&space_u);

   // 7. Set up the linear solver and Newton's method.
   const YAML::Node solver = config["solver"];
   mfem::UMFPackSolver linear_solver;

   mfem::NewtonSolver newton_solver;
   newton_solver.SetOperator(nonlinear_form);
   newton_solver.SetSolver(linear_solver);
   newton_solver.SetRelTol(solver["newton_rel_tol"].as<double>());
   newton_solver.SetAbsTol(solver["newton_abs_tol"].as<double>());
   newton_solver.SetMaxIter(solver["newton_max_iter"].as<int>());
   newton_solver.SetPrintLevel(-1);
   newton_solver.iterative_mode = true;

   // Monitor the Newton iterations.
   SystemTools::NewtonMonitor newton_monitor;
   newton_solver.SetMonitor(newton_monitor);

   // Remove the former results.
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   SystemTools::make_empty_dir(results_dir);

   SystemTools::save_gf(results_dir, "disp", 0, disp);
   internal_force_form.Mult(disp, internal_force);
   SystemTools::save_gf(results_dir, "internal_force", 0, internal_force);

   // 8. Set up the external force.
   mfem::LinearForm external_force(&space_u);
   external_force = 0.0;
   if (boundaries.is_traction_load())
      boundaries.add_traction_integrators(external_force);

   // Set zero displacement on the fixed faces.
   boundaries.apply_fixed_bc(disp);

   // Work vectors of the consistent predictor.
   mfem::GridFunction disp_target(&space_u);
   mfem::Vector prescribed_increment(space_u.GetTrueVSize()), coupling(space_u.GetTrueVSize());
   mfem::Vector predictor_rhs(space_u.GetTrueVSize()), predicted_increment(space_u.GetTrueVSize());

   // Loading loop.
   for (int step = 1; step <= num_load_steps; step++)
   {
      if (boundaries.is_traction_load())
      {
         boundaries.update_traction(step);
         external_force.Assemble();
      }

      // Consistent predictor: the initial guess of Newton's method is one
      // linear step from the converged state u with the new load, so that the
      // interior follows the prescribed boundary increment du_e instead of
      // only the boundary nodes moving,
      //    K_ff du_f = [f_ext - R_int(u)]_f - K_fe du_e,
      // with the full tangent of internal_force_form for K_fe.
      disp_target = disp;
      if (boundaries.is_disp_load())
         boundaries.apply_disp_load_bc(step, disp_target);
      prescribed_increment = disp_target;
      prescribed_increment -= disp;

      internal_force_form.Mult(disp, predictor_rhs);
      internal_force_form.GetGradient(disp).Mult(prescribed_increment, coupling);
      predictor_rhs.Neg();
      predictor_rhs += external_force;
      predictor_rhs -= coupling;
      // The tangent of nonlinear_form is the identity on the constrained dofs,
      // so their increment is du_e itself.
      for (int dof : ess_u)
         predictor_rhs(dof) = prescribed_increment(dof);

      linear_solver.SetOperator(nonlinear_form.GetGradient(disp));
      linear_solver.Mult(predictor_rhs, predicted_increment);
      disp += predicted_increment;
      // Set the prescribed values exactly, free of round-off.
      if (boundaries.is_disp_load())
         boundaries.apply_disp_load_bc(step, disp);

      boundaries.print_load_by_step(step, disp, external_force);
      SystemTools::print_newton_header();

      // Set the external force to zero on the essential dofs.
      external_force.SetSubVector(ess_u, 0.0);
      // Newton iterations for the displacement.
      newton_solver.Mult(external_force, disp);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at step " << step << ".");

      mfem::out << "converged in " << newton_solver.GetNumIterations() << " iterations\n";

      SystemTools::save_gf(results_dir, "disp", step, disp);
      internal_force_form.Mult(disp, internal_force);
      SystemTools::save_gf(results_dir, "internal_force", step, internal_force);
   }

   mfem::out << std::string(74, '=') << "\n\n";
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

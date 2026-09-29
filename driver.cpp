// ============================================================================
// driver.cpp
//
// Hyperelastostatics on the labelled mesh written by read_mesh. Boundary
// conditions are read from config.yaml and refer to faces by name.
//
// Author: Chongran Zhao
// Date: Sep. 28, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
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
                 : std::filesystem::path(SOURCE_DIR) / "config.yaml";
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Read the mesh file.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. Set up the finite element space.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   mfem::H1_FECollection fec(order, dim);
   mfem::FiniteElementSpace fespace(&mesh, &fec, dim, mfem::Ordering::byVDIM);
   mfem::GridFunction disp(&fespace);
   disp = 0.0;
   SystemTools::print_space(fespace);

   // 4. Set up the boundary conditions.
   BoundaryManager boundaries(config, fespace);
   const int num_load_steps = boundaries.get_num_load_steps();
   boundaries.print_fixed_bc();
   boundaries.print_load();

   // 5. Set up the material model.
   const MaterialModel material = get_material_model();

   // 6. Construct the nonlinear form of the internal force.
   mfem::NonlinearForm nonlinear_form(&fespace);
   nonlinear_form.AddDomainIntegrator(new Integrator_Displacement(material));
   nonlinear_form.SetEssentialTrueDofs(boundaries.get_ess_tdof_list());

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

   // 8. Set up the external force.
   mfem::LinearForm external_force(&fespace);
   external_force = 0.0;
   if (boundaries.is_traction_load())
      boundaries.add_traction_integrators(external_force);

   // Set zero displacement on the fixed faces.
   boundaries.apply_fixed_bc(disp);

   // Loading loop.
   for (int step = 1; step <= num_load_steps; step++)
   {
      if (boundaries.is_disp_load())
         boundaries.apply_disp_load_bc(step, disp);

      if (boundaries.is_traction_load())
      {
         boundaries.update_traction(step);
         external_force.Assemble();
      }

      boundaries.print_load_by_step(step, disp, external_force);
      SystemTools::print_newton_header();

      // Set the external force to zero on the essential dofs.
      external_force.SetSubVector(boundaries.get_ess_tdof_list(), 0.0);
      // Newton iterations for the displacement.
      newton_solver.Mult(external_force, disp);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at step " << step << ".");

      mfem::out << "converged in " << newton_solver.GetNumIterations() << " iterations\n";

      SystemTools::save_gf(results_dir, "disp", step, disp);
   }

   mfem::out << std::string(74, '=') << "\n\n";
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

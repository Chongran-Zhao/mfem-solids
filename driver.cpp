// ============================================================================
// driver.cpp
//
// Hyperelastostatics on the labelled mesh written by read_mesh. Boundary
// conditions are read from config.yaml and refer to faces by name.
// Step 1: read config.yaml.
// Step 2: read the labelled mesh written by read_mesh.
// Step 3: create the displacement finite element space.
// Step 4: read the Dirichlet conditions.
// Step 5: define the material.
// Step 6: build the nonlinear form and Newton's method.
// Step 7: raise the prescribed displacements over the load steps and save
//         the displacement of each step into results_gf, which
//         write_paraview and write_traction_disp read.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include "CompressibleHyperelasticIntegrator.hpp"
#include "DirichletBoundary.hpp"
#include "MaterialModel.hpp"
#include "SystemTools.hpp"
#include "mfem.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

int main(int argc, char *argv[])
{
   // 1. By default the config.yaml next to this source file is read.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path(SOURCE_DIR) / "config.yaml";
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. read_mesh writes the labelled mesh into the directory it runs in, so
   //    the driver runs in the same directory, normally build/.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. Continuous shape functions of the given order, with three components
   //    per node.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   mfem::H1_FECollection fec(order, dim);
   mfem::FiniteElementSpace fespace(&mesh, &fec, dim, mfem::Ordering::byVDIM);
   mfem::GridFunction disp(&fespace);
   disp = 0.0;
   SystemTools::print_space(fespace);

   // 4. The Dirichlet conditions of config.yaml: the constrained dofs and
   //    the final boundary displacement, see include/DirichletBoundary.hpp.
   const DirichletBoundary dirichlet(config["Dirichlet"], fespace);
   dirichlet.print_Dirichlet_bc();

   // 5. The material is given in include/MaterialModel.hpp, so that
   //    write_paraview and write_traction_disp use the same one.
   const MaterialModel material = get_material_model();

   // 6. The nonlinear form owns the integrator; the material must outlive it.
   //    Newton's method solves R(d) = 0 from the current disp, with the
   //    direct solver UMFPACK (SuiteSparse) for the linear systems.
   mfem::NonlinearForm nonlinear_form(&fespace);
   nonlinear_form.AddDomainIntegrator(new CompressibleHyperelasticIntegrator(material));
   nonlinear_form.SetEssentialTrueDofs(dirichlet.get_ess_tdof_list());

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

   // The Newton iterations are printed by SystemTools::NewtonMonitor.
   SystemTools::NewtonMonitor newton_monitor;
   newton_solver.SetMonitor(newton_monitor);

   // 7. At load step n of N the boundary displacement is n / N times its
   //    final value; the other nodes keep the previous solution, which is the
   //    initial guess of Newton's method.
   const int load_steps = dirichlet.get_num_load_steps();

   // The displacement of each step as an MFEM grid function,
   // <results>/disp_XXXX.gf, step 0 included, in an emptied folder.
   const std::filesystem::path results_dir = config["output"]["results"].as<std::string>();
   SystemTools::make_empty_dir(results_dir);

   SystemTools::save_gf(results_dir, "disp", 0, disp);

   const mfem::Vector zero_rhs;
   for (int step = 1; step <= load_steps; step++)
   {
      dirichlet.apply_fixed_bc(disp);
      if (dirichlet.get_is_disp_load())
         dirichlet.apply_disp_load_bc(step, disp);

      if (dirichlet.get_is_disp_load())
         dirichlet.print_disp_load_by_step(step);
      SystemTools::print_newton_header();

      newton_solver.Mult(zero_rhs, disp);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at step " << step << ".");

      mfem::out << "converged in " << newton_solver.GetNumIterations() << " iterations\n";

      SystemTools::save_gf(results_dir, "disp", step, disp);
   }

   mfem::out << std::string(74, '=') << "\n\n";
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

// ============================================================================
// driver_static_disp.cpp
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
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "BoundaryManager.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "LocalAssemblyTools.hpp"
#include "LocalAssembly_Disp.hpp"
#include "MaterialModelData.hpp"
#include "NonlinearSolver_Static_Disp.hpp"
#include "SystemTools.hpp"

int main(int argc, char *argv[])
{
   // Wall-clock time of the whole run.
   mfem::StopWatch total_timer;
   total_timer.Start();
   mfem::out << "\nJob started on " << SystemTools::get_time() << ' '
             << SystemTools::get_date() << '\n';

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
   auto boundary_manager = std::make_unique<BoundaryManager>(config, space_u);
   const int num_load_steps = boundary_manager->get_num_load_steps();
   boundary_manager->print_fixed_bc();
   boundary_manager->print_load();

   // 5. Set up the material model.
   std::unique_ptr<MaterialModel> material = get_material_model();

   // 6. Set up the assembly: the material goes to the local assembly, and the
   //    local assembly and the boundary conditions to the global one, which
   //    owns them.
   auto local_assembly = std::make_unique<LocalAssembly_Disp>(std::move(material));
   auto global_assembly = std::make_unique<GlobalAssembly_Disp>(
      space_u, std::move(local_assembly), std::move(boundary_manager));

   // The material of the stress output; it holds no state, so the output
   // creates its own.
   const std::unique_ptr<const MaterialModel> output_material = get_material_model();

   // Pressure p(J) and first Piola-Kirchhoff stress P at the element centers,
   // for vtu_writer: piecewise constant, P with the 9 components P_xx, P_xy,
   // ..., P_zz.
   mfem::L2_FECollection fec_center(0, dim);
   mfem::FiniteElementSpace space_pres(&mesh, &fec_center);
   mfem::FiniteElementSpace space_stress(&mesh, &fec_center, 9, mfem::Ordering::byVDIM);
   mfem::GridFunction pres(&space_pres), stress(&space_stress);

   // 7. Set up the nonlinear solver, which owns the global assembly.
   auto nonlinear_solver = std::make_unique<NonlinearSolver_Static_Disp>(
      std::move(global_assembly), config["solver"]);

   // Remove the former results.
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   SystemTools::make_empty_dir(results_dir);

   // Saves the displacement, the pressure and the stress of a step.
   auto save_results = [&](int step)
   {
      SystemTools::save_gf(results_dir, "disp", step, disp);

      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const Tensor2_3D F = LocalAssemblyTools::get_center_deformation_gradient(space_u, disp, ee);
         pres(ee) = output_material->get_p(F.det());
         const Tensor2_3D PK1 = output_material->get_1st_PK_stress(F);
         for (int ii = 0; ii < 3; ii++)
            for (int JJ = 0; JJ < 3; JJ++)
               stress(space_stress.DofToVDof(ee, 3 * ii + JJ)) = PK1(ii, JJ);
      }
      SystemTools::save_gf(results_dir, "pres", step, pres);
      SystemTools::save_gf(results_dir, "stress", step, stress);
   };
   save_results(0);

   // 8. Loading loop.
   mfem::StopWatch step_timer;
   for (int step = 1; step <= num_load_steps; step++)
   {
      // Wall-clock time of the step, up to the convergence.
      step_timer.Restart();

      const int iterations = nonlinear_solver->solve(step, disp);

      mfem::out << "converged in " << iterations
                << " iterations. Time taken: " << std::fixed << std::setprecision(2)
                << step_timer.RealTime() << " sec. " << SystemTools::get_time()
                << std::defaultfloat << '\n';

      save_results(step);
   }

   mfem::out << std::string(74, '=') << "\n\n";
   mfem::out << "Job finished on " << SystemTools::get_time() << ' ' << SystemTools::get_date()
             << ". Time taken: " << std::fixed << std::setprecision(2) << total_timer.RealTime()
             << " sec.\n\n" << std::defaultfloat;
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

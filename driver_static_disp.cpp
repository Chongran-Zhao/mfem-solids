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

   // Borrowed from the global assembly.
   BoundaryManager &boundaries = global_assembly->get_boundaries();
   const mfem::Array<int> &ess_u = global_assembly->get_ess_tdof_list();

   // Internal force R^a_k = int N_a,J P_kJ dV at every node: the reaction on
   // the constrained dofs, the load on the others.
   mfem::GridFunction internal_force(&space_u);

   // Pressure p(J) and first Piola-Kirchhoff stress P at the element centers,
   // for vtu_writer: piecewise constant, P with the 9 components P_xx, P_xy,
   // ..., P_zz.
   mfem::L2_FECollection fec_center(0, dim);
   mfem::FiniteElementSpace space_pres(&mesh, &fec_center);
   mfem::FiniteElementSpace space_stress(&mesh, &fec_center, 9, mfem::Ordering::byVDIM);
   mfem::GridFunction pres(&space_pres), stress(&space_stress);

   // 7. Set up the linear solver and Newton's method.
   const YAML::Node solver = config["solver"];
   mfem::UMFPackSolver linear_solver;

   mfem::NewtonSolver newton_solver;
   newton_solver.SetOperator(*global_assembly);
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

   // Saves the displacement, the internal force, the pressure and the stress
   // of a step.
   auto save_results = [&](int step)
   {
      SystemTools::save_gf(results_dir, "disp", step, disp);

      global_assembly->get_internal_force(disp, internal_force);
      SystemTools::save_gf(results_dir, "internal_force", step, internal_force);

      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const Tensor2_3D F = LocalAssemblyTools::get_center_deformation_gradient(space_u, disp, ee);
         pres(ee) = global_assembly->get_material().get_p(F.det());
         const Tensor2_3D PK1 = global_assembly->get_material().get_1st_PK_stress(F);
         for (int ii = 0; ii < 3; ii++)
            for (int JJ = 0; JJ < 3; JJ++)
               stress(space_stress.DofToVDof(ee, 3 * ii + JJ)) = PK1(ii, JJ);
      }
      SystemTools::save_gf(results_dir, "pres", step, pres);
      SystemTools::save_gf(results_dir, "stress", step, stress);
   };
   save_results(0);

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
   mfem::StopWatch step_timer;
   for (int step = 1; step <= num_load_steps; step++)
   {
      // Wall-clock time of the step, up to the convergence.
      step_timer.Restart();

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
      // with the full stiffness for K_fe.
      disp_target = disp;
      if (boundaries.is_disp_load())
         boundaries.apply_disp_load_bc(step, disp_target);
      prescribed_increment = disp_target;
      prescribed_increment -= disp;

      global_assembly->get_internal_force(disp, predictor_rhs);
      global_assembly->get_stiffness(disp).Mult(prescribed_increment, coupling);
      predictor_rhs.Neg();
      predictor_rhs += external_force;
      predictor_rhs -= coupling;
      // The tangent of the global assembly is the identity on the constrained
      // dofs, so their increment is du_e itself.
      for (int dof : ess_u)
         predictor_rhs(dof) = prescribed_increment(dof);

      linear_solver.SetOperator(global_assembly->GetGradient(disp));
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

      mfem::out << "converged in " << newton_solver.GetNumIterations()
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

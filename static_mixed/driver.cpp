// ============================================================================
// driver.cpp of static_mixed
//
// Hyperelastostatics in the mixed displacement-pressure (u/p) form, with
// Taylor-Hood elements. Boundary conditions are read from config.yaml and
// refer to faces by name; the material is given by MaterialModelData, whose
// volumetric model decides whether it is compressible or fully
// incompressible.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <filesystem>
#include <iomanip>
#include <memory>
#include <string>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "DirichletBoundary.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"
#include "VTK_Tools.hpp"

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

   // 3. Set up the finite element spaces, Taylor-Hood: the pressure one
   //    order below the displacement.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   MFEM_VERIFY(order >= 2, "The mixed driver needs space.order >= 2 for Taylor-Hood elements.");
   mfem::H1_FECollection fec_u(order, dim), fec_p(order - 1, dim);
   mfem::FiniteElementSpace space_u(&mesh, &fec_u, dim, mfem::Ordering::byVDIM);
   mfem::FiniteElementSpace space_p(&mesh, &fec_p);
   SystemTools::print_space(space_u);
   SystemTools::print_space(space_p);

   // Block 0 is the displacement, block 1 the pressure.
   mfem::Array<mfem::FiniteElementSpace *> spaces({&space_u, &space_p});
   mfem::Array<int> offsets({0, space_u.GetTrueVSize(),
                             space_u.GetTrueVSize() + space_p.GetTrueVSize()});
   mfem::BlockVector sol(offsets);
   sol = 0.0;

   // disp and pres are views of the two blocks of sol, not copies.
   mfem::GridFunction disp, pres;
   disp.MakeRef(&space_u, sol.GetBlock(0), 0);
   pres.MakeRef(&space_p, sol.GetBlock(1), 0);

   // 4. Set up the boundary conditions, on the displacement only.
   DirichletBoundary dirichlet(config["Dirichlet"], space_u);
   NeumannBoundary neumann(config["Neumann"], space_u);
   const int num_load_steps = config["loading"]["load_steps"].as<int>();

   // The loading is either a prescribed displacement or a traction.
   const std::string loading_type = config["loading"]["type"].as<std::string>();
   if (loading_type == "displacement")
   {
      MFEM_VERIFY(dirichlet.is_disp_load() && !neumann.is_traction_load(),
                  "The loading type is displacement, so disp_bc must have an "
                  "entry and Neumann no face.");
   }
   else if (loading_type == "traction")
   {
      MFEM_VERIFY(neumann.is_traction_load() && !dirichlet.is_disp_load(),
                  "The loading type is traction, so Neumann must have a face "
                  "and disp_bc no entry.");
   }
   else
      MFEM_ABORT("Unknown loading type \"" << loading_type << "\".");

   dirichlet.print_fixed_bc();
   if (dirichlet.is_disp_load())
      dirichlet.print_disp_load();
   else
      neumann.print_traction_load();

   // 5. Set up the material model.
   const std::unique_ptr<const MaterialModel> material = set_material_model();

   // 6. Construct the block nonlinear form; the pressure has no essential
   //    dofs.
   mfem::BlockNonlinearForm nonlinear_form(spaces);
   nonlinear_form.AddDomainIntegrator(new LocalAssembly_Mixed(*material));

   // Constrained dofs per block; the pressure has none, but MFEM needs a list
   // for every space.
   mfem::Array<int> ess_u = dirichlet.get_ess_tdof_list(), ess_p;
   mfem::Array<mfem::Array<int> *> ess({&ess_u, &ess_p});
   mfem::Array<mfem::Vector *> ess_rhs({nullptr, nullptr});
   nonlinear_form.SetEssentialTrueDofs(ess, ess_rhs);

   // Internal force R^a_k = int N_a,J (P_ich_kJ - p J F^-1_Jk) dV at every
   // node, from a second form without essential dofs, for the predictor.
   mfem::BlockNonlinearForm internal_force_form(spaces);
   internal_force_form.AddDomainIntegrator(new LocalAssembly_Mixed(*material));

   // First Piola-Kirchhoff stress P = P_ich - p J F^-T at the element
   // centers, for vtu_writer: piecewise constant, with the 9 components P_xx,
   // P_xy, ..., P_zz.
   mfem::L2_FECollection fec_stress(0, dim);
   mfem::FiniteElementSpace space_stress(&mesh, &fec_stress, 9, mfem::Ordering::byVDIM);
   mfem::GridFunction stress(&space_stress);

   // 7. Set up the linear solver and Newton's method.
   const YAML::Node solver = config["solver"];
   SystemTools::BlockUMFPackSolver linear_solver;

   mfem::NewtonSolver newton_solver;
   newton_solver.SetOperator(nonlinear_form);
   newton_solver.SetSolver(linear_solver);
   newton_solver.SetRelTol(solver["newton_rel_tol"].as<double>());
   newton_solver.SetAbsTol(solver["newton_abs_tol"].as<double>());
   newton_solver.SetMaxIter(solver["newton_max_iter"].as<int>());
   newton_solver.SetPrintLevel(-1);
   newton_solver.iterative_mode = true;

   // Monitor the Newton iterations, the displacement and the pressure blocks
   // apart.
   SystemTools::BlockNewtonMonitor newton_monitor(offsets);
   newton_solver.SetMonitor(newton_monitor);

   // Remove the former results.
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   SystemTools::make_empty_dir(results_dir);

   // Saves the displacement, the pressure and the stress of a step.
   auto save_results = [&](int step)
   {
      SystemTools::save_gf(results_dir, "disp", step, disp);
      SystemTools::save_gf(results_dir, "pres", step, pres);

      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const Tensor2_3D F = VTK_Tools::get_center_deformation_gradient(space_u, disp, ee);
         const double p = pres.GetValue(ee, mfem::Geometries.GetCenter(mesh.GetElementGeometry(ee)));
         const Tensor2_3D PK1 = material->get_1st_PK_stress_ich(F)
                                - p * F.det() * F.inverse().transpose();
         for (int ii = 0; ii < 3; ii++)
            for (int JJ = 0; JJ < 3; JJ++)
               stress(space_stress.DofToVDof(ee, 3 * ii + JJ)) = PK1(ii, JJ);
      }
      SystemTools::save_gf(results_dir, "stress", step, stress);
   };
   save_results(0);

   // 8. Set up the external force, on the displacement block; the pressure
   //    block of rhs stays zero.
   mfem::LinearForm external_force(&space_u);
   external_force = 0.0;
   if (neumann.is_traction_load())
      neumann.add_traction_integrators(external_force);

   mfem::BlockVector rhs(offsets);
   rhs = 0.0;

   // Set zero displacement on the fixed faces.
   dirichlet.apply_fixed_bc(disp);

   // Work vectors of the consistent predictor.
   mfem::GridFunction disp_target(&space_u);
   mfem::BlockVector prescribed_increment(offsets), coupling(offsets);
   mfem::BlockVector predictor_rhs(offsets), predicted_increment(offsets);

   // Loading loop.
   mfem::StopWatch step_timer;
   for (int step = 1; step <= num_load_steps; step++)
   {
      // Wall-clock time of the step, up to the convergence.
      step_timer.Restart();

      // Load factor t = n / N, the time of the loading functions.
      const double load_factor = static_cast<double>(step) / num_load_steps;

      if (neumann.is_traction_load())
      {
         neumann.set_time(load_factor);
         external_force.Assemble();
      }
      rhs.GetBlock(0) = external_force;

      // Consistent predictor: the initial guess of Newton's method is one
      // linear step from the converged state (u, p) with the new load, so
      // that the interior follows the prescribed boundary increment du_e
      // instead of only the boundary nodes moving,
      //    K_ff d(u, p)_f = [f_ext - R(u, p)]_f - K_fe du_e,
      // with the full tangent of internal_force_form for K_fe; the pressure
      // is predicted as well.
      disp_target = disp;
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(load_factor, disp_target);
      prescribed_increment = 0.0;
      prescribed_increment.GetBlock(0) = disp_target;
      prescribed_increment.GetBlock(0) -= disp;

      internal_force_form.Mult(sol, predictor_rhs);
      internal_force_form.GetGradient(sol).Mult(prescribed_increment, coupling);
      predictor_rhs.Neg();
      predictor_rhs += rhs;
      predictor_rhs -= coupling;
      // The tangent of nonlinear_form is the identity on the constrained dofs,
      // so their increment is du_e itself.
      for (int dof : ess_u)
         predictor_rhs.GetBlock(0)(dof) = prescribed_increment.GetBlock(0)(dof);

      linear_solver.SetOperator(nonlinear_form.GetGradient(sol));
      linear_solver.Mult(predictor_rhs, predicted_increment);
      sol += predicted_increment;
      // Set the prescribed values exactly, free of round-off.
      if (dirichlet.is_disp_load())
         dirichlet.apply_disp_load_bc(load_factor, disp);

      mfem::out << std::string(74, '=') << '\n'
                << "Load step " << step << " / " << num_load_steps << '\n';
      if (dirichlet.is_disp_load())
         dirichlet.print_disp_load_by_step(disp);
      else
         neumann.print_traction_load_by_step();
      SystemTools::print_block_newton_header();

      // Set the external force to zero on the essential dofs.
      external_force.SetSubVector(ess_u, 0.0);
      // Again, now zero on the constrained dofs.
      rhs.GetBlock(0) = external_force;
      // Newton iterations for the displacement and the pressure.
      newton_solver.Mult(rhs, sol);
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

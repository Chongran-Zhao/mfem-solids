// ============================================================================
// driver_mixed.cpp
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
#include <string>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "BoundaryManager.hpp"
#include "Integrator_Mixed.hpp"
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

   // 3. Set up the finite element spaces, Taylor-Hood: the pressure one
   //    order below the displacement.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   MFEM_VERIFY(order >= 2, "driver_mixed needs space.order >= 2 for Taylor-Hood elements.");
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
   BoundaryManager boundaries(config, space_u);
   const int num_load_steps = boundaries.get_num_load_steps();
   boundaries.print_fixed_bc();
   boundaries.print_load();

   // 5. Set up the material model.
   const MaterialModel material = get_material_model();

   // 6. Construct the block nonlinear form; the pressure has no essential
   //    dofs.
   mfem::BlockNonlinearForm nonlinear_form(spaces);
   nonlinear_form.AddDomainIntegrator(new Integrator_Mixed(material));

   // Constrained dofs per block; the pressure has none, but MFEM needs a list
   // for every space.
   mfem::Array<int> ess_u = boundaries.get_ess_tdof_list(), ess_p;
   mfem::Array<mfem::Array<int> *> ess({&ess_u, &ess_p});
   mfem::Array<mfem::Vector *> ess_rhs({nullptr, nullptr});
   nonlinear_form.SetEssentialTrueDofs(ess, ess_rhs);

   // Internal force R^a_k = int N_a,J (P_ich_kJ - p J F^-1_Jk) dV at every
   // node, the displacement block of a second form without essential dofs:
   // the reaction on the constrained dofs, the load on the others.
   mfem::BlockNonlinearForm internal_force_form(spaces);
   internal_force_form.AddDomainIntegrator(new Integrator_Mixed(material));
   mfem::BlockVector internal_force_blocks(offsets);
   mfem::GridFunction internal_force(&space_u);

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

   SystemTools::save_gf(results_dir, "disp", 0, disp);
   SystemTools::save_gf(results_dir, "pres", 0, pres);
   internal_force_form.Mult(sol, internal_force_blocks);
   internal_force = internal_force_blocks.GetBlock(0);
   SystemTools::save_gf(results_dir, "internal_force", 0, internal_force);

   // 8. Set up the external force, on the displacement block; the pressure
   //    block of rhs stays zero.
   mfem::LinearForm external_force(&space_u);
   external_force = 0.0;
   if (boundaries.is_traction_load())
      boundaries.add_traction_integrators(external_force);

   mfem::BlockVector rhs(offsets);
   rhs = 0.0;

   // Set zero displacement on the fixed faces.
   boundaries.apply_fixed_bc(disp);

   // Work vectors of the consistent predictor.
   mfem::GridFunction disp_target(&space_u);
   mfem::BlockVector prescribed_increment(offsets), coupling(offsets);
   mfem::BlockVector predictor_rhs(offsets), predicted_increment(offsets);

   // Loading loop.
   for (int step = 1; step <= num_load_steps; step++)
   {
      if (boundaries.is_traction_load())
      {
         boundaries.update_traction(step);
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
      if (boundaries.is_disp_load())
         boundaries.apply_disp_load_bc(step, disp_target);
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
      if (boundaries.is_disp_load())
         boundaries.apply_disp_load_bc(step, disp);

      boundaries.print_load_by_step(step, disp, external_force);
      SystemTools::print_block_newton_header();

      // Set the external force to zero on the essential dofs.
      external_force.SetSubVector(ess_u, 0.0);
      // Again, now zero on the constrained dofs.
      rhs.GetBlock(0) = external_force;
      // Newton iterations for the displacement and the pressure.
      newton_solver.Mult(rhs, sol);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at step " << step << ".");

      mfem::out << "converged in " << newton_solver.GetNumIterations() << " iterations\n";

      SystemTools::save_gf(results_dir, "disp", step, disp);
      SystemTools::save_gf(results_dir, "pres", step, pres);
      internal_force_form.Mult(sol, internal_force_blocks);
      internal_force = internal_force_blocks.GetBlock(0);
      SystemTools::save_gf(results_dir, "internal_force", step, internal_force);
   }

   mfem::out << std::string(74, '=') << "\n\n";
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

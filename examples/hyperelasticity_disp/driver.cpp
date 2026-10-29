// ============================================================================
// driver.cpp of hyperelasticity_disp
//
// Hyperelasticity in the displacement form, static or dynamic by the type of
// time_method in config.yaml: static, without inertia, or gen_alpha, the
// generalized-alpha method in physical time. The time steps and the boundary
// conditions are read from config.yaml, the latter referring to faces by
// name; the material is given by MaterialModelData, and the initial velocity
// of the dynamics and the loading by LoadData. The initial displacement is
// zero.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <filesystem>
#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "LoadData.hpp"
#include "LocalAssembly_Disp.hpp"
#include "MaterialModel.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_Dynamic.hpp"
#include "NonlinearSolver_Static.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha_2nd_Order.hpp"
#include "TimeSolver.hpp"
#include "Vector_3D.hpp"

int main(int argc, char *argv[])
{
   // Start MPI and hypre; only rank 0 prints.
   mfem::Mpi::Init(argc, argv);
   mfem::Hypre::Init();
   if (!mfem::Mpi::Root())
      mfem::out.Disable();

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

   // 2. Read the mesh file on every rank and split it among the ranks; the
   //    serial mesh is only needed for the split.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh serial_mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, serial_mesh);
   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   // 3. Set up the finite element space of the displacement, also that of
   //    the velocity and the acceleration.
   const int dim = mesh.Dimension();
   const int elem_order = config["space"]["elem_order"].as<int>();
   mfem::H1_FECollection fec_u(elem_order, dim);
   mfem::ParFiniteElementSpace space_u(&mesh, &fec_u, dim, mfem::Ordering::byVDIM);
   SystemTools::print_space(space_u);

   // 4. Set the initial state: zero displacement, and in the dynamics the
   //    velocity LoadData::initial_velo at the nodes.
   const std::string time_method_type = config["time_method"]["type"].as<std::string>();
   MFEM_VERIFY(time_method_type == "static" || time_method_type == "gen_alpha",
               "Unknown time_method type \"" << time_method_type << "\".");
   const bool is_dynamic = time_method_type == "gen_alpha";
   mfem::VectorFunctionCoefficient initial_velo_value(dim,
      [](const mfem::Vector &pt, mfem::Vector &value)
   {
      const Vector_3D velo_0 = LoadData::initial_velo(pt);
      for (int comp = 0; comp < 3; comp++)
         value(comp) = velo_0(comp);
   });
   mfem::ParGridFunction disp(&space_u), velo(&space_u), acce(&space_u);
   disp = 0.0;
   velo = 0.0;
   if (is_dynamic)
      velo.ProjectCoefficient(initial_velo_value);
   acce = 0.0;

   // 5. Set up the boundary conditions; a prescribed displacement and a
   //    traction may act together.
   auto dirichlet = std::make_unique<DirichletBoundary>(config["Dirichlet"], space_u);
   auto neumann = std::make_unique<NeumannBoundary>(config["Neumann"], space_u,
                                                    LoadData::surface_traction);
   dirichlet->print_fixed_bc();
   if (dirichlet->is_disp_load())
      dirichlet->print_disp_load();
   if (neumann->is_traction_load())
      neumann->print_traction_load();

   // 6. Set up the material model.
   std::unique_ptr<MaterialModel> material = set_material_model();

   // 7. Set up the assembly: the material goes to the local assembly, and the
   //    local assembly and the boundary conditions to the global one, which
   //    owns them.
   auto local_assembly = std::make_unique<LocalAssembly_Disp>(std::move(material));
   local_assembly->set_quad_order(config["space"]["quad_order"].as<int>());
   auto global_assembly = std::make_unique<GlobalAssembly_Disp>(
      space_u, std::move(local_assembly), std::move(dirichlet), std::move(neumann));

   // 8. Set up the nonlinear solver of the time method, which owns the global
   //    assembly and, in the dynamics, the generalized-alpha parameters, and
   //    the time solver, which owns the nonlinear solver.
   std::unique_ptr<NonlinearSolver> nonlinear_solver;
   if (is_dynamic)
   {
      auto time_method = std::make_unique<TimeMethod_GenAlpha_2nd_Order>(
         config["time_method"]["rho_inf"].as<double>());
      nonlinear_solver = std::make_unique<NonlinearSolver_Dynamic>(
         std::move(global_assembly), std::move(time_method), config["solver"]);
   }
   else
      nonlinear_solver = std::make_unique<NonlinearSolver_Static>(
         std::move(global_assembly), config["solver"]);
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   auto time_solver = std::make_unique<TimeSolver>(
      std::move(nonlinear_solver), TimeSteps(config["time"]), results_dir, mesh,
      config["output"]["gf_interval"].as<int>());

   // 9. Solve the time steps.
   time_solver->run(disp, velo, acce);

   mfem::out << std::string(74, '=') << "\n\n";
   mfem::out << "Job finished on " << SystemTools::get_time() << ' ' << SystemTools::get_date()
             << ". Time taken: " << std::fixed << std::setprecision(2) << total_timer.RealTime()
             << " sec.\n\n" << std::defaultfloat << std::setprecision(6);
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

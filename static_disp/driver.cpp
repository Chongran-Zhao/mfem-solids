// ============================================================================
// driver.cpp of static_disp
//
// Hyperelastostatics in the displacement form, the displacement being the
// only unknown. Boundary conditions are read from config.yaml and refer to
// faces by name; the material is given by MaterialModelData.
// It runs in parallel, e.g. mpirun -np 4 ./driver: the mesh is split among
// the MPI ranks, and the linear solver, MUMPS or MINRES with BoomerAMG, is
// chosen in config.yaml.
//
// Author: Chongran Zhao
// Date: Sep. 28, 2026
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
#include "LocalAssembly_Disp.hpp"
#include "MaterialModel.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_Static_Disp.hpp"
#include "SystemTools.hpp"
#include "TimeSolver_Static_Disp.hpp"

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

   // 2. Read the mesh file on every rank and split it among the ranks.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 3. Set up the finite element space of the displacement.
   const int dim = pmesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   mfem::H1_FECollection fec_u(order, dim);
   mfem::ParFiniteElementSpace space_u(&pmesh, &fec_u, dim, mfem::Ordering::byVDIM);
   mfem::ParGridFunction disp(&space_u);
   disp = 0.0;
   SystemTools::print_space(space_u);

   // 4. Set up the boundary conditions.
   auto dirichlet = std::make_unique<DirichletBoundary>(config["Dirichlet"], space_u);
   auto neumann = std::make_unique<NeumannBoundary>(config["Neumann"], space_u);

   // The loading is either a prescribed displacement or a traction.
   const std::string loading_type = config["loading"]["type"].as<std::string>();
   if (loading_type == "displacement")
   {
      MFEM_VERIFY(dirichlet->is_disp_load() && !neumann->is_traction_load(),
                  "The loading type is displacement, so disp_bc must have an "
                  "entry and Neumann no face.");
   }
   else if (loading_type == "traction")
   {
      MFEM_VERIFY(neumann->is_traction_load() && !dirichlet->is_disp_load(),
                  "The loading type is traction, so Neumann must have a face "
                  "and disp_bc no entry.");
   }
   else
      MFEM_ABORT("Unknown loading type \"" << loading_type << "\".");

   dirichlet->print_fixed_bc();
   if (dirichlet->is_disp_load())
      dirichlet->print_disp_load();
   else
      neumann->print_traction_load();

   // 5. Set up the material model.
   std::unique_ptr<MaterialModel> material = set_material_model();

   // 6. Set up the assembly: the material goes to the local assembly, and the
   //    local assembly and the boundary conditions to the global one, which
   //    owns them.
   auto local_assembly = std::make_unique<LocalAssembly_Disp>(std::move(material));
   auto global_assembly = std::make_unique<GlobalAssembly_Disp>(
      space_u, std::move(local_assembly), std::move(dirichlet), std::move(neumann));

   // 7. Set up the nonlinear solver, which owns the global assembly, and the
   //    time solver, which owns the nonlinear solver.
   auto nonlinear_solver = std::make_unique<NonlinearSolver_Static_Disp>(
      std::move(global_assembly), config["solver"]);
   const int num_load_steps = config["loading"]["load_steps"].as<int>();
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   auto time_solver = std::make_unique<TimeSolver_Static_Disp>(
      std::move(nonlinear_solver), num_load_steps, results_dir, pmesh);

   // 8. Solve the load steps.
   time_solver->run(disp);

   mfem::out << std::string(74, '=') << "\n\n";
   mfem::out << "Job finished on " << SystemTools::get_time() << ' ' << SystemTools::get_date()
             << ". Time taken: " << std::fixed << std::setprecision(2) << total_timer.RealTime()
             << " sec.\n\n" << std::defaultfloat << std::setprecision(6);
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

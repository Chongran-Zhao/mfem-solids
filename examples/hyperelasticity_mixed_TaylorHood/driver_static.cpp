// ============================================================================
// driver_static.cpp of hyperelasticity_mixed_TaylorHood
//
// Hyperelastostatics in the mixed displacement-pressure (u/p) form, with
// Taylor-Hood elements, without inertia: the load steps are the time steps of
// config_static.yaml, the loading functions of LoadData receiving their
// times, the load factors with final_time 1. The boundary conditions are read
// from config_static.yaml and refer to faces by name; the material is given
// by MaterialModelData, whose volumetric model decides whether it is
// compressible or fully incompressible. The initial displacement and pressure
// are zero.
//
// Author: Chongran Zhao
// Date: Sep. 29, 2026
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
#include "GlobalAssembly_Mixed.hpp"
#include "LoadData.hpp"
#include "LocalAssembly_Mixed_TaylorHood.hpp"
#include "MaterialModel.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_Static.hpp"
#include "SystemTools.hpp"
#include "TimeSolver.hpp"

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

   // 1. Read the input, the file given as first argument, else
   //    config_static.yaml.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path("config_static.yaml");
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Read the mesh file on every rank and split it among the ranks; the
   //    serial mesh is only needed for the split.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh serial_mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, serial_mesh);
   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   // 3. Set up the finite element spaces, Taylor-Hood: the pressure one
   //    order below the displacement.
   const int dim = mesh.Dimension();
   const int elem_order = config["space"]["elem_order"].as<int>();
   MFEM_VERIFY(elem_order >= 2, "The mixed driver needs space.elem_order >= 2 for Taylor-Hood elements.");
   mfem::H1_FECollection fec_u(elem_order, dim), fec_p(elem_order - 1, dim);
   mfem::ParFiniteElementSpace space_u(&mesh, &fec_u, dim, mfem::Ordering::byVDIM);
   mfem::ParFiniteElementSpace space_p(&mesh, &fec_p);
   SystemTools::print_space(space_u);
   SystemTools::print_space(space_p);

   // 4. Set the initial state, zero displacement and pressure; the velocity
   //    and the acceleration, which the time solver takes, stay zero.
   mfem::ParGridFunction disp(&space_u), velo(&space_u), acce(&space_u), pres(&space_p);
   disp = 0.0;
   velo = 0.0;
   acce = 0.0;
   pres = 0.0;

   // 5. Set up the boundary conditions, on the displacement only; a
   //    prescribed displacement and a traction may act together.
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
   auto local_assembly = std::make_unique<LocalAssembly_Mixed_TaylorHood>(std::move(material));
   local_assembly->set_quad_order(config["space"]["quad_order"].as<int>());
   auto global_assembly = std::make_unique<GlobalAssembly_Mixed>(
      space_u, space_p, std::move(local_assembly), std::move(dirichlet), std::move(neumann));

   // 8. Set up the nonlinear solver, which owns the global assembly, and the
   //    time solver, which owns the nonlinear solver.
   auto nonlinear_solver = std::make_unique<NonlinearSolver_Static>(
      space_p, std::move(global_assembly), config["solver"]);
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   auto time_solver = std::make_unique<TimeSolver>(
      std::move(nonlinear_solver), TimeSteps(config["time"]), results_dir, mesh,
      config["output"]["gf_interval"].as<int>());

   // 9. Solve the load steps.
   time_solver->run(disp, velo, acce, pres);

   mfem::out << std::string(74, '=') << "\n\n";
   mfem::out << "Job finished on " << SystemTools::get_time() << ' ' << SystemTools::get_date()
             << ". Time taken: " << std::fixed << std::setprecision(2) << total_timer.RealTime()
             << " sec.\n\n" << std::defaultfloat << std::setprecision(6);
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

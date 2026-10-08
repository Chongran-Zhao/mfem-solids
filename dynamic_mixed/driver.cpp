// ============================================================================
// driver.cpp of dynamic_mixed
//
// Hyperelastodynamics in the mixed displacement-pressure (u/p) form, with
// Taylor-Hood elements and the generalized-alpha method in physical time. The
// time steps and the boundary conditions are read from config.yaml, the
// latter referring to faces by name; the material is given by
// MaterialModelData, whose volumetric model decides whether it is
// compressible or fully incompressible, and the initial velocity and the
// loading by LoadData. The initial displacement and pressure are zero.
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
#include "GlobalAssembly_Mixed.hpp"
#include "LoadData.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "MaterialModel.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_Dynamic_Mixed.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha.hpp"
#include "TimeSolver_Dynamic_Mixed.hpp"
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

   // 3. Set up the finite element spaces, Taylor-Hood: the pressure one
   //    order below the displacement, whose space is also that of the
   //    velocity and the acceleration.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   MFEM_VERIFY(order >= 2, "The mixed driver needs space.order >= 2 for Taylor-Hood elements.");
   mfem::H1_FECollection fec_u(order, dim), fec_p(order - 1, dim);
   mfem::ParFiniteElementSpace space_u(&mesh, &fec_u, dim, mfem::Ordering::byVDIM);
   mfem::ParFiniteElementSpace space_p(&mesh, &fec_p);
   SystemTools::print_space(space_u);
   SystemTools::print_space(space_p);

   // 4. Set the initial state: zero displacement and pressure, and the
   //    velocity LoadData::initial_velo at the nodes.
   mfem::VectorFunctionCoefficient initial_velo_value(dim,
      [](const mfem::Vector &pt, mfem::Vector &value)
   {
      const Vector_3D velo_0 = LoadData::initial_velo(pt);
      for (int comp = 0; comp < 3; comp++)
         value(comp) = velo_0(comp);
   });
   mfem::ParGridFunction disp(&space_u), velo(&space_u), acce(&space_u), pres(&space_p);
   disp = 0.0;
   velo.ProjectCoefficient(initial_velo_value);
   acce = 0.0;
   pres = 0.0;

   // 5. Set up the boundary conditions, on the displacement only; a
   //    prescribed displacement and a traction may act together.
   auto dirichlet = std::make_unique<DirichletBoundary>(config["Dirichlet"], space_u);
   auto neumann = std::make_unique<NeumannBoundary>(config["Neumann"], space_u);
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
   auto local_assembly = std::make_unique<LocalAssembly_Mixed>(std::move(material));
   auto global_assembly = std::make_unique<GlobalAssembly_Mixed>(
      space_u, space_p, std::move(local_assembly), std::move(dirichlet), std::move(neumann));

   // 8. Set up the nonlinear solver, which owns the global assembly and the
   //    time method, and the time solver, which owns the nonlinear solver.
   auto time_method =
      std::make_unique<TimeMethod_GenAlpha>(config["time_method"]["rho_inf"].as<double>());
   auto nonlinear_solver = std::make_unique<NonlinearSolver_Dynamic_Mixed>(
      std::move(global_assembly), std::move(time_method), config["solver"]);
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   auto time_solver = std::make_unique<TimeSolver_Dynamic_Mixed>(
      std::move(nonlinear_solver), config["time"]["dt"].as<double>(),
      config["time"]["final_time"].as<double>(), results_dir, mesh);

   // 9. Solve the time steps.
   time_solver->run(disp, velo, acce, pres);

   mfem::out << std::string(74, '=') << "\n\n";
   mfem::out << "Job finished on " << SystemTools::get_time() << ' ' << SystemTools::get_date()
             << ". Time taken: " << std::fixed << std::setprecision(2) << total_timer.RealTime()
             << " sec.\n\n" << std::defaultfloat << std::setprecision(6);
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

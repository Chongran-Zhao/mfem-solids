// ============================================================================
// driver_TaylorHood_dynamic.cpp of brain_simon
//
// The driver_TaylorHood_dynamic of sphere_rotation for the brain of SIMon,
// driven by the rigid motion of its skull: the same steps, with the loading,
// the materials of the parts and the nonlinear solver of its include/, which
// solves the linear systems by GMRES.
//
// Author: Chongran Zhao
// Date: Nov. 25, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <array>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Mixed.hpp"
#include "LoadData.hpp"
#include "LocalAssembly_Mixed_TaylorHood_Regions.hpp"
#include "MaterialModel.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_TaylorHood_Dynamic.hpp"
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

   // 1. Read the input, the file given as first argument, else
   //    config_TaylorHood_dynamic.yaml.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path("config_TaylorHood_dynamic.yaml");
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // The motion of the skull, from the load curves, which must last from
   // t = 0 to final_time; the velocities and the center in the units of the
   // keyword file, mm, scaled by mesh.length_scale.
   const YAML::Node loading = config["loading"];
   const double length_scale = config["mesh"]["length_scale"].as<double>();
   std::array<double, 3> center = loading["center"].as<std::array<double, 3>>();
   for (double &coord : center)
      coord *= length_scale;
   LoadData::read_motion(loading["motion_file"].as<std::string>(),
                         loading["translation_curves"].as<std::array<int, 3>>(),
                         loading["rotation_curves"].as<std::array<int, 3>>(), length_scale,
                         center, loading["substeps"].as<int>());
   MFEM_VERIFY(LoadData::get_start_time() <= 0.0
               && config["time"]["final_time"].as<double>() <= LoadData::get_end_time(),
               "The motion of the skull, on [" << LoadData::get_start_time() << ", "
               << LoadData::get_end_time() << "] s, does not cover [0, time.final_time].");

   // 2. Read the mesh file on every rank and split it among the ranks; the
   //    serial mesh is only needed for the split.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh serial_mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, serial_mesh);
   check_region_names(serial_mesh);
   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   // 3. Set up the finite element spaces, Taylor-Hood: the pressure one
   //    order below the displacement, whose space is also that of the
   //    velocity and the acceleration.
   const int dim = mesh.Dimension();
   const int elem_order = config["space"]["elem_order"].as<int>();
   MFEM_VERIFY(elem_order >= 2, "The mixed driver needs space.elem_order >= 2 for Taylor-Hood elements.");
   mfem::H1_FECollection fec_u(elem_order, dim), fec_p(elem_order - 1, dim);
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
   auto neumann = std::make_unique<NeumannBoundary>(config["Neumann"], space_u,
                                                    LoadData::surface_traction);
   dirichlet->print_fixed_bc();
   if (dirichlet->is_disp_load())
      dirichlet->print_disp_load();
   if (neumann->is_traction_load())
      neumann->print_traction_load();

   // 6. Set up the material of each region, by element attribute.
   std::vector<std::unique_ptr<const MaterialModel>> materials;
   for (int attribute = 1; attribute <= static_cast<int>(region_materials.size()); attribute++)
      materials.push_back(set_material_model(attribute));

   // 7. Set up the assembly: the materials go to the local assembly, and the
   //    local assembly and the boundary conditions to the global one, which
   //    owns them.
   auto local_assembly = std::make_unique<LocalAssembly_Mixed_TaylorHood_Regions>(std::move(materials));
   local_assembly->set_quad_order(config["space"]["quad_order"].as<int>());
   auto global_assembly = std::make_unique<GlobalAssembly_Mixed>(
      space_u, space_p, std::move(local_assembly), std::move(dirichlet), std::move(neumann));

   // 8. Set up the nonlinear solver, which owns the global assembly and the
   //    time method, assembles the mass matrix with the density of each
   //    region and the pressure operators of its preconditioner, and the
   //    time solver, which owns the nonlinear solver.
   auto time_method =
      std::make_unique<TimeMethod_GenAlpha_2nd_Order>(config["time_method"]["rho_inf"].as<double>());
   auto nonlinear_solver = std::make_unique<NonlinearSolver_TaylorHood_Dynamic>(
      space_u, space_p, std::move(global_assembly), std::move(time_method), config["solver"]);
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   auto time_solver = std::make_unique<TimeSolver>(
      std::move(nonlinear_solver), TimeSteps(config["time"]), results_dir, mesh,
      config["output"]["gf_interval"].as<int>());

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

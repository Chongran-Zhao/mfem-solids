// ============================================================================
// driver.cpp of sphere_rotation_vms
//
// The driver of dynamic_mixed_vms for the gelatin ball in its skull shell of
// sphere_rotation_TaylorHood, spun up by its surface: the rate form with equal-order
// elements stabilized by VMS, the materials of the regions, and GMRES. The
// VMS term acts in the incompressible regions only; as in PERIGEE, the
// compressible ones, here the skull, have tau = 0.
//
// Author: Chongran Zhao
// Date: Oct. 15, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <cmath>
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
#include "LocalAssembly_Mixed_VMS_Regions.hpp"
#include "MaterialModel.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha_1st_Order.hpp"
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

   // The measured rotation of the surface, which must last to final_time.
   LoadData::read_rotation(config["loading"]["rotation_file"].as<std::string>());
   MFEM_VERIFY(config["time"]["final_time"].as<double>() <= LoadData::get_end_time(),
               "time.final_time is beyond the rotation data, which ends at "
               << LoadData::get_end_time() << " s.");

   // 2. Read the mesh file on every rank and split it among the ranks; the
   //    serial mesh is only needed for the split.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh serial_mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, serial_mesh);
   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   // 3. Set up the finite element spaces, of equal order: the pressure of
   //    the order of the displacement, whose space is also that of the
   //    velocity and the acceleration.
   const int dim = mesh.Dimension();
   const int elem_order = config["space"]["elem_order"].as<int>();
   mfem::H1_FECollection fec_u(elem_order, dim), fec_p(elem_order, dim);
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

   // 6. Set up the material, the shear modulus and the tau_factor of each
   //    region, by element attribute; tau_factor of config.yaml in the
   //    incompressible regions, 0 in the compressible ones.
   const double tau_factor = config["space"]["tau_factor"].as<double>();
   std::vector<std::unique_ptr<const MaterialModel>> materials;
   std::vector<double> shear_moduli, tau_factors;
   for (int attribute = 1; attribute <= static_cast<int>(region_materials.size()); attribute++)
   {
      const region_material &region = get_region_material(attribute);
      materials.push_back(set_material_model(attribute));
      shear_moduli.push_back(region.mu);
      tau_factors.push_back(std::isinf(region.kappa) ? tau_factor : 0.0);
   }

   // 7. Set up the assembly: the materials go to the local assembly, and the
   //    local assembly and the boundary conditions to the global one, which
   //    owns them. The nonlinear solver keeps a reference to the local
   //    assembly, to give it the rates of each step.
   auto local_assembly = std::make_unique<LocalAssembly_Mixed_VMS_Regions>(
      std::move(materials), shear_moduli, tau_factors);
   local_assembly->set_quad_order(config["space"]["quad_order"].as<int>());
   LocalAssembly_Mixed_VMS_Regions &local_assembly_ref = *local_assembly;
   auto global_assembly = std::make_unique<GlobalAssembly_Mixed>(
      space_u, space_p, std::move(local_assembly), std::move(dirichlet), std::move(neumann));

   // 8. Set up the nonlinear solver, which owns the global assembly and the
   //    time method, assembles the mass matrix with the density of each
   //    region and the pressure operators of its preconditioner, and the
   //    time solver, which owns the nonlinear solver.
   auto time_method =
      std::make_unique<TimeMethod_GenAlpha_1st_Order>(config["time_method"]["rho_inf"].as<double>());
   auto nonlinear_solver = std::make_unique<NonlinearSolver>(
      space_u, space_p, std::move(global_assembly), local_assembly_ref, std::move(time_method),
      config["solver"]);
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   auto time_solver = std::make_unique<TimeSolver>(
      std::move(nonlinear_solver), config["time"]["dt"].as<double>(),
      config["time"]["final_time"].as<double>(), results_dir, mesh,
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

// ============================================================================
// driver.cpp of elastic_fracture
//
// Quasi-static phase-field fracture by alternate minimization, following the
// notes Phase-field-fracture.pdf: the displacement problem with the phase
// field fixed, LocalAssembly_Disp_Regions with the energy factor g(alpha)
// where the material breaks, and the phase-field problem with the
// displacement fixed, LocalAssembly_Fracture_Regions, solved in turn at each
// load step of config.yaml. Each element attribute is a region with its own
// material, breakable or not, as in MaterialModelData. The case is the
// helical crack of Bourdin (2007, Section 3), described in config.yaml; the
// loading is given by LoadData. The initial displacement is zero and the
// initial phase field 1, the sound material, but 0 on the faces of the
// initial crack of config.yaml, if any.
//
// Author: Chongran Zhao
// Date: Nov. 8, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
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
#include "GlobalAssembly_Disp.hpp"
#include "GlobalAssembly_Fracture.hpp"
#include "ISurfaceEnergy.hpp"
#include "LoadData.hpp"
#include "LocalAssembly_Disp_Regions.hpp"
#include "LocalAssembly_Fracture_Regions.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_Disp.hpp"
#include "NonlinearSolver_Fracture.hpp"
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

   // 1. Read the input, the file given as first argument, else config.yaml.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path("config.yaml");
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Read the mesh file on every rank and split it among the ranks; the
   //    serial mesh is only needed for the split.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh serial_mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, serial_mesh);
   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   // 3. Set up the finite element spaces of the displacement and of the
   //    phase field, of the same order on the same mesh.
   const int dim = mesh.Dimension();
   const int elem_order = config["space"]["elem_order"].as<int>();
   const int quad_order = config["space"]["quad_order"].as<int>();
   mfem::H1_FECollection fec(elem_order, dim);
   mfem::ParFiniteElementSpace space_u(&mesh, &fec, dim, mfem::Ordering::byVDIM);
   mfem::ParFiniteElementSpace space_alpha(&mesh, &fec);
   SystemTools::print_space(space_u);
   SystemTools::print_space(space_alpha);

   // 4. Set the initial state: zero displacement, alpha = 1.
   mfem::ParGridFunction disp(&space_u), alpha(&space_alpha);
   disp = 0.0;
   alpha = 1.0;

   // 5. The degradation g(alpha) at the points, the energy factor of the
   //    displacement problem; it reads alpha at every assembly. alpha,
   //    surface_energy and the coefficients are declared before the
   //    solvers, so that they outlive them.
   const std::unique_ptr<ISurfaceEnergy> surface_energy = set_surface_energy();
   mfem::GridFunctionCoefficient alpha_coefficient(&alpha);
   mfem::TransformedCoefficient g_coefficient(
      &alpha_coefficient, [&surface_energy](double value) { return surface_energy->get_g(value); });

   // The body force of LoadData, on its region, at the time the displacement
   // problem sets; declared before the solvers, as the coefficients above.
   mfem::VectorFunctionCoefficient body_force_coefficient(
      3, [](const mfem::Vector &pt, double tt, mfem::Vector &value)
      {
         const Vector_3D body_force = LoadData::body_force(pt, tt);
         for (int comp = 0; comp < 3; comp++)
            value(comp) = body_force(comp);
      });

   // 6. Set up the boundary conditions of the displacement.
   auto dirichlet = std::make_unique<DirichletBoundary>(config["Dirichlet"], space_u);
   auto neumann = std::make_unique<NeumannBoundary>(config["Neumann"], space_u,
                                                    LoadData::surface_traction);
   dirichlet->print_fixed_bc();
   if (dirichlet->is_disp_load())
      dirichlet->print_disp_load();
   if (neumann->is_traction_load())
      neumann->print_traction_load();

   // 7. The displacement problem: the material of each region goes to the
   //    local assembly, with the energy factor g(alpha) where the region
   //    breaks; the local assembly and the boundary conditions go to the
   //    global one, which the nonlinear solver owns.
   const int num_regions = static_cast<int>(region_materials.size());
   std::vector<std::unique_ptr<const IMaterialModel>> materials_disp;
   for (int attribute = 1; attribute <= num_regions; attribute++)
      materials_disp.push_back(set_material_model(attribute));
   auto local_assembly_disp = std::make_unique<LocalAssembly_Disp_Regions>(std::move(materials_disp));
   local_assembly_disp->set_quad_order(quad_order);
   for (int attribute = 1; attribute <= num_regions; attribute++)
      if (get_region_material(attribute).breakable)
         local_assembly_disp->set_energy_factor(attribute, g_coefficient);
   auto global_assembly_disp = std::make_unique<GlobalAssembly_Disp>(
      space_u, std::move(local_assembly_disp), std::move(dirichlet), std::move(neumann));
   mfem::Array<int> body_force_marker(mesh.attributes.Max());
   body_force_marker = 0;
   body_force_marker[LoadData::body_force_region - 1] = 1;
   global_assembly_disp->set_body_force(body_force_coefficient, body_force_marker);
   auto solver_disp = std::make_unique<NonlinearSolver_Disp>(std::move(global_assembly_disp),
                                                             config["solver_disp"]);

   // 8. The phase-field problem: the surface energy, materials of its own,
   //    whether each region breaks and the displacement go to the local
   //    assembly, the local assembly to the global one, which the nonlinear
   //    solver owns.
   std::vector<std::unique_ptr<const IMaterialModel>> materials_fracture;
   std::vector<bool> breakable;
   for (int attribute = 1; attribute <= num_regions; attribute++)
   {
      materials_fracture.push_back(set_material_model(attribute));
      breakable.push_back(get_region_material(attribute).breakable);
   }
   auto local_assembly_fracture = std::make_unique<LocalAssembly_Fracture_Regions>(
      set_surface_energy(), std::move(materials_fracture), breakable, disp);
   local_assembly_fracture->set_quad_order(quad_order);
   auto global_assembly_fracture = std::make_unique<GlobalAssembly_Fracture>(
      space_alpha, std::move(local_assembly_fracture));
   auto solver_fracture = std::make_unique<NonlinearSolver_Fracture>(
      std::move(global_assembly_fracture), config["solver_fracture"]);

   // The initial crack: the dofs of the faces of fracture.initial_crack,
   // named in mesh.faces of config.yaml, where alpha = 0 at every step.
   mfem::Array<int> initial_crack;
   for (const YAML::Node &face : config["fracture"]["initial_crack"])
   {
      mfem::Array<int> face_marker =
         mesh.bdr_attribute_sets.GetAttributeSetMarker(face.as<std::string>());
      mfem::Array<int> face_dofs;
      space_alpha.GetEssentialTrueDofs(face_marker, face_dofs);
      initial_crack.Append(face_dofs);
   }
   const int crack_size = solver_fracture->set_initial_crack(initial_crack);
   mfem::out << "\ninitial crack: " << crack_size << " dofs with alpha = 0\n";

   // 9. The time solver, which owns the two nonlinear solvers.
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   auto time_solver = std::make_unique<TimeSolver>(
      std::move(solver_disp), std::move(solver_fracture), config["fracture"],
      TimeSteps(config["time"]), results_dir, mesh, config["output"]["gf_interval"].as<int>());

   // 10. Solve the load steps.
   time_solver->run(disp, alpha);

   mfem::out << std::string(74, '=') << "\n\n";
   mfem::out << "Job finished on " << SystemTools::get_time() << ' ' << SystemTools::get_date()
             << ". Time taken: " << std::fixed << std::setprecision(2) << total_timer.RealTime()
             << " sec.\n\n" << std::defaultfloat << std::setprecision(6);
   SystemTools::print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

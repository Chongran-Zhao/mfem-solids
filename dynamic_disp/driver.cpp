// Finite-strain displacement dynamics with generalized-alpha time integration.
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "LocalAssembly_Disp.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_Dynamic_Disp.hpp"
#include "SystemTools.hpp"
#include "TimeMethod_GenAlpha.hpp"
#include "TimeSolver_Dynamic_Disp.hpp"

int main(int argc, char *argv[])
{
   const auto yaml_file = argc > 1 ? std::filesystem::path(argv[1])
                                   : std::filesystem::path("config.yaml");
   const YAML::Node config = YAML::LoadFile(yaml_file.string());
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   MFEM_VERIFY(mesh.Dimension() == 3, "The solid material models require a 3D mesh.");
   SystemTools::print_mesh(mesh_file, mesh);
   const int order = config["space"]["order"].as<int>();
   MFEM_VERIFY(order >= 1, "The displacement order must be at least one.");
   mfem::H1_FECollection fec(order, 3);
   mfem::FiniteElementSpace space(&mesh, &fec, 3, mfem::Ordering::byVDIM);
   MFEM_VERIFY(space.GetVSize() == space.GetTrueVSize(),
               "The dynamic driver currently requires a conforming mesh.");
   SystemTools::print_space(space);
   mfem::GridFunction disp(&space), velo(&space), acce(&space);

   const std::string profile = config["initial"]["profile"].as<std::string>();
   MFEM_VERIFY(profile == "uniform" || profile == "linear_x", "Unknown initial profile.");
   mfem::Vector lo, hi;
   mesh.GetBoundingBox(lo, hi);
   for (const std::string field : {"displacement", "velocity"})
   {
      const YAML::Node components = config["initial"][field];
      MFEM_VERIFY(components.IsSequence() && components.size() == 3,
                  "Initial fields must have three components.");
      mfem::Vector amplitude(3);
      for (int i = 0; i < 3; ++i) amplitude(i) = components[i].as<double>();
      mfem::VectorFunctionCoefficient value(3,
         [&](const mfem::Vector &x, mfem::Vector &out)
         {
            out = amplitude;
            if (profile == "linear_x") out *= (x(0) - lo(0)) / (hi(0) - lo(0));
         });
      (field == "displacement" ? disp : velo).ProjectCoefficient(value);
   }
   acce = 0.0;

   auto dirichlet = std::make_unique<DirichletBoundary>(config["Dirichlet"], space);
   auto neumann = std::make_unique<NeumannBoundary>(config["Neumann"], space);
   dirichlet->print_fixed_bc();
   dirichlet->print_disp_load();
   neumann->print_traction_load();
   auto local_assembly = std::make_unique<LocalAssembly_Disp>(set_material_model());
   auto global_assembly = std::make_unique<GlobalAssembly_Disp>(
      space, std::move(local_assembly), std::move(dirichlet), std::move(neumann));
   const YAML::Node dynamics = config["dynamics"];
   auto time_method = std::make_unique<TimeMethod_GenAlpha>(dynamics["rho_inf"].as<double>());
   auto nonlinear_solver = std::make_unique<NonlinearSolver_Dynamic_Disp>(
      std::move(global_assembly), dynamics["density"].as<double>(), std::move(time_method), config["solver"]);
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   auto time_solver = std::make_unique<TimeSolver_Dynamic_Disp>(
      std::move(nonlinear_solver), dynamics["dt"].as<double>(),
      dynamics["final_time"].as<double>(), results_dir);
   time_solver->run(disp, velo, acce);
   SystemTools::print_saved(results_dir);
}

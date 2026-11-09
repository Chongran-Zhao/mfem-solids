// ============================================================================
// vtu_writer.cpp of fracture_bar
//
// Writes the displacement, the phase field alpha, the pressure, the Cauchy
// and the second Piola-Kirchhoff stresses and the Green-Lagrange strain of
// each saved load step for ParaView, at the times of time.csv, sampled
// within each element by VTK_Tools. The steps and the fields are those the
// driver saved. The stress is that of the material of MaterialModelData,
// which must be the one the driver ran with, degraded by g(alpha) of its
// surface energy: g(alpha) sigma_0, the stress of the displacement problem.
// For the linear elastic material, sigma_0 = lambda tr(eps) I + 2 mu eps
// stands for the first Piola-Kirchhoff stress, from which VTK_Tools derives
// the Cauchy and the second Piola-Kirchhoff stresses with the formulas of
// finite strain; the three agree only to first order in the strain. It runs
// on MPI ranks, each writing every num_ranks-th step, and rank 0 the PVD
// file once all are written; every rank reads the results on
// results/mesh.mesh, the mesh the driver gathered them onto.
//
// Author: Chongran Zhao
// Date: Nov. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "IMaterialModel.hpp"
#include "ISurfaceEnergy.hpp"
#include "MaterialModelData.hpp"
#include "SystemTools.hpp"
#include "Tensor2_3D.hpp"
#include "VTK_Tools.hpp"

int main(int argc, char *argv[])
{
   mfem::Mpi::Init(argc, argv);
   if (!mfem::Mpi::Root())
      mfem::out.Disable();

   // 1. Read config.yaml.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path("config.yaml");
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Read the mesh the driver saved with its results, on which they live.
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   const std::string mesh_file = (results_dir / "mesh.mesh").string();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. Read the steps and their times from time.csv, whose columns are
   //    step,time,dt,iterations.
   std::ifstream time_file(results_dir / "time.csv");
   MFEM_VERIFY(time_file, "Cannot open " << (results_dir / "time.csv").string()
               << "; run the driver first.");
   std::vector<std::pair<int, double>> saved_steps;   // step and time
   std::string line;
   std::getline(time_file, line);
   while (std::getline(time_file, line))
   {
      std::istringstream row(line);
      std::string step_entry, time_entry;
      std::getline(row, step_entry, ',');
      std::getline(row, time_entry, ',');
      saved_steps.emplace_back(std::stoi(step_entry), std::stod(time_entry));
   }

   // Reads <results>/<prefix>_XXXX.gf of a step, with its own space.
   auto read_gf = [&](const std::string &prefix, int step)
   {
      std::ostringstream name;
      name << prefix << '_' << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ifstream gf_file(results_dir / name.str());
      MFEM_VERIFY(gf_file, "Cannot open " << (results_dir / name.str()).string()
                  << "; run the driver first.");
      return std::make_unique<mfem::GridFunction>(&mesh, gf_file);
   };

   // The material and the surface energy; they hold no state, so this
   // program creates its own.
   const std::unique_ptr<const IMaterialModel> material = set_material_model();
   const std::unique_ptr<const ISurfaceEnergy> surface_energy = set_surface_energy();

   // sigma_0 of the material, at the deformation gradient F of a point.
   auto first_PK_stress = [&](int, const Tensor2_3D &F, double)
   {
      return material->get_1st_PK_stress(F);
   };

   // g(alpha) at the points, the factor of the stress; it reads the alpha of
   // the step being written.
   mfem::GridFunctionCoefficient alpha_coefficient;
   mfem::TransformedCoefficient g_coefficient(
      &alpha_coefficient, [&surface_energy](double value) { return surface_energy->get_g(value); });

   VTK_Tools output(config["output"]["vtu"].as<std::string>(),
                    config["output"]["vtu_refine"].as<int>());
   output.set_energy_factor(g_coefficient);

   // 4. Write each saved step, each rank every num_ranks-th of them.
   for (std::size_t ss = mfem::Mpi::WorldRank(); ss < saved_steps.size();
        ss += mfem::Mpi::WorldSize())
   {
      const int step = saved_steps[ss].first;
      const std::unique_ptr<mfem::GridFunction> disp = read_gf("disp", step);
      const std::unique_ptr<mfem::GridFunction> alpha = read_gf("alpha", step);
      if (step == 0)
         SystemTools::print_space(*disp->FESpace());

      alpha_coefficient.SetGridFunction(alpha.get());
      output.save(step, *disp, nullptr, nullptr, nullptr, first_PK_stress,
                  {{"alpha", alpha.get()}});
   }

   // The PVD file, once every rank has written its steps.
   MPI_Barrier(MPI_COMM_WORLD);
   if (mfem::Mpi::Root())
      output.write_pvd(saved_steps);

   mfem::out << '\n';
   SystemTools::print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

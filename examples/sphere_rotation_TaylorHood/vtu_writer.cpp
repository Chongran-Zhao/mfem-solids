// ============================================================================
// vtu_writer.cpp of sphere_rotation_TaylorHood
//
// The vtu_writer of hyperelasticity_mixed_TaylorHood, sampled within each element by
// VTK_Tools_Sphere, in spherical components. Writes the
// displacement, the velocity, the acceleration, the Cauchy and the second
// Piola-Kirchhoff stresses and the Green-Lagrange strain E = 1/2 (F^T F - I)
// in spherical components, the displacement also in Cartesian ones, and the
// pressure at the sampling points, and the region of each element, for
// every vtu_interval-th saved time step, at the physical times of time.csv,
// with each element cut into vtu_refine^3 hexahedra. The
// steps and the fields are those the driver saved; the stresses take the
// material of the region of the element from MaterialModelData, which must
// be the one the driver ran with.
// It runs on MPI ranks, each writing every num_ranks-th step, and rank 0 the
// PVD file once all are written; every rank reads the results on
// results/mesh.mesh, the mesh the driver gathered them onto.
//
// Author: Chongran Zhao
// Date: Oct. 11, 2026
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

#include "MaterialModel.hpp"
#include "MaterialModelData.hpp"
#include "SystemTools.hpp"
#include "Tensor2_3D.hpp"
#include "VTK_Tools_Sphere.hpp"

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

   // 3. Read the steps and their physical times from time.csv, whose
   //    columns are step,time,dt,iterations.
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

   // The materials of the regions, by element attribute; they hold no
   // state, so this program creates its own.
   std::vector<std::unique_ptr<const MaterialModel>> materials;
   for (int attribute = 1; attribute <= static_cast<int>(region_materials.size()); attribute++)
      materials.push_back(set_material_model(attribute));

   // P = P_ich - p J F^-T of the material of the element attribute.
   auto first_PK_stress = [&](int attribute, const Tensor2_3D &F, double p)
   {
      return materials[attribute - 1]->get_1st_PK_stress_ich(F)
             - p * F.det() * F.inverse().transpose();
   };

   const int vtu_interval = config["output"]["vtu_interval"].as<int>();
   VTK_Tools_Sphere output(config["output"]["vtu"].as<std::string>(),
                           config["output"]["vtu_refine"].as<int>());

   // 4. Write every vtu_interval-th saved step, and the last one, each rank
   //    every num_ranks-th of them.
   std::vector<std::pair<int, double>> written_steps;   // step and time
   for (std::size_t ss = 0; ss < saved_steps.size(); ss++)
      if (ss % vtu_interval == 0 || ss + 1 == saved_steps.size())
         written_steps.push_back(saved_steps[ss]);
   for (std::size_t ss = mfem::Mpi::WorldRank(); ss < written_steps.size();
        ss += mfem::Mpi::WorldSize())
   {
      const int step = written_steps[ss].first;
      const std::unique_ptr<mfem::GridFunction> disp = read_gf("disp", step);
      const std::unique_ptr<mfem::GridFunction> velo = read_gf("velo", step);
      const std::unique_ptr<mfem::GridFunction> acce = read_gf("acce", step);
      const std::unique_ptr<mfem::GridFunction> pres = read_gf("pres", step);
      if (step == 0)
      {
         SystemTools::print_space(*disp->FESpace());
         SystemTools::print_space(*pres->FESpace());
      }

      output.save(step, *disp, velo.get(), acce.get(), pres.get(), first_PK_stress);
   }

   // The PVD file, once every rank has written its steps.
   MPI_Barrier(MPI_COMM_WORLD);
   if (mfem::Mpi::Root())
      output.write_pvd(written_steps);

   mfem::out << '\n';
   SystemTools::print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

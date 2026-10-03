// ============================================================================
// vtu_writer.cpp of dynamic_mixed
//
// Writes the displacement, the velocity, the acceleration, the pressure, and
// the first and second Piola-Kirchhoff stresses of each saved time step for
// ParaView, at the physical times of time.csv. The steps and the fields are
// those the driver saved, the pressure nodal; the stresses are computed at
// the element centers with the material of MaterialModelData, which must be
// the one the driver ran with.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
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
#include "VTK_Tools.hpp"

int main(int argc, char *argv[])
{
   // 1. Read config.yaml.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path("config.yaml");
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Read the mesh file.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. Read the steps and their physical times from time.csv, whose
   //    columns are step,time,dt,iterations.
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
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

   // The material; it holds no state, so this program creates its own.
   const std::unique_ptr<const MaterialModel> material = set_material_model();

   // First and second Piola-Kirchhoff stresses P and S at the element
   // centers.
   std::vector<Tensor2_3D> PK1(mesh.GetNE()), PK2(mesh.GetNE());

   VTK_Tools output(config["output"]["vtu"].as<std::string>());

   // 4. Write each saved step.
   for (const auto &[step, time] : saved_steps)
   {
      const std::unique_ptr<mfem::GridFunction> disp = read_gf("disp", step);
      const std::unique_ptr<mfem::GridFunction> velo = read_gf("velo", step);
      const std::unique_ptr<mfem::GridFunction> acce = read_gf("acce", step);
      const std::unique_ptr<mfem::GridFunction> pres = read_gf("pres", step);
      if (step == 0)
      {
         SystemTools::print_space(*disp->FESpace());
         SystemTools::print_space(*pres->FESpace());
      }

      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const Tensor2_3D F =
            VTK_Tools::get_center_deformation_gradient(*disp->FESpace(), *disp, ee);
         const double p = pres->GetValue(ee, mfem::Geometries.GetCenter(mesh.GetElementGeometry(ee)));
         PK1[ee] = material->get_1st_PK_stress_ich(F)
                    - p * F.det() * F.inverse().transpose();
         PK2[ee] = F.inverse() * PK1[ee];
      }

      output.save(step, time, *disp, *velo, *acce, *pres, PK1, PK2);
   }

   mfem::out << '\n';
   SystemTools::print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

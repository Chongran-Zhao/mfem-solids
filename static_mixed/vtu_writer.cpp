// ============================================================================
// vtu_writer.cpp
//
// Writes the displacement, the pressure, and the first and second
// Piola-Kirchhoff stresses of each load step for ParaView, from the
// displacement and nodal pressure saved by the mixed driver. Stresses are
// computed at the element centers with the material of MaterialModelData,
// which must be the one the driver ran with. It is serial and reads the
// results on results/mesh.mesh, the mesh the driver gathered them onto.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
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

   // 2. Read the mesh the driver saved with its results, on which they live.
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   const std::string mesh_file = (results_dir / "mesh.mesh").string();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. Read the results of each step and write them.
   const int load_steps = config["loading"]["load_steps"].as<int>();

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

   // First and second Piola-Kirchhoff stresses at element centers.
   std::vector<Tensor2_3D> PK1(mesh.GetNE()), PK2(mesh.GetNE());

   VTK_Tools output(config["output"]["vtu"].as<std::string>());

   for (int step = 0; step <= load_steps; step++)
   {
      const std::unique_ptr<mfem::GridFunction> disp = read_gf("disp", step);
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

      const double time = static_cast<double>(step) / load_steps;
      output.save(step, time, *disp, *pres, PK1, PK2);
   }

   mfem::out << '\n';
   SystemTools::print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

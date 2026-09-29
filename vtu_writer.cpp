// ============================================================================
// vtu_writer.cpp
//
// Writes the displacement and the first and second Piola-Kirchhoff stresses
// of each load step for ParaView.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "MaterialModelData.hpp"
#include "SystemTools.hpp"
#include "VTK_Tools.hpp"

int main(int argc, char *argv[])
{
   // 1. Read config.yaml.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path(SOURCE_DIR) / "config.yaml";
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Read the mesh file.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. Set up the material model.
   const MaterialModel_Hyperelasticity material = get_material_model();

   // 4. Read the displacement of each step and write it.
   const int load_steps = config["loading"]["load_steps"].as<int>();
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();

   VTK_Tools output(config["output"]["vtu"].as<std::string>());

   for (int step = 0; step <= load_steps; step++)
   {
      std::ostringstream name;
      name << "disp_" << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ifstream disp_file(results_dir / name.str());
      MFEM_VERIFY(disp_file, "Cannot open " << (results_dir / name.str()).string()
                  << "; run the driver first.");

      mfem::GridFunction disp(&mesh, disp_file);
      if (step == 0)
         SystemTools::print_space(*disp.FESpace());

      const double time = static_cast<double>(step) / load_steps;
      output.save(step, time, *disp.FESpace(), disp, material);
   }

   mfem::out << '\n';
   SystemTools::print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

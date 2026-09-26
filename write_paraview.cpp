// ============================================================================
// write_paraview.cpp
//
// Writes the results of the driver for ParaView: one VTU file per load step
// on the deformed mesh, with the displacement and the first and second
// Piola-Kirchhoff stresses, and a PVD file listing them.
// Step 1: read config.yaml.
// Step 2: read the labelled mesh written by read_mesh.
// Step 3: define the material, the same as in the driver.
// Step 4: read the displacement of each step and write it for ParaView.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include "MaterialModel.hpp"
#include "PrintInfo.hpp"
#include "VTK_write.hpp"
#include "mfem.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

int main(int argc, char *argv[])
{
   // 1. By default the config.yaml next to this source file is read.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path(SOURCE_DIR) / "config.yaml";
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Like the driver, this program runs in the directory of the mesh
   //    written by read_mesh and of the results of the driver, normally
   //    build/.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   print_mesh(mesh_file, mesh);

   // 3. The stresses written to the VTU files are computed from the
   //    displacement with the material of include/MaterialModel.hpp.
   const MaterialModel material = get_material_model();

   // 4. The driver saved step 0 and each of the load steps as
   //    <results>/disp_XXXX.gf. Each file holds its finite element space and
   //    the dof values, so the grid function is built from the mesh and the
   //    file alone. Step n is written at time n / N.
   const int load_steps = config["load_steps"].as<int>();
   const std::filesystem::path results_dir = config["output"]["results"].as<std::string>();

   VTK_write output(config["output"]["paraview"].as<std::string>());

   for (int step = 0; step <= load_steps; step++)
   {
      std::ostringstream name;
      name << "disp_" << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ifstream disp_file(results_dir / name.str());
      MFEM_VERIFY(disp_file, "Cannot open " << (results_dir / name.str()).string()
                  << "; run the driver first.");

      mfem::GridFunction disp(&mesh, disp_file);
      if (step == 0)
         print_space(disp.FESpace()->GetMaxElementOrder(), *disp.FESpace());

      const double time = static_cast<double>(step) / load_steps;
      output.save(step, time, *disp.FESpace(), disp, material);
   }

   mfem::out << '\n';
   print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

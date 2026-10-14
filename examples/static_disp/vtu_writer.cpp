// ============================================================================
// vtu_writer.cpp
//
// Writes the displacement, the pressure, the Cauchy and the second
// Piola-Kirchhoff stresses and the Green-Lagrange strain of each load step
// for ParaView, from the displacement saved by the driver, sampled within
// each element by VTK_Tools, which cuts it into vtu_refine^3 hexahedra. The
// stress is computed at the sampling points with the material of
// MaterialModelData, which must be the one the driver ran with, and the
// pressure is p = -tr(sigma) / 3. It is serial and reads the results on
// results/mesh.mesh, the mesh the driver gathered them onto.
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

   // P = F S(F) of the material, at the deformation gradient F of a point.
   auto first_PK_stress = [&](int, const Tensor2_3D &F, double)
   {
      return F * material->get_2nd_PK_stress(F);
   };

   VTK_Tools output(config["output"]["vtu"].as<std::string>(),
                    config["output"]["vtu_refine"].as<int>());

   for (int step = 0; step <= load_steps; step++)
   {
      const std::unique_ptr<mfem::GridFunction> disp = read_gf("disp", step);
      if (step == 0)
         SystemTools::print_space(*disp->FESpace());

      const double time = static_cast<double>(step) / load_steps;
      output.save(step, time, *disp, nullptr, nullptr, nullptr, first_PK_stress);
   }

   mfem::out << '\n';
   SystemTools::print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

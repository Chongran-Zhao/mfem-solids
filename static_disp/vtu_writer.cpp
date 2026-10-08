// ============================================================================
// vtu_writer.cpp
//
// Writes the displacement, the pressure, and the first and second
// Piola-Kirchhoff stresses of each load step for ParaView, from the
// displacement saved by the driver. The pressure p(J) and the stress are
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

   // Pressure p(J), piecewise constant, and the first and second
   // Piola-Kirchhoff stresses P and S, at the element centers.
   const int dim = mesh.Dimension();
   mfem::L2_FECollection fec_center(0, dim);
   mfem::FiniteElementSpace space_pres(&mesh, &fec_center);
   mfem::GridFunction pres(&space_pres);
   std::vector<Tensor2_3D> PK1(mesh.GetNE()), PK2(mesh.GetNE());

   VTK_Tools output(config["output"]["vtu"].as<std::string>());

   for (int step = 0; step <= load_steps; step++)
   {
      const std::unique_ptr<mfem::GridFunction> disp = read_gf("disp", step);
      if (step == 0)
         SystemTools::print_space(*disp->FESpace());

      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const Tensor2_3D F =
            VTK_Tools::get_center_deformation_gradient(*disp->FESpace(), *disp, ee);
         pres(ee) = material->get_p(F.det());
         PK2[ee] = material->get_2nd_PK_stress(F);
         PK1[ee] = F * PK2[ee];
      }

      const double time = static_cast<double>(step) / load_steps;
      output.save(step, time, *disp, pres, PK1, PK2);
   }

   mfem::out << '\n';
   SystemTools::print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

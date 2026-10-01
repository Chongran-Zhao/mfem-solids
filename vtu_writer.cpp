// ============================================================================
// vtu_writer.cpp
//
// Writes the displacement, the pressure, and the first and second
// Piola-Kirchhoff stresses of each load step for ParaView, from the
// displacement saved by the driver. The pressure p(J) and the stress are
// computed at the element centers with the material of MaterialModelData,
// which must be the one the driver ran with.
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
#include "LocalAssemblyTools.hpp"
#include "MaterialModelData.hpp"
#include "SystemTools.hpp"
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

   // 3. Read the results of each step and write them.
   const int load_steps = config["loading"]["load_steps"].as<int>();
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();

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
   const std::unique_ptr<const MaterialModel> material = get_material_model();

   // Pressure p(J) and first Piola-Kirchhoff stress P at the element centers:
   // piecewise constant, P with the 9 components P_xx, P_xy, ..., P_zz.
   const int dim = mesh.Dimension();
   mfem::L2_FECollection fec_center(0, dim);
   mfem::FiniteElementSpace space_pres(&mesh, &fec_center);
   mfem::FiniteElementSpace space_stress(&mesh, &fec_center, 9, mfem::Ordering::byVDIM);
   mfem::GridFunction pres(&space_pres), stress(&space_stress);

   VTK_Tools output(config["output"]["vtu"].as<std::string>());

   for (int step = 0; step <= load_steps; step++)
   {
      const std::unique_ptr<mfem::GridFunction> disp = read_gf("disp", step);
      if (step == 0)
         SystemTools::print_space(*disp->FESpace());

      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const Tensor2_3D F =
            LocalAssemblyTools::get_center_deformation_gradient(*disp->FESpace(), *disp, ee);
         pres(ee) = material->get_p(F.det());
         const Tensor2_3D PK1 = material->get_1st_PK_stress(F);
         for (int ii = 0; ii < 3; ii++)
            for (int JJ = 0; JJ < 3; JJ++)
               stress(space_stress.DofToVDof(ee, 3 * ii + JJ)) = PK1(ii, JJ);
      }

      const double time = static_cast<double>(step) / load_steps;
      output.save(step, time, *disp, pres, stress);
   }

   mfem::out << '\n';
   SystemTools::print_saved(output.get_pvd_path());
   mfem::out << '\n';

   return 0;
}

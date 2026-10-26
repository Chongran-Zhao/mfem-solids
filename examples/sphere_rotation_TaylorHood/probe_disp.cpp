// ============================================================================
// probe_disp.cpp of sphere_rotation_TaylorHood
//
// Writes the history of the circumferential displacement u_phi, in mm, of
// material points at radii on a ray of the equatorial plane z = 0 to a CSV
// file, one column per radius, the columns of probe_disp.py of
// sphere_rotation_vms. The default points are those of the measurements,
// 11.12, 32.31 and 53.38 mm on the ray 60 degrees from the x axis. Unlike
// probe_disp.py, which samples the VTU series of vtu_writer at fixed points
// of the current configuration, each element cut into linear hexahedra, it
// reads the fields the driver saved and evaluates them with the shape
// functions of their elements at the material points, found once in the
// reference configuration, u_phi = e_phi(X) . u(X).
//    ./probe_disp U_phi.csv [config.yaml]
// It is serial and reads the results on results/mesh.mesh, the mesh the
// driver gathered them onto.
//
// Author: Chongran Zhao
// Date: Oct. 25, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "SystemTools.hpp"

int main(int argc, char *argv[])
{
   MFEM_VERIFY(argc > 1, "Usage: ./probe_disp <output.csv> [config.yaml]");
   const std::filesystem::path csv_file(argv[1]);

   // The material points: radii in mm on the ray of z = 0 at angle degrees
   // from the x axis.
   const std::vector<double> radii = {11.12, 32.31, 53.38};
   const double angle = 60.0 * M_PI / 180.0;

   // 1. Read config.yaml.
   const std::filesystem::path yaml_file =
      (argc > 2) ? std::filesystem::path(argv[2])
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

   // 4. Find the element and the reference point of each material point,
   //    in the reference configuration, the mesh.
   const int num_points = static_cast<int>(radii.size());
   mfem::DenseMatrix point_mat(3, num_points);
   for (int pp = 0; pp < num_points; pp++)
   {
      point_mat(0, pp) = radii[pp] * 1.0e-3 * std::cos(angle);
      point_mat(1, pp) = radii[pp] * 1.0e-3 * std::sin(angle);
      point_mat(2, pp) = 0.0;
   }
   mfem::Array<int> elem_ids;
   mfem::Array<mfem::IntegrationPoint> ref_points;
   MFEM_VERIFY(mesh.FindPoints(point_mat, elem_ids, ref_points) == num_points,
               "Not all the material points lie in the mesh.");

   // 5. u_phi = -sin(angle) u_x + cos(angle) u_y at each saved step, in mm.
   std::ofstream out(csv_file);
   MFEM_VERIFY(out, "Cannot open " << csv_file.string());
   out << "time_ms";
   for (const double radius : radii)
      out << ",r" << radius << "mm";
   out << '\n' << std::setprecision(16);

   mfem::Vector value(3);
   for (const auto &[step, time] : saved_steps)
   {
      std::ostringstream name;
      name << "disp_" << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ifstream gf_file(results_dir / name.str());
      MFEM_VERIFY(gf_file, "Cannot open " << (results_dir / name.str()).string()
                  << "; run the driver first.");
      const mfem::GridFunction disp(&mesh, gf_file);

      out << time * 1.0e3;
      for (int pp = 0; pp < num_points; pp++)
      {
         disp.GetVectorValue(elem_ids[pp], ref_points[pp], value);
         out << ',' << (-std::sin(angle) * value(0) + std::cos(angle) * value(1)) * 1.0e3;
      }
      out << '\n';
   }

   mfem::out << '\n';
   SystemTools::print_saved(csv_file.string());
   mfem::out << '\n';

   return 0;
}

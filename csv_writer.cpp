// ============================================================================
// csv_writer.cpp
//
// Writes the mean displacement and the resultant force on the faces and
// directions of the csv_writer section of config.yaml at each load step, one
// CSV file per face.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "SystemTools.hpp"

// One reported face.
struct reported_face
{
   std::string name;                                // face name
   mfem::Array<int> face_marker;                    // marker of its boundary attributes
   std::vector<int> dirs;                           // reported directions, 0, 1, 2 for x, y, z
   std::array<mfem::Array<int>, 3> component_dofs;  // dofs of x, y, z on the face
   double area;                                     // reference area A_0
   std::ofstream csv;                               // CSV file of the face
};

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

   // 3. Set up the displacement space of the driver; the internal force
   //    lives in the same space.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   mfem::H1_FECollection fec(order, dim);
   mfem::FiniteElementSpace fespace(&mesh, &fec, dim, mfem::Ordering::byVDIM);
   mfem::GridFunction disp(&fespace);
   mfem::Vector internal_force(fespace.GetTrueVSize());
   SystemTools::print_space(fespace);

   // 4. Collect the faces and directions of the csv_writer section.
   const int load_steps = config["loading"]["load_steps"].as<int>();
   const std::filesystem::path results_dir = config["output"]["gf"].as<std::string>();
   const std::filesystem::path output_dir = config["output"]["csv"].as<std::string>();
   std::filesystem::create_directories(output_dir);

   const std::array<std::string, 3> component_names = {"x", "y", "z"};
   const std::map<std::string, int> dir_map = {{"x", 0}, {"y", 1}, {"z", 2}};

   // Reference area A_0 = int dA of a face.
   auto get_area = [&](const mfem::Array<int> &face_marker)
   {
      double area = 0.0;
      for (int be = 0; be < mesh.GetNBE(); be++)
      {
         if (face_marker[mesh.GetBdrAttribute(be) - 1] == 0)
            continue;

         const mfem::FiniteElement &face_elem = *fespace.GetBE(be);
         mfem::ElementTransformation &face_map = *fespace.GetBdrElementTransformation(be);
         const mfem::IntegrationRule &quad_rule =
            mfem::IntRules.Get(face_elem.GetGeomType(), 2 * face_elem.GetOrder());
         for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
         {
            const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
            face_map.SetIntPoint(&quad_pt);
            area += quad_pt.weight * face_map.Weight();
         }
      }
      return area;
   };

   // The entries of one face go into one CSV file.
   std::vector<reported_face> faces;
   for (const YAML::Node &entry : config["csv_writer"])
   {
      const std::string input_face = entry["face"].as<std::string>();
      const std::string input_dir = entry["dir"].as<std::string>();
      MFEM_VERIFY(mesh.bdr_attribute_sets.AttributeSetExists(input_face),
                  "Unknown face \"" << input_face << "\" in csv_writer.");
      MFEM_VERIFY(dir_map.count(input_dir),
                  "Unknown dir \"" << input_dir << "\" in csv_writer.");

      auto found = std::find_if(faces.begin(), faces.end(),
         [&](const reported_face &face) { return face.name == input_face; });
      if (found == faces.end())
      {
         reported_face face;
         face.name = input_face;
         face.face_marker = mesh.bdr_attribute_sets.GetAttributeSetMarker(input_face);
         for (int axis = 0; axis < 3; axis++)
            fespace.GetEssentialTrueDofs(face.face_marker, face.component_dofs[axis], axis);
         face.area = get_area(face.face_marker);
         faces.push_back(std::move(face));
         found = faces.end() - 1;
      }
      found->dirs.push_back(dir_map.at(input_dir));
   }
   MFEM_VERIFY(!faces.empty(), "The csv_writer section of config.yaml has no entry.");

   // Mean displacement u_mean_k = (1 / A_0) int u_k dA of a face.
   auto get_mean_disp = [&](const reported_face &face)
   {
      std::array<double, 3> out = {0.0, 0.0, 0.0};
      mfem::Array<int> vdofs;
      mfem::Vector face_disp, shape;
      for (int be = 0; be < mesh.GetNBE(); be++)
      {
         if (face.face_marker[mesh.GetBdrAttribute(be) - 1] == 0)
            continue;

         const mfem::FiniteElement &face_elem = *fespace.GetBE(be);
         mfem::ElementTransformation &face_map = *fespace.GetBdrElementTransformation(be);
         const mfem::IntegrationRule &quad_rule =
            mfem::IntRules.Get(face_elem.GetGeomType(), 2 * face_elem.GetOrder());

         const int num_nodes = face_elem.GetDof();
         shape.SetSize(num_nodes);
         fespace.GetBdrElementVDofs(be, vdofs);
         disp.GetSubVector(vdofs, face_disp);

         for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
         {
            const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
            face_map.SetIntPoint(&quad_pt);
            face_elem.CalcShape(quad_pt, shape);
            const double dA = quad_pt.weight * face_map.Weight();
            for (int kk = 0; kk < 3; kk++)
               for (int aa = 0; aa < num_nodes; aa++)
                  out[kk] += face_disp(aa + kk * num_nodes) * shape(aa) * dA;
         }
      }
      for (int kk = 0; kk < 3; kk++)
         out[kk] /= face.area;
      return out;
   };

   // CSV header: step, load factor, then u, F and t = F / A_0 in the
   // reported directions.
   for (reported_face &face : faces)
   {
      face.csv.open(output_dir / (face.name + ".csv"));
      face.csv << "step,load_factor";
      const std::array<std::string, 3> quantities = {"u", "F", "t"};
      for (int qq = 0; qq < 3; qq++)
         for (int dir : face.dirs)
            face.csv << ',' << quantities[qq] << '_' << component_names[dir];
      face.csv << '\n' << std::scientific << std::setprecision(10);
   }

   // Reads <results>/<prefix>_XXXX.gf of a step into target, whose size must
   // match that of the file.
   auto read_gf = [&](const std::string &prefix, int step, mfem::Vector &target)
   {
      std::ostringstream name;
      name << prefix << '_' << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ifstream gf_file(results_dir / name.str());
      MFEM_VERIFY(gf_file, "Cannot open " << (results_dir / name.str()).string()
                  << "; run the driver first.");

      mfem::GridFunction file_gf(&mesh, gf_file);
      MFEM_VERIFY(file_gf.Size() == target.Size(),
                  "The space in " << name.str() << " differs from that of config.yaml.");
      target = file_gf;
   };

   // 5. Read the displacement and the internal force R^a_k of each step,
   //    saved by the driver, and write, for each face,
   //       u_mean_k = (1 / A_0) int u_k dA        mean displacement
   //       F_k      = sum_{a on face} R^a_k       resultant force
   //       t_k      = F_k / A_0                   mean nominal traction
   //    F is the reaction on a constrained face and the load on a traction
   //    face; on a free face it picks up the reactions of the edges it shares
   //    with constrained faces, so it has no meaning there.
   for (int step = 0; step <= load_steps; step++)
   {
      read_gf("disp", step, disp);
      read_gf("internal_force", step, internal_force);

      const double factor = static_cast<double>(step) / load_steps;

      mfem::out << std::string(74, '=') << '\n'
                << "Load step " << step << " / " << load_steps << "\n\n"
                << std::left << std::setw(10) << "face" << std::setw(6) << ""
                << std::setw(19) << "u_mean"
                << std::setw(19) << "F"
                << "t = F / A_0\n";

      for (reported_face &face : faces)
      {
         const std::array<double, 3> mean_disp = get_mean_disp(face);

         std::array<double, 3> force = {0.0, 0.0, 0.0};
         for (int kk = 0; kk < 3; kk++)
            for (int dof : face.component_dofs[kk])
               force[kk] += internal_force(dof);

         face.csv << step << ',' << factor;
         for (int dir : face.dirs)
            face.csv << ',' << mean_disp[dir];
         for (int dir : face.dirs)
            face.csv << ',' << force[dir];
         for (int dir : face.dirs)
            face.csv << ',' << force[dir] / face.area;
         face.csv << '\n';

         for (int dir : face.dirs)
            mfem::out << std::left << std::setw(10) << (dir == face.dirs[0] ? face.name : "")
                      << std::setw(6) << component_names[dir] << std::scientific
                      << std::setprecision(6) << std::setw(19) << mean_disp[dir]
                      << std::setw(19) << force[dir] << force[dir] / face.area
                      << std::defaultfloat << '\n';
      }
   }

   mfem::out << std::string(74, '=') << "\n\n";
   for (const reported_face &face : faces)
      SystemTools::print_saved(output_dir / (face.name + ".csv"));
   mfem::out << '\n';

   return 0;
}

// ============================================================================
// write_traction_disp.cpp
//
// For each face named by read_mesh: the mean
// displacement and the resultant force on the face at every load step,
// computed from the displacement saved by the driver.
// Step 1: read config.yaml.
// Step 2: read the labelled mesh written by read_mesh.
// Step 3: create the displacement space and the nonlinear form of the driver.
// Step 4: collect the faces named by read_mesh.
// Step 5: for each step, the mean displacement and the resultant force on
//         each face, printed and written to one CSV file per face.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "CompressibleHyperelasticIntegrator.hpp"
#include "MaterialModel.hpp"
#include "SystemTools.hpp"

// One face to report: its name, the marker array of its boundary attributes,
// the dofs of each component on it, its area and the CSV file.
struct ReportedFace
{
   std::string name;
   mfem::Array<int> face_marker;
   std::array<mfem::Array<int>, 3> component_dofs;
   double area;
   std::ofstream csv;
};

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
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. The same space and nonlinear form as in the driver, but without
   //    essential dofs, so that Mult gives the internal force
   //       R^a_k = int N_a,J P_kJ dV
   //    at every node. At a free node it vanishes up to the Newton
   //    tolerance; at a constrained node it is the reaction, the force the
   //    support applies to the body there.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   mfem::H1_FECollection fec(order, dim);
   mfem::FiniteElementSpace fespace(&mesh, &fec, dim, mfem::Ordering::byVDIM);
   mfem::GridFunction disp(&fespace);
   mfem::Vector internal_force(fespace.GetTrueVSize());
   SystemTools::print_space(fespace);

   const MaterialModel material = get_material_model();
   mfem::NonlinearForm nonlinear_form(&fespace);
   nonlinear_form.AddDomainIntegrator(new CompressibleHyperelasticIntegrator(material));

   // 4. For each face: its marker, the dofs of each component on it and its
   //    area A_0 in the reference configuration. All faces named by
   //    read_mesh, all steps from 0 to load_steps and all components x, y, z
   //    are reported.
   const int load_steps = config["loading"]["load_steps"].as<int>();
   const std::filesystem::path results_dir = config["output"]["results"].as<std::string>();
   const std::filesystem::path output_dir = config["output"]["csv"].as<std::string>();
   std::filesystem::create_directories(output_dir);

   const std::array<std::string, 3> component_names = {"x", "y", "z"};
   const std::vector<int> components = {0, 1, 2};

   std::vector<ReportedFace> faces;
   for (const std::string &name : mesh.bdr_attribute_sets.GetAttributeSetNames())
   {
      ReportedFace face;
      face.name = name;
      face.face_marker = mesh.bdr_attribute_sets.GetAttributeSetMarker(name);
      for (int axis = 0; axis < 3; axis++)
         fespace.GetEssentialTrueDofs(face.face_marker, face.component_dofs[axis], axis);
      face.area = 0.0;
      faces.push_back(std::move(face));
   }

   // Mean displacement on one face, u_mean_k = (1 / A_0) int u_k dA, on its
   // boundary elements; the area A_0 is summed on the way.
   auto get_mean_disp = [&](ReportedFace &face)
   {
      std::array<double, 3> out = {0.0, 0.0, 0.0};
      face.area = 0.0;
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

            face.area += dA;
            for (int kk = 0; kk < 3; kk++)
               for (int aa = 0; aa < num_nodes; aa++)
                  out[kk] += face_disp(aa + kk * num_nodes) * shape(aa) * dA;
         }
      }
      for (int kk = 0; kk < 3; kk++)
         out[kk] /= face.area;
      return out;
   };

   // CSV header: step, load factor, then u, F and t = F / A_0 for each
   // chosen component, e.g. u_x.
   for (ReportedFace &face : faces)
   {
      face.csv.open(output_dir / (face.name + ".csv"));
      face.csv << "step,load_factor";
      const std::array<std::string, 3> quantities = {"u", "F", "t"};
      for (int qq = 0; qq < 3; qq++)
         for (int component : components)
            face.csv << ',' << quantities[qq] << '_' << component_names[component];
      face.csv << '\n' << std::scientific << std::setprecision(10);
   }

   // 5. For each step: the displacement saved by the driver, the internal
   //    force from it, and for each face
   //       u_mean_k = (1 / A_0) int u_k dA        mean displacement
   //       F_k      = sum_{a on face} R^a_k       resultant force
   //       t_k      = F_k / A_0                   mean nominal traction
   //    printed as one table per step and appended to the CSV of the face.
   const mfem::Vector zero_rhs;
   for (int step = 0; step <= load_steps; step++)
   {
      std::ostringstream name;
      name << "disp_" << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ifstream disp_file(results_dir / name.str());
      MFEM_VERIFY(disp_file, "Cannot open " << (results_dir / name.str()).string()
                  << "; run the driver first.");

      mfem::GridFunction file_disp(&mesh, disp_file);
      MFEM_VERIFY(file_disp.Size() == disp.Size(),
                  "The space in " << name.str() << " differs from that of config.yaml.");
      disp = file_disp;
      nonlinear_form.Mult(disp, internal_force);

      const double factor = static_cast<double>(step) / load_steps;

      mfem::out << std::string(74, '=') << '\n'
                << "Load step " << step << " / " << load_steps << "\n\n"
                << std::left << std::setw(10) << "face" << std::setw(6) << ""
                << std::setw(19) << "u_mean"
                << std::setw(19) << "F"
                << "t = F / A_0\n";

      for (ReportedFace &face : faces)
      {
         const std::array<double, 3> mean_disp = get_mean_disp(face);

         std::array<double, 3> force = {0.0, 0.0, 0.0};
         for (int kk = 0; kk < 3; kk++)
            for (int dof : face.component_dofs[kk])
               force[kk] += internal_force(dof);

         face.csv << step << ',' << factor;
         for (int component : components)
            face.csv << ',' << mean_disp[component];
         for (int component : components)
            face.csv << ',' << force[component];
         for (int component : components)
            face.csv << ',' << force[component] / face.area;
         face.csv << '\n';

         for (int component : components)
            mfem::out << std::left << std::setw(10) << (component == components[0] ? face.name : "")
                      << std::setw(6) << component_names[component] << std::scientific
                      << std::setprecision(6) << std::setw(19) << mean_disp[component]
                      << std::setw(19) << force[component] << force[component] / face.area
                      << std::defaultfloat << '\n';
      }
   }

   mfem::out << std::string(74, '=') << "\n\n";
   for (const ReportedFace &face : faces)
      SystemTools::print_saved(output_dir / (face.name + ".csv"));
   mfem::out << '\n';

   return 0;
}

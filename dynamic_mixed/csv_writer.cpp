// ============================================================================
// csv_writer.cpp of dynamic_mixed
//
// Writes, at each saved time step of time.csv,
//    mean displacement, mean pressure and reaction force on the faces and
//    directions of the csv_writer section of config.yaml, one CSV file per
//    face. The reaction is the displacement part of the residual of the
//    equation of motion, M a + F_int(u,p) - F_ext(t), at the saved state, on
//    the constrained dofs: the force the supports exert, inertia included;
//    the energies of the whole body, energy.csv: the kinetic energy
//    1/2 v^T M v, the strain energy of the mixed form
//    int Psi_vol(J(p)) + Psi_ich(F) dV, and their sum, which stays constant
//    without loads and prescribed motion.
// Everything is computed with the material of MaterialModelData, which must
// be the one the driver ran with.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Mixed.hpp"
#include "LocalAssemblyTools.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "MaterialModel.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "SystemTools.hpp"
#include "Tensor2_3D.hpp"

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
                 : std::filesystem::path("config.yaml");
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. Read the mesh file.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   SystemTools::print_mesh(mesh_file, mesh);

   // 3. Set up the displacement and pressure spaces of the mixed driver.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   MFEM_VERIFY(order >= 2, "The mixed csv_writer needs space.order >= 2.");
   mfem::H1_FECollection fec_u(order, dim), fec_p(order - 1, dim);
   mfem::FiniteElementSpace space_u(&mesh, &fec_u, dim, mfem::Ordering::byVDIM);
   mfem::FiniteElementSpace space_p(&mesh, &fec_p);
   mfem::GridFunction disp(&space_u), velo(&space_u), acce(&space_u), pres(&space_p);
   SystemTools::print_space(space_u);
   SystemTools::print_space(space_p);

   // The global assembly of the driver, for the residual R(u,p) and the mass
   // matrix M: the material and the boundary conditions are created anew
   // from MaterialModelData and config.yaml.
   auto dirichlet = std::make_unique<DirichletBoundary>(config["Dirichlet"], space_u);
   auto neumann = std::make_unique<NeumannBoundary>(config["Neumann"], space_u);
   const bool is_traction_load = neumann->is_traction_load();
   auto local_assembly = std::make_unique<LocalAssembly_Mixed>(set_material_model());
   auto global_assembly = std::make_unique<GlobalAssembly_Mixed>(
      space_u, space_p, std::move(local_assembly), std::move(dirichlet), std::move(neumann));
   const std::unique_ptr<mfem::SparseMatrix> mass = global_assembly->assemble_mass();
   const std::unique_ptr<const MaterialModel> material = set_material_model();

   // sol = [u; p], the input of the residual, and residual its blocks; the
   // reaction is the displacement block.
   mfem::BlockVector sol(global_assembly->get_offsets());
   mfem::BlockVector residual(global_assembly->get_offsets());
   mfem::Vector momentum(space_u.GetTrueVSize());

   // 4. Read the steps and their physical times from time.csv, whose
   //    columns are step,time,dt,iterations, and collect the faces and
   //    directions of the csv_writer section.
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

         const mfem::FiniteElement &face_elem = *space_u.GetBE(be);
         mfem::ElementTransformation &face_map = *space_u.GetBdrElementTransformation(be);
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
            space_u.GetEssentialTrueDofs(face.face_marker, face.component_dofs[axis], axis);
         face.area = get_area(face.face_marker);
         faces.push_back(std::move(face));
         found = faces.end() - 1;
      }
      found->dirs.push_back(dir_map.at(input_dir));
   }
   MFEM_VERIFY(!faces.empty(), "The csv_writer section of config.yaml has no entry.");

   // Reference-area means of displacement and pressure on a face.
   // Pressure is scalar, with compression positive.
   auto get_mean_fields = [&](const reported_face &face)
   {
      std::array<double, 4> out = {0.0, 0.0, 0.0, 0.0};
      mfem::Array<int> vdofs;
      mfem::Vector face_disp, shape;
      for (int be = 0; be < mesh.GetNBE(); be++)
      {
         if (face.face_marker[mesh.GetBdrAttribute(be) - 1] == 0)
            continue;

         const mfem::FiniteElement &face_elem = *space_u.GetBE(be);
         mfem::ElementTransformation &face_map = *space_u.GetBdrElementTransformation(be);
         const mfem::IntegrationRule &quad_rule =
            mfem::IntRules.Get(face_elem.GetGeomType(), 2 * face_elem.GetOrder());

         const int num_nodes = face_elem.GetDof();
         shape.SetSize(num_nodes);
         space_u.GetBdrElementVDofs(be, vdofs);
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
            out[3] += pres.GetValue(face_map, quad_pt) * dA;
         }
      }
      for (double &value : out)
         value /= face.area;
      return out;
   };

   // Strain energy of the body in the mixed form, int Psi_vol(J(p)) +
   // Psi_ich(F) dV, with the quadrature rule of the local assembly.
   auto get_strain_energy = [&]()
   {
      double energy = 0.0;
      mfem::Array<int> vdofs;
      mfem::Vector elem_disp;
      mfem::DenseMatrix dN_dxi, dN_dX;
      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const mfem::FiniteElement &elem = *space_u.GetFE(ee);
         mfem::ElementTransformation &elem_map = *space_u.GetElementTransformation(ee);
         const mfem::IntegrationRule &quad_rule = LocalAssemblyTools::get_quad_rule(elem, elem_map);

         const int num_nodes = elem.GetDof();
         dN_dxi.SetSize(num_nodes, 3);
         dN_dX.SetSize(num_nodes, 3);
         space_u.GetElementVDofs(ee, vdofs);
         disp.GetSubVector(vdofs, elem_disp);

         for (int qq = 0; qq < quad_rule.GetNPoints(); qq++)
         {
            const mfem::IntegrationPoint &quad_pt = quad_rule.IntPoint(qq);
            elem_map.SetIntPoint(&quad_pt);
            elem.CalcDShape(quad_pt, dN_dxi);
            mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
            const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(elem_disp, dN_dX);
            const double p = pres.GetValue(elem_map, quad_pt);
            energy += quad_pt.weight * elem_map.Weight() * material->get_strain_energy(F, p);
         }
      }
      return energy;
   };

   // CSV header of a face: step, time, then u and F in the reported
   // directions, followed by reference face area and face-mean pressure p.
   for (reported_face &face : faces)
   {
      face.csv.open(output_dir / (face.name + ".csv"));
      face.csv << "step,time";
      const std::array<std::string, 2> quantities = {"u", "F"};
      for (int qq = 0; qq < 2; qq++)
         for (int dir : face.dirs)
            face.csv << ',' << quantities[qq] << '_' << component_names[dir];
      face.csv << ",area,p\n" << std::scientific << std::setprecision(10);
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

   // CSV header of the body.
   std::ofstream energy_csv(output_dir / "energy.csv");
   energy_csv << "step,time,kinetic,strain,total\n" << std::scientific << std::setprecision(10);

   // 5. Read the displacement, the velocity, the acceleration and the
   //    pressure of each saved step and write, for each face,
   //       u_mean_k = (1 / A_0) int u_k dA        mean displacement
   //       F_k      = sum_{a on face} R^a_k       reaction force
   //       area     = A_0                        reference face area
   //       p_mean   = (1 / A_0) int p dA          mean pressure
   //    with R the displacement block of [M a; 0] + R(u,p,t), and, for the
   //    body,
   //       kinetic  = 1/2 v^T M v
   //       strain   = int Psi_vol(J(p)) + Psi_ich(F) dV
   //       total    = kinetic + strain.
   //    F is the force the supports exert on a constrained face, inertia
   //    included. On a free face it is small but not zero: the equation of
   //    motion holds at the intermediate states of generalized-alpha, not at
   //    the saved ones.
   for (const auto &[step, time] : saved_steps)
   {
      read_gf("disp", step, disp);
      read_gf("velo", step, velo);
      read_gf("acce", step, acce);
      read_gf("pres", step, pres);
      sol.GetBlock(0) = disp;
      sol.GetBlock(1) = pres;

      if (is_traction_load)
         global_assembly->set_traction_load(time);
      global_assembly->assemble_residual(sol, residual);
      mass->AddMult(acce, residual.GetBlock(0));

      mass->Mult(velo, momentum);
      const double kinetic_energy = 0.5 * (velo * momentum);
      const double strain_energy = get_strain_energy();
      energy_csv << step << ',' << time << ',' << kinetic_energy << ',' << strain_energy
                 << ',' << kinetic_energy + strain_energy << '\n';

      mfem::out << std::string(74, '=') << '\n'
                << "Time step " << step << ", t = " << time << "\n\n"
                << std::left << std::setw(10) << "face" << std::setw(6) << ""
                << std::setw(19) << "u_mean"
                << std::setw(19) << "F"
                << std::setw(19) << "area" << "p_mean\n";

      for (reported_face &face : faces)
      {
         const std::array<double, 4> mean_fields = get_mean_fields(face);

         std::array<double, 3> force = {0.0, 0.0, 0.0};
         for (int kk = 0; kk < 3; kk++)
            for (int dof : face.component_dofs[kk])
               force[kk] += residual(dof);

         face.csv << step << ',' << time;
         for (int dir : face.dirs)
            face.csv << ',' << mean_fields[dir];
         for (int dir : face.dirs)
            face.csv << ',' << force[dir];
         face.csv << ',' << face.area << ',' << mean_fields[3] << '\n';

         for (int dir : face.dirs)
            mfem::out << std::left << std::setw(10) << (dir == face.dirs[0] ? face.name : "")
                      << std::setw(6) << component_names[dir] << std::scientific
                      << std::setprecision(6) << std::setw(19) << mean_fields[dir]
                      << std::setw(19) << force[dir] << std::setw(19) << face.area
                      << mean_fields[3]
                      << std::defaultfloat << '\n';
      }

      mfem::out << '\n' << std::left << std::setw(16) << "kinetic energy" << std::scientific
                << std::setprecision(6) << kinetic_energy << '\n'
                << std::setw(16) << "strain energy" << strain_energy << '\n'
                << std::setw(16) << "total energy" << kinetic_energy + strain_energy
                << std::defaultfloat << '\n';
   }

   mfem::out << std::string(74, '=') << "\n\n";
   for (const reported_face &face : faces)
      SystemTools::print_saved(output_dir / (face.name + ".csv"));
   SystemTools::print_saved(output_dir / "energy.csv");
   mfem::out << '\n';

   return 0;
}

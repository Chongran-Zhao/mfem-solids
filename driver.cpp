// ============================================================================
// driver.cpp
//
// Hyperelastostatics on the labelled mesh written by read_mesh. Boundary
// conditions are read from config.yaml and refer to faces by name.
// Step 1: read config.yaml.
// Step 2: read the labelled mesh written by read_mesh.
// Step 3: create the displacement finite element space.
// Step 4: collect the constrained dofs of each Dirichlet condition.
// Step 5: define the material.
// Step 6: build the nonlinear form and Newton's method.
// Step 7: raise the prescribed displacements over the load steps and save
//         the displacement of each step into results_gf, which
//         write_paraview and write_traction_disp read.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include "CompressibleHyperelasticIntegrator.hpp"
#include "MaterialModel.hpp"
#include "NewtonMonitor.hpp"
#include "PrescribedComponent.hpp"
#include "PrintInfo.hpp"
#include "mfem.hpp"
#include <yaml-cpp/yaml.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
   // 1. By default the config.yaml next to this source file is read.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path(SOURCE_DIR) / "config.yaml";
   const YAML::Node config = YAML::LoadFile(yaml_file.string());

   // 2. read_mesh writes the labelled mesh into the directory it runs in, so
   //    the driver runs in the same directory, normally build/.
   const std::string mesh_file = config["mesh"]["output"].as<std::string>();
   mfem::Mesh mesh(mesh_file);
   print_mesh(mesh_file, mesh);

   // 3. Continuous shape functions of the given order, with three components
   //    per node.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   mfem::H1_FECollection fec(order, dim);
   mfem::FiniteElementSpace fespace(&mesh, &fec, dim, mfem::Ordering::byVDIM);
   mfem::GridFunction disp(&fespace);
   disp = 0.0;
   print_space(order, fespace);

   // 4. Each Dirichlet condition names a face, the constrained components and
   //    their values. The face name gives the marker array of its boundary
   //    attributes, and GetEssentialTrueDofs collects the dofs of one
   //    component on the marked faces. A node on two constrained faces is
   //    collected twice, hence Sort and Unique at the end. Each prescribed
   //    component is kept for the load steps.
   const std::array<std::string, 3> component_names = {"x", "y", "z"};
   mfem::Array<int> ess_tdof_list;
   std::vector<PrescribedComponent> prescribed;

   for (const YAML::Node &condition : config["dirichlet"])
   {
      const std::string face = condition["face"].as<std::string>();
      const auto components = condition["components"].as<std::vector<std::string>>();
      const auto values = condition["values"].as<std::vector<double>>();

      MFEM_VERIFY(mesh.bdr_attribute_sets.AttributeSetExists(face),
                  "Unknown face \"" << face << "\" in config.yaml.");
      MFEM_VERIFY(components.size() == values.size(),
                  "Face \"" << face << "\": components and values differ in size.");

      mfem::Array<int> face_marker = mesh.bdr_attribute_sets.GetAttributeSetMarker(face);

      for (std::size_t cc = 0; cc < components.size(); cc++)
      {
         int component = -1;
         for (int axis = 0; axis < 3; axis++)
            if (components[cc] == component_names[axis])
               component = axis;
         MFEM_VERIFY(component >= 0, "Unknown component \"" << components[cc] << "\".");

         mfem::Array<int> component_dofs;
         fespace.GetEssentialTrueDofs(face_marker, component_dofs, component);
         ess_tdof_list.Append(component_dofs);
         prescribed.push_back({face, face_marker, component, values[cc],
                               component_dofs.Size()});
      }
   }
   ess_tdof_list.Sort();
   ess_tdof_list.Unique();
   print_dirichlet(prescribed, ess_tdof_list.Size());

   // 5. The material is given in include/MaterialModel.hpp, so that
   //    write_paraview and write_traction_disp use the same one.
   const MaterialModel material = get_material_model();

   // 6. The nonlinear form owns the integrator; the material must outlive it.
   //    Newton's method solves R(d) = 0 from the current disp, with CG for
   //    the linear systems.
   mfem::NonlinearForm nonlinear_form(&fespace);
   nonlinear_form.AddDomainIntegrator(new CompressibleHyperelasticIntegrator(material));
   nonlinear_form.SetEssentialTrueDofs(ess_tdof_list);

   const YAML::Node solver = config["solver"];

   mfem::GSSmoother preconditioner;
   mfem::CGSolver linear_solver;
   linear_solver.SetRelTol(solver["linear_rel_tol"].as<double>());
   linear_solver.SetAbsTol(0.0);
   linear_solver.SetMaxIter(solver["linear_max_iter"].as<int>());
   linear_solver.SetPrintLevel(-1);
   linear_solver.SetPreconditioner(preconditioner);

   mfem::NewtonSolver newton_solver;
   newton_solver.SetOperator(nonlinear_form);
   newton_solver.SetSolver(linear_solver);
   newton_solver.SetRelTol(solver["newton_rel_tol"].as<double>());
   newton_solver.SetAbsTol(0.0);
   newton_solver.SetMaxIter(solver["newton_max_iter"].as<int>());
   newton_solver.SetPrintLevel(-1);
   newton_solver.iterative_mode = true;

   NewtonMonitor newton_monitor;
   newton_solver.SetMonitor(newton_monitor);

   // 7. At load step n of N every prescribed value is scaled by n / N and
   //    projected onto its face; the other nodes keep the previous solution,
   //    which is the initial guess of Newton's method.
   const int load_steps = config["load_steps"].as<int>();
   const std::string length_unit = config["units"]["length"].as<std::string>();

   // The displacement of each step as an MFEM grid function,
   // <results>/disp_XXXX.gf, step 0 included. The folder is emptied first so
   // that no files of an earlier run with more steps are left.
   const std::filesystem::path results_dir = config["output"]["results"].as<std::string>();
   std::filesystem::remove_all(results_dir);
   std::filesystem::create_directories(results_dir);

   auto save_disp = [&](int step)
   {
      std::ostringstream name;
      name << "disp_" << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ofstream disp_file(results_dir / name.str());
      disp_file.precision(16);
      disp.Save(disp_file);
   };
   save_disp(0);

   const mfem::Vector zero_rhs;
   for (int step = 1; step <= load_steps; step++)
   {
      const double factor = static_cast<double>(step) / load_steps;

      for (const PrescribedComponent &item : prescribed)
      {
         mfem::ConstantCoefficient value(factor * item.value);
         std::array<mfem::Coefficient *, 3> face_disp = {nullptr, nullptr, nullptr};
         face_disp[item.component] = &value;
         disp.ProjectBdrCoefficient(face_disp.data(), item.face_marker);
      }

      print_load_step(step, load_steps, factor, prescribed, length_unit);

      newton_solver.Mult(zero_rhs, disp);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at step " << step << ".");

      mfem::out << "converged in " << newton_solver.GetNumIterations() << " iterations\n";

      save_disp(step);
   }

   mfem::out << std::string(74, '=') << "\n\n";
   print_saved(results_dir);
   mfem::out << '\n';

   return 0;
}

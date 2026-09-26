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
// Step 7: raise the prescribed displacements over the load steps.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include "CompressibleHyperelasticIntegrator.hpp"
#include "CompressibleNeoHookean.hpp"
#include "VTK_write.hpp"
#include "mfem.hpp"
#include <yaml-cpp/yaml.h>
#include <array>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// Prints the residual norm of every Newton iteration, absolute and relative
// to the first iteration of the load step. NewtonSolver calls
// MonitorResidual once per iteration, and once more with final = true.
class NewtonMonitor : public mfem::IterativeSolverMonitor
{
public:
   void MonitorResidual(int it, mfem::real_t norm, const mfem::Vector &,
                        bool final) override
   {
      if (final)
         return;
      if (it == 0)
         initial_norm = norm;
      mfem::out << std::left << std::setw(12) << it << std::scientific
                << std::setprecision(6) << std::setw(18) << norm
                << norm / initial_norm << std::defaultfloat << '\n';
   }

private:
   double initial_norm = 1.0;
};

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

   mfem::out << "\nMesh\n" << std::string(74, '-') << '\n' << std::left
             << std::setw(20) << "file" << std::filesystem::absolute(mesh_file).string() << '\n'
             << std::setw(20) << "elements" << mesh.GetNE() << '\n'
             << std::setw(20) << "boundary elements" << mesh.GetNBE() << '\n'
             << std::setw(20) << "faces";
   for (const std::string &name : mesh.bdr_attribute_sets.GetAttributeSetNames())
      mfem::out << name << ' ';
   mfem::out << '\n' << std::string(74, '-') << '\n';

   // 3. Continuous shape functions of the given order, with three components
   //    per node.
   const int dim = mesh.Dimension();
   const int order = config["space"]["order"].as<int>();
   mfem::H1_FECollection fec(order, dim);
   mfem::FiniteElementSpace fespace(&mesh, &fec, dim, mfem::Ordering::byVDIM);
   mfem::GridFunction disp(&fespace);
   disp = 0.0;

   mfem::out << std::setw(20) << "order" << order << '\n'
             << std::setw(20) << "unknowns" << fespace.GetTrueVSize() << '\n'
             << std::string(74, '-') << '\n';

   // 4. Each Dirichlet condition names a face, the constrained components and
   //    their values. The face name gives the marker array of its boundary
   //    attributes, and GetEssentialTrueDofs collects the dofs of one
   //    component on the marked faces. A node on two constrained faces is
   //    collected twice, hence Sort and Unique at the end.
   const std::array<std::string, 3> component_names = {"x", "y", "z"};
   mfem::Array<int> ess_tdof_list;

   // One prescribed component on one face, kept for the load steps.
   struct PrescribedComponent
   {
      std::string face;
      mfem::Array<int> face_marker;
      int component;
      double value;
   };
   std::vector<PrescribedComponent> prescribed;

   mfem::out << "\nDirichlet conditions\n"
             << std::setw(10) << "face" << std::setw(12) << "component"
             << std::setw(12) << "value" << "dofs\n" << std::string(74, '-') << '\n';

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
         prescribed.push_back({face, face_marker, component, values[cc]});

         mfem::out << std::setw(10) << face << std::setw(12) << components[cc]
                   << std::setw(12) << values[cc] << component_dofs.Size() << '\n';
      }
   }
   ess_tdof_list.Sort();
   ess_tdof_list.Unique();

   mfem::out << std::string(74, '-') << '\n'
             << std::setw(34) << "constrained unknowns" << ess_tdof_list.Size() << "\n\n";

   // 5. Material, kept in the code for now: Young's modulus and Poisson's
   //    ratio of the reference case, converted to shear and bulk moduli.
   const double young = 540.0e3;
   const double poisson = 0.324;
   const double mu = young / (2.0 * (1.0 + poisson));
   const double kappa = young / (3.0 * (1.0 - 2.0 * poisson));
   const CompressibleNeoHookean material(kappa, mu);

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
   //    which is the initial guess of Newton's method. Each converged step is
   //    saved for ParaView.
   const int load_steps = config["load_steps"].as<int>();
   const std::string length_unit = config["units"]["length"].as<std::string>();

   // One VTU file per step on the deformed mesh, with the displacement and
   // the first and second Piola-Kirchhoff stresses; step 0 is the undeformed
   // state.
   VTK_write output(config["output"]["paraview"].as<std::string>());
   output.save(0, 0.0, fespace, disp, material);


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

      // Header: the displacement applied on each face at this step.
      mfem::out << std::string(74, '=') << '\n'
                << "Load step " << step << " / " << load_steps << '\n';
      for (std::size_t ii = 0; ii < prescribed.size(); ii++)
      {
         const PrescribedComponent &item = prescribed[ii];
         const bool new_face = (ii == 0 || prescribed[ii - 1].face != item.face);
         if (new_face)
            mfem::out << (ii == 0 ? "" : "\n") << "  " << std::left << std::setw(8)
                      << item.face << "displacement ";
         else
            mfem::out << ", ";
         mfem::out << 'u' << component_names[item.component] << " = "
                   << factor * item.value << ' ' << length_unit;
      }
      mfem::out << "\n\n"
                << std::left << std::setw(12) << "iteration" << std::setw(18) << "||R||"
                << "||R|| / ||R_0||\n";

      newton_solver.Mult(zero_rhs, disp);
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton did not converge at step " << step << ".");

      mfem::out << "converged in " << newton_solver.GetNumIterations() << " iterations\n";

      output.save(step, factor, fespace, disp, material);
   }

   mfem::out << std::string(74, '=') << "\n\n" << std::left << std::setw(20) << "saved"
             << std::filesystem::absolute(output.get_pvd_path()).string() << "\n\n";

   return 0;
}

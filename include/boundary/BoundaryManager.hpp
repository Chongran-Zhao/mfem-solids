// ============================================================================
// BoundaryManager.hpp
//
// This class manages the two types of boundary conditions, Dirichlet and
// Neumann.
//
// Author: Chongran Zhao
// Date: Sep. 28, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef BOUNDARY_MANAGER_HPP
#define BOUNDARY_MANAGER_HPP

#include "DirichletBoundary.hpp"
#include "LoadData.hpp"
#include "NeumannBoundary.hpp"
#include "mfem.hpp"
#include <yaml-cpp/yaml.h>
#include <string>

class BoundaryManager
{
public:

   // Constructor.
   BoundaryManager(const YAML::Node &config, mfem::FiniteElementSpace &fespace)
      : num_load_steps(config["loading"]["load_steps"].as<int>()),
        dirichlet(config["Dirichlet"], fespace),
        neumann(config["Neumann"], fespace)
   {
      const std::string loading_type = config["loading"]["type"].as<std::string>();
      if (loading_type == "displacement")
      {
         MFEM_VERIFY(is_disp_load() && !is_traction_load(),
                     "The loading type is displacement, so disp_bc must have an "
                     "entry and Neumann no face.");
      }
      else if (loading_type == "traction")
      {
         MFEM_VERIFY(is_traction_load() && !is_disp_load(),
                     "The loading type is traction, so Neumann must have a face "
                     "and disp_bc no entry.");
      }
      else
         MFEM_ABORT("Unknown loading type \"" << loading_type << "\".");
   }

   // Number of load steps defined in config.yaml.
   int get_num_load_steps() const { return num_load_steps; }

   // All constrained dofs of the Dirichlet bc.
   mfem::Array<int> get_ess_tdof_list() const { return dirichlet.get_ess_tdof_list(); }

   // Whether the loading is a prescribed displacement.
   bool is_disp_load() const { return dirichlet.is_disp_load(); }

   // Whether the loading is a traction.
   bool is_traction_load() const { return neumann.is_traction_load(); }

   // Set disp to zero on the fixed dofs.
   void apply_fixed_bc(mfem::GridFunction &disp) const
   { dirichlet.apply_fixed_bc(disp); }

   // Set the prescribed disp on the displacement-driven dofs.
   void apply_disp_load_bc(int step, mfem::GridFunction &disp) const
   {
      dirichlet.apply_disp_load_bc(static_cast<double>(step) / num_load_steps, disp);
   }

   // Add the traction integrators to the LinearForm of the external force.
   void add_traction_integrators(mfem::LinearForm &external_force)
   {
      neumann.add_traction_integrators(external_force);
   }

   // Update the tractions to the given load step.
   void update_traction(int step)
   { neumann.set_time(static_cast<double>(step) / num_load_steps); }

   // Print the fixed faces.
   void print_fixed_bc() const { dirichlet.print_fixed_bc(); }

   // Print the loaded faces (displacement or traction).
   void print_load() const
   {
      if (is_disp_load())
         dirichlet.print_disp_load();
      else
         neumann.print_traction_load();
   }

   // Print the header of the load step and the load value at that step.
   void print_load_by_step(int step, const mfem::GridFunction &disp,
                           const mfem::LinearForm &traction_force) const
   {
      mfem::out << std::string(74, '=') << '\n'
                << "Load step " << step << " / " << get_num_load_steps() << '\n';

      if (is_disp_load())
         dirichlet.print_disp_load_by_step(disp);
      else
         neumann.print_traction_load_by_step(traction_force);
   }

private:
   const int num_load_steps;                      // number of load steps N
   DirichletBoundary dirichlet;                   // Dirichlet section
   NeumannBoundary neumann;                       // Neumann section
};

#endif

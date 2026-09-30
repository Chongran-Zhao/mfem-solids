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

#include <string>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "DirichletBoundary.hpp"
#include "LoadData.hpp"
#include "NeumannBoundary.hpp"

class BoundaryManager
{
public:

   // Constructor.
   BoundaryManager(const YAML::Node &config, mfem::FiniteElementSpace &fespace)
      : dirichlet(config["Dirichlet"], fespace),
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

   // All constrained dofs of the Dirichlet bc.
   mfem::Array<int> get_ess_tdof_list() const { return dirichlet.get_ess_tdof_list(); }

   // Whether the loading is a prescribed displacement.
   bool is_disp_load() const { return dirichlet.is_disp_load(); }

   // Whether the loading is a traction.
   bool is_traction_load() const { return neumann.is_traction_load(); }

   // Set disp to zero on the fixed dofs.
   void apply_fixed_bc(mfem::GridFunction &disp) const
   { dirichlet.apply_fixed_bc(disp); }

   // Set the prescribed disp at time tt on the displacement-driven dofs.
   void apply_disp_load_bc(double tt, mfem::GridFunction &disp) const
   {
      dirichlet.apply_disp_load_bc(tt, disp);
   }

   // Add the traction integrators to the LinearForm of the external force.
   void add_traction_integrators(mfem::LinearForm &external_force)
   {
      neumann.add_traction_integrators(external_force);
   }

   // Update the tractions to time tt.
   void update_traction(double tt)
   { neumann.set_time(tt); }

   // Print the fixed faces.
   void print_fixed_bc() const { dirichlet.print_fixed_bc(); }

   // Print the loaded faces (displacement or traction).
   void print_load_faces() const
   {
      if (is_disp_load())
         dirichlet.print_disp_load();
      else
         neumann.print_traction_load();
   }

   // Print the load value: the prescribed disp, or the resultant traction force.
   void print_load(const mfem::GridFunction &disp,
                         const mfem::LinearForm &traction_force) const
   {
      if (is_disp_load())
         dirichlet.print_disp_load_time(disp);
      else
         neumann.print_traction_load_time(traction_force);
   }

private:
   DirichletBoundary dirichlet;                   // Dirichlet section
   NeumannBoundary neumann;                       // Neumann section
};

#endif

// ============================================================================
// DirichletBoundary.hpp
//
// Reads the Dirichlet boundary conditions from config.yaml and applies them
// to the displacement at each load step.
//
// Author: Chongran Zhao
// Date: Sep. 27, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef DIRICHLET_BOUNDARY_HPP
#define DIRICHLET_BOUNDARY_HPP

#include <algorithm>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "LoadData.hpp"

class DirichletBoundary
{
public:
   // Reads the Dirichlet section of config.yaml.
   DirichletBoundary(const YAML::Node &paras, mfem::FiniteElementSpace &fespace)
   {
      mfem::Mesh &mesh = *fespace.GetMesh();

      // Construct face_attribute_map.
      for (const std::string &name :
           mesh.bdr_attribute_sets.GetAttributeSetNames())
         face_attribute_map[name] =
            mesh.bdr_attribute_sets.GetAttributeSetMarker(name);

      // Fixed faces.
      for (const YAML::Node &bc : paras["fixed_bc"])
      {
         const std::string input_face = bc["face"].as<std::string>();
         const std::string input_dir = bc["dir"].as<std::string>();

         disp_fixed fixed;
         fixed.face = input_face;
         fixed.dir = dir_map.at(input_dir);
         fespace.GetEssentialTrueDofs(face_attribute_map.at(input_face)
            ,fixed.dofs, fixed.dir);

         ess_tdof_list.Append(fixed.dofs);
         disp_fixed_list.push_back(fixed);
      }

      // Displacement-driven faces; the values come from LoadData::disp_driven.
      for (const YAML::Node &bc : paras["disp_bc"])
      {
         const std::string input_face = bc["face"].as<std::string>();
         const std::string input_dir = bc["dir"].as<std::string>();

         disp_load load;
         load.face = input_face;
         load.dir = dir_map.at(input_dir);
         fespace.GetEssentialTrueDofs(face_attribute_map.at(input_face),
            load.dofs, load.dir);

         ess_tdof_list.Append(load.dofs);
         disp_load_list.push_back(load);
      }

      ess_tdof_list.Sort();
      ess_tdof_list.Unique();
   }

   // Whether disp_bc has any entry.
   bool is_disp_load() const { return !disp_load_list.empty(); }

   // All constrained dofs, for NonlinearForm::SetEssentialTrueDofs.
   mfem::Array<int> get_ess_tdof_list() const { return ess_tdof_list; }

   // Set the displacement to zero on the fixed faces.
   void apply_fixed_bc(mfem::GridFunction &disp) const
   {
      for (const disp_fixed &fixed : disp_fixed_list)
         for (int dof : fixed.dofs)
            disp(dof) = 0.0;
   }

   // Apply the disp loading given by LoadData::disp_driven(pt, tt).
   void apply_disp_load_bc(double tt, mfem::GridFunction &disp) const
   {
      for (const disp_load &load : disp_load_list)
      {
         mfem::FunctionCoefficient value([&](const mfem::Vector &pt)
         { return LoadData::disp_driven(pt, tt, load.face)(load.dir); });

         mfem::Coefficient *coeff[3] = {nullptr, nullptr, nullptr};
         coeff[load.dir] = &value;
         disp.ProjectBdrCoefficient(coeff, face_attribute_map.at(load.face));
      }
   }


   // Print the fixed faces and the number of all constrained unknowns.
   void print_fixed_bc() const
   {
      mfem::out << "\nFixed boundary\n" << std::left
                << std::setw(10) << "face"
                << std::setw(12) << "dir"
                << "dofs\n"
                << std::string(74, '-') << '\n';

      for (const disp_fixed &fixed : disp_fixed_list)
         mfem::out << std::setw(10) << fixed.face
                   << std::setw(12) << "xyz"[fixed.dir]
                   << fixed.dofs.Size() << '\n';

      mfem::out << std::string(74, '-') << '\n'
                << std::setw(34) << "constrained unknowns"
                << ess_tdof_list.Size() << "\n\n";
   }

   // Print the displacement-driven faces.
   void print_disp_load() const
   {
      mfem::out << "Displacement loading\n" << std::left
                << std::setw(10) << "face"
                << std::setw(12) << "dir"
                << "dofs\n"
                << std::string(74, '-') << '\n';

      for (const disp_load &load : disp_load_list)
         mfem::out << std::setw(10) << load.face
                   << std::setw(12) << "xyz"[load.dir]
                   << load.dofs.Size() << '\n';

      mfem::out << std::string(74, '-') << "\n\n";
   }


   // Print the prescribed displacement of each driven face.
   void print_disp_load_by_step(const mfem::GridFunction &disp) const
   {
      for (const disp_load &load : disp_load_list)
      {
         double min_disp = std::numeric_limits<double>::max();
         double max_disp = std::numeric_limits<double>::lowest();
         for (int dof : load.dofs)
         {
            min_disp = std::min(min_disp, disp(dof));
            max_disp = std::max(max_disp, disp(dof));
         }

         mfem::out << "  " << std::left
                   << std::setw(8) << load.face
                   << "displacement u" << "xyz"[load.dir];
         if (min_disp == max_disp)
            mfem::out << " = " << min_disp << '\n';
         else
            mfem::out << " in [" << min_disp << ", " << max_disp << "]\n";
      }
   }

private:
   // One direction on a fixed face.
   struct disp_fixed
   {
      std::string face;          // face name
      int dir;                   // 0, 1, 2 for x, y, z
      mfem::Array<int> dofs;     // dofs of this direction on the face
   };

   // One direction on a displacement-driven face.
   struct disp_load
   {
      std::string face;          // face name
      int dir;                   // 0, 1, 2 for x, y, z
      mfem::Array<int> dofs;     // dofs of this direction on the face
   };

   // "x", "y", "z" -> 0, 1, 2
   inline static const std::map<std::string, int> dir_map = {{"x", 0}, {"y", 1}, {"z", 2}};

   // Face name -> marker array of its boundary attributes, e.g.
   // {"left":   [1, 0, 0, 0, 0, 0],
   //  "right":  [0, 1, 0, 0, 0, 0],
   //  "front":  [0, 0, 1, 0, 0, 0],
   //  "back":   [0, 0, 0, 1, 0, 0],
   //  "bottom": [0, 0, 0, 0, 1, 0],
   //  "top":    [0, 0, 0, 0, 0, 1]}
   std::map<std::string, mfem::Array<int>> face_attribute_map;

   mfem::Array<int> ess_tdof_list;           // union of all constrained dofs
   std::vector<disp_fixed> disp_fixed_list;  // one per entry of fixed_bc
   std::vector<disp_load> disp_load_list;    // one per entry of disp_bc
};

#endif

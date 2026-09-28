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

#include "mfem.hpp"
#include <yaml-cpp/yaml.h>
#include <map>
#include <iomanip>
#include <string>
#include <vector>

class DirichletBoundary
{
public:
   // Reads the Dirichlet section of config.yaml.
   DirichletBoundary(const YAML::Node &paras, mfem::FiniteElementSpace &fespace)
      : is_disp_load(paras["is_disp_load"].as<bool>()),
        num_load_steps(is_disp_load ? paras["load_steps"].as<int>() : 0)
   {
      mfem::Mesh &mesh = *fespace.GetMesh();

      // Face name -> marker array of its boundary attributes, e.g.
      // {"left":   [1, 0, 0, 0, 0, 0],
      //  "right":  [0, 1, 0, 0, 0, 0],
      //  "front":  [0, 0, 1, 0, 0, 0],
      //  "back":   [0, 0, 0, 1, 0, 0],
      //  "bottom": [0, 0, 0, 0, 1, 0],
      //  "top":    [0, 0, 0, 0, 0, 1]}
      std::map<std::string, mfem::Array<int>> face_attribute_map;
      for (const std::string &name :
           mesh.bdr_attribute_sets.GetAttributeSetNames())
         face_attribute_map[name] =
            mesh.bdr_attribute_sets.GetAttributeSetMarker(name);

      // Fixed faces, one entry per direction.
      for (const YAML::Node &bc : paras["fixed_bc"])
      {
         const std::string input_face = bc["face"].as<std::string>();
         const std::string input_dir = bc["dir"].as<std::string>();

         disp_fixed_face fixed_face;
         fixed_face.face = input_face;
         fixed_face.dir = dir_map.at(input_dir);
         fespace.GetEssentialTrueDofs(face_attribute_map.at(input_face)
            ,fixed_face.dofs, fixed_face.dir);

         ess_tdof_list.Append(fixed_face.dofs);
         disp_fixed_list.push_back(fixed_face);
      }

      // Displacement-driven faces, read only if is_disp_load.
      if (is_disp_load)
         // One entry per direction.
         for (const YAML::Node &bc : paras["disp_bc"])
         {
            const std::string input_face = bc["face"].as<std::string>();
            const std::string input_dir = bc["dir"].as<std::string>();
            const double input_disp = bc["disp"].as<double>();

            disp_load_face load_face;
            load_face.face = input_face;
            load_face.dir = dir_map.at(input_dir);
            load_face.target_disp = input_disp;
            fespace.GetEssentialTrueDofs(face_attribute_map.at(input_face),
               load_face.dofs, load_face.dir);

            ess_tdof_list.Append(load_face.dofs);
            disp_load_list.push_back(load_face);
         }

      ess_tdof_list.Sort();
      ess_tdof_list.Unique();
   }

   // Number of load steps; defined only with displacement loading.
   int get_num_load_steps() const
   {
      MFEM_VERIFY(is_disp_load, "is_disp_load is false in config.yaml,"
         " so there are no displacement load steps.");
      return num_load_steps;
   }

   // All constrained dofs, for NonlinearForm::SetEssentialTrueDofs.
   mfem::Array<int> get_ess_tdof_list() const { return ess_tdof_list; }

   // Whether disp_bc is read.
   bool get_is_disp_load() const { return is_disp_load; }

   // Sets disp to zero on fixed faces.
   void apply_fixed_bc(mfem::GridFunction &disp) const
   {
      for (const disp_fixed_face &fixed_face : disp_fixed_list)
         for (int dof : fixed_face.dofs)
            disp(dof) = 0.0;
   }

   // Sets disp at load step n to n / N * target_disp on driven faces.
   void apply_disp_load_bc(int step, mfem::GridFunction &disp) const
   {
      const double factor = static_cast<double>(step) / num_load_steps;
      for (const disp_load_face &load_face : disp_load_list)
         for (int dof : load_face.dofs)
            disp(dof) = factor * load_face.target_disp;
   }


   // Prints all Dirichlet conditions.
   void print_Dirichlet_bc() const
   {
      mfem::out << "\nDirichlet conditions\n" << std::left
                << std::setw(10) << "face"
                << std::setw(12) << "dir"
                << std::setw(12) << "disp"
                << "dofs\n"
                << std::string(74, '-') << '\n';

      for (const disp_fixed_face &fixed_face : disp_fixed_list)
         mfem::out << std::setw(10) << fixed_face.face
                   << std::setw(12) << "xyz"[fixed_face.dir]
                   << std::setw(12) << "fixed"
                   << fixed_face.dofs.Size() << '\n';

      if (is_disp_load)
         for (const disp_load_face &load_face : disp_load_list)
            mfem::out << std::setw(10) << load_face.face
                      << std::setw(12) << "xyz"[load_face.dir]
                      << std::setw(12) << load_face.target_disp
                      << load_face.dofs.Size() << '\n';

      mfem::out << std::string(74, '-') << '\n'
                << std::setw(34) << "constrained unknowns"
                << ess_tdof_list.Size() << '\n';
      if (is_disp_load)
         mfem::out << std::setw(34) << "load steps" << num_load_steps << '\n';
      mfem::out << '\n';
   }


   // Prints load step n and the displacement of each driven face.
   void print_disp_load_by_step(int step) const
   {
      mfem::out << std::string(74, '=') << '\n'
                << "Load step " << step << " / " << num_load_steps << '\n';

      const double factor = static_cast<double>(step) / num_load_steps;
      for (const disp_load_face &load_face : disp_load_list)
         mfem::out << "  " << std::left
                   << std::setw(8) << load_face.face
                   << "displacement u" << "xyz"[load_face.dir]
                   << " = " << factor * load_face.target_disp << '\n';
   }

private:
   // One direction on a fixed face.
   struct disp_fixed_face
   {
      std::string face;          // face name
      int dir;                   // 0, 1, 2 for x, y, z
      mfem::Array<int> dofs;     // dofs of this direction on the face
   };

   // One direction on a displacement-driven face.
   struct disp_load_face
   {
      std::string face;          // face name
      int dir;                   // 0, 1, 2 for x, y, z
      double target_disp;        // displacement reached at the final step
      mfem::Array<int> dofs;     // dofs of this direction on the face
   };

   // "x", "y", "z" -> 0, 1, 2
   inline static const std::map<std::string, int> dir_map = {{"x", 0}, {"y", 1}, {"z", 2}};

   const bool is_disp_load;                       // whether disp_bc is read
   const int num_load_steps;
   mfem::Array<int> ess_tdof_list;                // union of all constrained dofs
   std::vector<disp_fixed_face> disp_fixed_list;  // one per entry of fixed_bc
   std::vector<disp_load_face> disp_load_list;    // one per entry of disp_bc
};

#endif

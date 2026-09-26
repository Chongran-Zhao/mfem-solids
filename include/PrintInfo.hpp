// ============================================================================
// PrintInfo.hpp
//
// The tables and headers printed by the programs of this project: the mesh,
// the finite element space, the Dirichlet conditions, the header of a load
// step and the saved files.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef PRINT_INFO_HPP
#define PRINT_INFO_HPP

#include "PrescribedComponent.hpp"
#include "mfem.hpp"
#include <filesystem>
#include <iomanip>
#include <string>
#include <vector>

// Mesh file, number of elements and boundary elements, and the face names.
inline void print_mesh(const std::string &mesh_file, mfem::Mesh &mesh)
{
   mfem::out << "\nMesh\n" << std::string(74, '-') << '\n' << std::left
             << std::setw(20) << "file" << std::filesystem::absolute(mesh_file).string() << '\n'
             << std::setw(20) << "elements" << mesh.GetNE() << '\n'
             << std::setw(20) << "boundary elements" << mesh.GetNBE() << '\n'
             << std::setw(20) << "faces";
   for (const std::string &name : mesh.bdr_attribute_sets.GetAttributeSetNames())
      mfem::out << name << ' ';
   mfem::out << '\n' << std::string(74, '-') << '\n';
}

// Order of the shape functions and number of unknowns, below the mesh table.
inline void print_space(int order, const mfem::FiniteElementSpace &fespace)
{
   mfem::out << std::left << std::setw(20) << "order" << order << '\n'
             << std::setw(20) << "unknowns" << fespace.GetTrueVSize() << '\n'
             << std::string(74, '-') << '\n';
}

// One row per prescribed component, and the number of constrained unknowns;
// a node on two constrained faces counts once in the total.
inline void print_dirichlet(const std::vector<PrescribedComponent> &prescribed,
                            int num_constrained)
{
   mfem::out << "\nDirichlet conditions\n" << std::left
             << std::setw(10) << "face" << std::setw(12) << "component"
             << std::setw(12) << "value" << "dofs\n" << std::string(74, '-') << '\n';
   for (const PrescribedComponent &item : prescribed)
      mfem::out << std::setw(10) << item.face << std::setw(12) << "xyz"[item.component]
                << std::setw(12) << item.value << item.num_dofs << '\n';
   mfem::out << std::string(74, '-') << '\n'
             << std::setw(34) << "constrained unknowns" << num_constrained << "\n\n";
}

// Header of load step n of N: the displacement applied on each face, which
// is factor = n / N times its final value, and the header of the Newton
// iterations.
inline void print_load_step(int step, int load_steps, double factor,
                            const std::vector<PrescribedComponent> &prescribed,
                            const std::string &length_unit)
{
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
      mfem::out << 'u' << "xyz"[item.component] << " = "
                << factor * item.value << ' ' << length_unit;
   }
   mfem::out << "\n\n"
             << std::left << std::setw(12) << "iteration" << std::setw(18) << "||R||"
             << "||R|| / ||R_0||\n";
}

// "saved  <absolute path>" for a file or folder a program has written.
inline void print_saved(const std::filesystem::path &path)
{
   mfem::out << std::left << std::setw(20) << "saved"
             << std::filesystem::absolute(path).string() << '\n';
}

#endif

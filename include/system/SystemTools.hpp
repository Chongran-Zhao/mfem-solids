// ============================================================================
// SystemTools.hpp
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef SYSTEM_TOOLS_HPP
#define SYSTEM_TOOLS_HPP

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <mfem.hpp>

class SystemTools
{
public:
   // Mesh file, number of elements and boundary elements, and the face names.
   static void print_mesh(const std::string &mesh_file, mfem::Mesh &mesh)
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

   // Finite element space.
   static void print_space(const mfem::FiniteElementSpace &fespace)
   {
      const bool by_vdim = (fespace.GetOrdering() == mfem::Ordering::byVDIM);
      mfem::out << "\nFinite Element Space\n" << std::string(74, '-') << '\n' << std::left
                << std::setw(20) << "space" << fespace.FEColl()->Name() << '\n'
                << std::setw(20) << "order" << fespace.GetMaxElementOrder() << '\n'
                << std::setw(20) << "nodes per element" << fespace.GetFE(0)->GetDof() << '\n'
                << std::setw(20) << "unknowns" << fespace.GetTrueVSize() << '\n'
                << std::setw(20) << "ordering" << (by_vdim ? "byVDIM" : "byNODES") << '\n'
                << std::string(74, '-') << '\n';
   }

   // Header of the Newton iterations printed by NewtonMonitor.
   static void print_newton_header()
   {
      mfem::out << '\n' << std::left << std::setw(12) << "iteration" << std::setw(18) << "||R||"
                << "||R|| / ||R_0||\n";
   }

   // Prints the residual norm of every Newton iteration, absolute and
   // relative to the first iteration of the load step. NewtonSolver calls
   // MonitorResidual once per iteration, and once more with final = true,
   // which is skipped.
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

   // Creates an empty folder; an existing one is emptied first, so that no
   // files of an earlier run are left.
   static void make_empty_dir(const std::filesystem::path &dir)
   {
      std::filesystem::remove_all(dir);
      std::filesystem::create_directories(dir);
   }

   // Saves a grid function of load step n as <dir>/<name>_XXXX.gf, with 16
   // significant digits, e.g. results_gf/disp_0025.gf.
   static void save_gf(const std::filesystem::path &dir, const std::string &name,
                       int step, const mfem::GridFunction &gf)
   {
      std::ostringstream file_name;
      file_name << name << '_' << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ofstream gf_file(dir / file_name.str());
      gf_file.precision(16);
      gf.Save(gf_file);
   }

   // "saved  <absolute path>" for a file or folder a program has written.
   static void print_saved(const std::filesystem::path &path)
   {
      mfem::out << std::left << std::setw(20) << "saved"
                << std::filesystem::absolute(path).string() << '\n';
   }
};

#endif

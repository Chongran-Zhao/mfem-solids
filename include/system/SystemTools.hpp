// ============================================================================
// SystemTools.hpp
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef SYSTEM_TOOLS_HPP
#define SYSTEM_TOOLS_HPP

#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

#include <mfem.hpp>

class SystemTools
{
public:
   // Present local time as HH:MM:SS.
   static std::string get_time()
   {
      return format_now("%H:%M:%S");
   }

   // Present local date as YYYY-MM-DD.
   static std::string get_date()
   {
      return format_now("%Y-%m-%d");
   }

   // Mesh file, number of elements and boundary elements, and the face names
   // of the serial mesh; with MPI, also the number of ranks it is split into.
   static void print_mesh(const std::string &mesh_file, mfem::Mesh &mesh)
   {
      mfem::out << "\nMesh\n" << std::string(74, '-') << '\n' << std::left
                << std::setw(20) << "file" << std::filesystem::absolute(mesh_file).string() << '\n'
                << std::setw(20) << "elements" << mesh.GetNE() << '\n'
                << std::setw(20) << "boundary elements" << mesh.GetNBE() << '\n'
                << std::setw(20) << "faces";
      for (const std::string &name : mesh.bdr_attribute_sets.GetAttributeSetNames())
         mfem::out << name << ' ';
      mfem::out << '\n';
      if (mfem::Mpi::IsInitialized())
         mfem::out << std::setw(20) << "MPI ranks" << mfem::Mpi::WorldSize() << '\n';
      mfem::out << std::string(74, '-') << '\n';
   }

   // Finite element space. The element is named by its polynomial space,
   // e.g. Q2 hexahedron, since the name of an MFEM collection, e.g.
   // H1_3D_P2, gives the order only. The ordering is left out for a scalar
   // field, where byNODES and byVDIM are the same. The unknowns of a
   // parallel space are those of all ranks; it is then collective.
   static void print_space(const mfem::FiniteElementSpace &fespace)
   {
      const bool by_vdim = (fespace.GetOrdering() == mfem::Ordering::byVDIM);
      const auto *par_fespace = dynamic_cast<const mfem::ParFiniteElementSpace *>(&fespace);
      const long long num_unknowns = par_fespace ? par_fespace->GlobalTrueVSize()
                                                 : fespace.GetTrueVSize();
      mfem::out << "\nFinite Element Space\n" << std::string(74, '-') << '\n' << std::left
                << std::setw(20) << "element" << get_element_name(fespace) << '\n'
                << std::setw(20) << "components" << fespace.GetVDim() << '\n'
                << std::setw(20) << "nodes per element" << fespace.GetFE(0)->GetDof() << '\n'
                << std::setw(20) << "unknowns" << num_unknowns << '\n';
      if (fespace.GetVDim() > 1)
         mfem::out << std::setw(20) << "ordering" << (by_vdim ? "byVDIM" : "byNODES") << '\n';
      mfem::out << std::string(74, '-') << '\n';
   }

   // Q_k on a hexahedron, degree <= k in each variable; P_k on a
   // tetrahedron, total degree <= k. Any other element keeps the name of its
   // collection.
   static std::string get_element_name(const mfem::FiniteElementSpace &fespace)
   {
      const int order = fespace.GetMaxElementOrder();
      const bool is_h1 = dynamic_cast<const mfem::H1_FECollection *>(fespace.FEColl()) != nullptr;
      if (is_h1)
         switch (fespace.GetFE(0)->GetGeomType())
         {
            case mfem::Geometry::CUBE:        return "Q" + std::to_string(order) + " hexahedron";
            case mfem::Geometry::TETRAHEDRON: return "P" + std::to_string(order) + " tetrahedron";
            default: break;
         }
      return fespace.FEColl()->Name();
   }

   // Header of the Newton iterations printed by NewtonMonitor.
   static void print_newton_header()
   {
      mfem::out << '\n' << std::left << std::setw(12) << "iteration" << std::setw(18) << "||R||"
                << "||R|| / ||R_0||\n";
   }

   // Header of the Newton iterations printed by BlockNewtonMonitor.
   static void print_block_newton_header()
   {
      mfem::out << '\n' << std::left << std::setw(11) << "iteration"
                << std::setw(15) << "||R_u||" << std::setw(15) << "/ ||R_u,0||"
                << std::setw(15) << "||R_p||" << "/ ||R_p,0||\n";
   }

   // Prints the residual norm of every Newton iteration, absolute and
   // relative to the first iteration of the load step. NewtonSolver calls
   // MonitorResidual once per iteration, and once more with final = true,
   // which is skipped.
   class NewtonMonitor : public mfem::IterativeSolverMonitor
   {
   public:
      // Required by MFEM: overrides mfem::IterativeSolverMonitor::
      // MonitorResidual, which NewtonSolver calls at every iteration.
      void MonitorResidual(int it, mfem::real_t norm, const mfem::Vector &,
                           bool final) override
      {
         if (final)
            return;
         if (it == 0)
            initial_norm = norm;
         mfem::out << std::left << std::setw(12) << it << std::scientific
                   << std::setprecision(6) << std::setw(18) << norm
                   << (initial_norm > 0.0 ? norm / initial_norm : 0.0)
                   << std::defaultfloat << '\n';
      }

   private:
      double initial_norm = 1.0;
   };

   // As NewtonMonitor, for a residual of two blocks, the displacement R_u
   // and the pressure R_p, whose scales differ by orders of magnitude: the
   // norm of each block, absolute and relative to the first iteration of the
   // load step. A relative norm is left out when its first norm is zero. The
   // residual is that of this rank, on the offsets of its blocks; the norms
   // are over all ranks of comm.
   class BlockNewtonMonitor : public mfem::IterativeSolverMonitor
   {
   public:
      BlockNewtonMonitor(const mfem::Array<int> &input_offsets, MPI_Comm input_comm)
         : offsets(input_offsets), comm(input_comm) {}

      // Required by MFEM: overrides mfem::IterativeSolverMonitor::
      // MonitorResidual, which NewtonSolver calls at every iteration.
      void MonitorResidual(int it, mfem::real_t, const mfem::Vector &r,
                           bool final) override
      {
         if (final)
            return;

         // ||R_u|| and ||R_p||, the norms of r over the two blocks.
         std::array<double, 2> norm = {0.0, 0.0};
         for (int bb = 0; bb < 2; bb++)
            for (int ii = offsets[bb]; ii < offsets[bb + 1]; ii++)
               norm[bb] += r(ii) * r(ii);
         MPI_Allreduce(MPI_IN_PLACE, norm.data(), 2, MPI_DOUBLE, MPI_SUM, comm);
         for (int bb = 0; bb < 2; bb++)
            norm[bb] = std::sqrt(norm[bb]);
         if (it == 0)
            initial_norm = norm;

         mfem::out << std::left << std::setw(11) << it << std::scientific << std::setprecision(6);
         for (int bb = 0; bb < 2; bb++)
         {
            // No padding after the last column.
            const int width = (bb == 0) ? 15 : 0;
            mfem::out << std::setw(15) << norm[bb];
            if (initial_norm[bb] > 0.0)
               mfem::out << std::setw(width) << norm[bb] / initial_norm[bb];
            else
               mfem::out << std::setw(width) << "-";
         }
         mfem::out << std::defaultfloat << '\n';
      }

   private:
      const mfem::Array<int> offsets;             // [0, n_u, n_u + n_p] of this rank
      const MPI_Comm comm;                        // the ranks of the residual
      std::array<double, 2> initial_norm = {1.0, 1.0};
   };

   // Creates an empty folder on rank 0; an existing one is emptied first, so
   // that no files of an earlier run are left. Collective: the other ranks
   // wait until it exists.
   static void make_empty_dir(const std::filesystem::path &dir)
   {
      if (mfem::Mpi::Root())
      {
         std::filesystem::remove_all(dir);
         std::filesystem::create_directories(dir);
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Saves serial_mesh, the mesh of all ranks that ParMesh::GetSerialMesh(0)
   // gathers on rank 0, as <dir>/mesh.mesh, with the named faces of pmesh;
   // the saved grid functions live on it, in its element order, which is
   // that of the ranks and not the one of the mesh file read by the driver.
   static void save_serial_mesh(const std::filesystem::path &dir, const mfem::ParMesh &pmesh,
                                mfem::Mesh &serial_mesh)
   {
      if (!mfem::Mpi::Root())
         return;
      pmesh.bdr_attribute_sets.Copy(serial_mesh.bdr_attribute_sets);
      std::ofstream mesh_file(dir / "mesh.mesh");
      mesh_file.precision(16);
      serial_mesh.Print(mesh_file);
   }

   // Gathers a grid function of load step n onto serial_mesh and saves it on
   // rank 0 as <dir>/<name>_XXXX.gf, with 16 significant digits, e.g.
   // results_gf/disp_0025.gf. Collective.
   static void save_gf(const std::filesystem::path &dir, const std::string &name,
                       int step, const mfem::ParGridFunction &gf, mfem::Mesh &serial_mesh)
   {
      const mfem::GridFunction serial_gf = gf.GetSerialGridFunction(0, serial_mesh);
      if (!mfem::Mpi::Root())
         return;
      std::ostringstream file_name;
      file_name << name << '_' << std::setw(4) << std::setfill('0') << step << ".gf";
      std::ofstream gf_file(dir / file_name.str());
      gf_file.precision(16);
      serial_gf.Save(gf_file);
   }

   // Present local time in a strftime format.
   static std::string format_now(const char *format)
   {
      const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      std::ostringstream out;
      out << std::put_time(std::localtime(&now), format);
      return out.str();
   }

   // "saved  <absolute path>" for a file or folder a program has written.
   static void print_saved(const std::filesystem::path &path)
   {
      mfem::out << std::left << std::setw(20) << "saved"
                << std::filesystem::absolute(path).string() << '\n';
   }
};

#endif

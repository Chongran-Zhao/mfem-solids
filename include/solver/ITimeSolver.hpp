// ============================================================================
// ITimeSolver.hpp
//
// Interface of the time solvers of the cases, each the loop over the load or
// time steps of its case, in the TimeSolver.hpp of the case folder; it holds
// what they share: the results folder, which rank 0 empties first, with the
// serial mesh of ParMesh::GetSerialMesh saved as <results>/mesh.mesh, whose
// element order is that of the ranks, not that of the mesh file read by the
// driver; the saving of the fields of a step onto that mesh; the line
// printed when a step converges; and, for the dynamics, the number of time
// steps and <results>/time.csv.
//
// Author: Chongran Zhao
// Date: Oct. 14, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef ITIME_SOLVER_HPP
#define ITIME_SOLVER_HPP

#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <utility>

#include <mfem.hpp>

#include "SystemTools.hpp"

class ITimeSolver
{
public:
   virtual ~ITimeSolver() = default;

protected:
   // The results go to results_dir, which rank 0 empties first and where it
   // saves the serial mesh of mesh, with its named faces.
   ITimeSolver(const std::filesystem::path &input_results_dir, mfem::ParMesh &mesh)
      : results_dir(input_results_dir),
        serial_mesh(mesh.GetSerialMesh(0))
   {
      if (mfem::Mpi::Root())
      {
         SystemTools::make_empty_dir(results_dir);
         mesh.bdr_attribute_sets.Copy(serial_mesh.bdr_attribute_sets);
         std::ofstream mesh_file(results_dir / "mesh.mesh");
         mesh_file.precision(16);
         serial_mesh.Print(mesh_file);
      }
      MPI_Barrier(mesh.GetComm());
   }

   // Gathers the fields of a step onto serial_mesh and saves them on rank 0
   // as <results>/<name>_XXXX.gf; every rank takes part.
   void save_fields(int step,
                    std::initializer_list<std::pair<const char *, const mfem::ParGridFunction *>> fields)
   {
      for (const auto &[name, field] : fields)
      {
         const mfem::GridFunction serial_field = field->GetSerialGridFunction(0, serial_mesh);
         if (mfem::Mpi::Root())
            SystemTools::save_gf(results_dir, name, step, serial_field);
      }
   }

   // Prints the Newton iterations of a converged step and its wall-clock
   // time since step_timer was restarted.
   static void print_converged(int iterations, mfem::StopWatch &step_timer)
   {
      mfem::out << "converged in " << iterations
                << " iterations. Time taken: " << std::fixed << std::setprecision(2)
                << step_timer.RealTime() << " sec. " << SystemTools::get_time()
                << std::defaultfloat << std::setprecision(6) << '\n';
   }

   // Number of the time steps of nominal_dt up to final_time, the last one
   // shortened to end there; a last step shorter than 1e-12 dt is left out.
   static int get_num_steps(double final_time, double nominal_dt)
   {
      return static_cast<int>(std::ceil(final_time / nominal_dt - 1.0e-12));
   }

   // <results>/time.csv on rank 0, with its header and step 0; not open
   // elsewhere.
   std::ofstream open_time_file() const
   {
      std::ofstream time_file;
      if (mfem::Mpi::Root())
      {
         time_file.open(results_dir / "time.csv");
         time_file << "step,time,dt,iterations\n" << std::scientific << std::setprecision(16)
                   << 0 << ',' << 0.0 << ',' << 0.0 << ',' << 0 << '\n';
      }
      return time_file;
   }

   // One row of time.csv, written on rank 0.
   static void write_time(std::ofstream &time_file, int step, double time, double dt,
                          int iterations)
   {
      if (mfem::Mpi::Root())
         time_file << step << ',' << time << ',' << dt << ',' << iterations << '\n';
   }

   const std::filesystem::path results_dir;   // folder of the results
   mfem::Mesh serial_mesh;                    // all elements on rank 0, empty elsewhere
};

#endif

// ============================================================================
// VTK_Tools.hpp
//
// Writes one VTU file per load step on the deformed mesh, and a PVD file
// listing them, for ParaView.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef VTK_TOOLS_HPP
#define VTK_TOOLS_HPP

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <mfem.hpp>
#include "LocalAssemblyTools.hpp"

class VTK_Tools
{
public:
   // Create the output folder.
   VTK_Tools(const std::string &input_dir) : dir(input_dir)
   {
      std::filesystem::create_directories(dir);
   }

   // Write step_XXXX.vtu and update the PVD file. pres is nodal in H1 from
   // driver_static_mixed, or p(J) at the element centers from driver_static_displacement;
   // stress holds P at the element centers.
   void save(int step, double time, const mfem::GridFunction &disp,
             const mfem::GridFunction &pres, const mfem::GridFunction &stress)
   {
      const mfem::FiniteElementSpace &fespace = *disp.FESpace();
      const mfem::Mesh &mesh = *fespace.GetMesh();
      // Only the vertex values are written; in an H1 space of any order, the
      // first dofs are those of the vertices.
      MFEM_VERIFY(dynamic_cast<const mfem::H1_FECollection *>(fespace.FEColl()),
                  "VTK_Tools needs the displacement in an H1 space.");
      const bool is_nodal_pres =
         dynamic_cast<const mfem::H1_FECollection *>(pres.FESpace()->FEColl()) != nullptr;

      std::ostringstream file_name;
      file_name << "step_" << std::setw(4) << std::setfill('0') << step << ".vtu";

      std::ofstream out(dir / file_name.str());
      out << std::setprecision(10)
          << "<?xml version=\"1.0\"?>\n"
          << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
          << "<UnstructuredGrid>\n"
          << "<Piece NumberOfPoints=\"" << mesh.GetNV()
          << "\" NumberOfCells=\"" << mesh.GetNE() << "\">\n";

      // Displacement at the vertices.
      out << "<PointData Vectors=\"displacement\">\n";
      begin_array(out, "displacement", {"x", "y", "z"});
      for (int vv = 0; vv < mesh.GetNV(); vv++)
         out << disp(fespace.DofToVDof(vv, 0)) << ' ' << disp(fespace.DofToVDof(vv, 1))
             << ' ' << disp(fespace.DofToVDof(vv, 2)) << '\n';
      out << "</DataArray>\n";

      // Nodal pressure at the vertices.
      if (is_nodal_pres)
      {
         begin_array(out, "pressure", {"p"});
         for (int vv = 0; vv < mesh.GetNV(); vv++)
            out << pres(vv) << '\n';
         out << "</DataArray>\n";
      }
      out << "</PointData>\n";

      // Stresses at the element centers: P from the drivers, S = F^-1 P.
      std::vector<Tensor2_3D> PK1(mesh.GetNE()), PK2(mesh.GetNE());
      const mfem::FiniteElementSpace &space_stress = *stress.FESpace();
      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         for (int ii = 0; ii < 3; ii++)
            for (int JJ = 0; JJ < 3; JJ++)
               PK1[ee](ii, JJ) = stress(space_stress.DofToVDof(ee, 3 * ii + JJ));
         const Tensor2_3D F = LocalAssemblyTools::get_center_deformation_gradient(fespace, disp, ee);
         PK2[ee] = F.inverse() * PK1[ee];
      }

      out << "<CellData>\n";

      // Pressure at the element centers.
      if (!is_nodal_pres)
      {
         begin_array(out, "pressure", {"p"});
         for (int ee = 0; ee < mesh.GetNE(); ee++)
            out << pres(ee) << '\n';
         out << "</DataArray>\n";
      }

      begin_array(out, "first_PK_stress", {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"});
      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         for (int ii = 0; ii < 3; ii++)
            for (int JJ = 0; JJ < 3; JJ++)
               out << PK1[ee](ii, JJ) << ' ';
         out << '\n';
      }
      out << "</DataArray>\n";

      // Only the 6 independent components, in ParaView's order.
      begin_array(out, "second_PK_stress", {"xx", "yy", "zz", "xy", "yz", "xz"});
      for (int ee = 0; ee < mesh.GetNE(); ee++)
         out << PK2[ee](0, 0) << ' ' << PK2[ee](1, 1) << ' ' << PK2[ee](2, 2) << ' '
             << PK2[ee](0, 1) << ' ' << PK2[ee](1, 2) << ' ' << PK2[ee](0, 2) << '\n';
      out << "</DataArray>\n</CellData>\n";

      // Deformed vertex positions x = X + u.
      out << "<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
      for (int vv = 0; vv < mesh.GetNV(); vv++)
      {
         const double *coord = mesh.GetVertex(vv);
         for (int comp = 0; comp < 3; comp++)
            out << coord[comp] + disp(fespace.DofToVDof(vv, comp)) << ' ';
         out << '\n';
      }
      out << "</DataArray>\n</Points>\n";

      // Elements; MFEM and VTK order the vertices of linear elements the
      // same way.
      mfem::Array<int> elem_vertices;
      std::ostringstream connectivity, offsets, types;
      int offset = 0;
      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         mesh.GetElementVertices(ee, elem_vertices);
         for (int vv : elem_vertices)
            connectivity << vv << ' ';
         connectivity << '\n';
         offset += elem_vertices.Size();
         offsets << offset << '\n';
         types << get_vtk_cell_type(mesh.GetElementGeometry(ee)) << '\n';
      }
      out << "<Cells>\n"
          << "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n"
          << connectivity.str() << "</DataArray>\n"
          << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n"
          << offsets.str() << "</DataArray>\n"
          << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n"
          << types.str() << "</DataArray>\n"
          << "</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";

      steps.emplace_back(time, file_name.str());
      write_pvd();
   }

   // Path of the PVD file, the one to open in ParaView.
   std::filesystem::path get_pvd_path() const
   {
      return dir / (dir.filename().string() + ".pvd");
   }

private:
   std::filesystem::path dir;
   std::vector<std::pair<double, std::string>> steps; // time and file of each step

   // Opening tag of a Float64 array with named components.
   static void begin_array(std::ofstream &out, const std::string &name,
                           const std::vector<std::string> &component_names)
   {
      out << "<DataArray type=\"Float64\" Name=\"" << name
          << "\" NumberOfComponents=\"" << component_names.size() << '"';
      for (std::size_t cc = 0; cc < component_names.size(); cc++)
         out << " ComponentName" << cc << "=\"" << component_names[cc] << '"';
      out << " format=\"ascii\">\n";
   }

   static int get_vtk_cell_type(mfem::Geometry::Type geom)
   {
      switch (geom)
      {
         case mfem::Geometry::CUBE:        return 12; // VTK_HEXAHEDRON
         case mfem::Geometry::TETRAHEDRON: return 10; // VTK_TETRA
         case mfem::Geometry::PRISM:       return 13; // VTK_WEDGE
         default: MFEM_ABORT("Unsupported element geometry for VTU output.");
      }
      return 0;
   }

   // List every step with its time.
   void write_pvd() const
   {
      std::ofstream out(get_pvd_path());
      out << std::setprecision(10)
          << "<?xml version=\"1.0\"?>\n"
          << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
          << "<Collection>\n";
      for (const auto &[time, file] : steps)
         out << "<DataSet timestep=\"" << time << "\" file=\"" << file << "\"/>\n";
      out << "</Collection>\n</VTKFile>\n";
   }
};

#endif

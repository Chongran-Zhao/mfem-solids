// ============================================================================
// VTK_Tools.hpp
//
// include/visualization/VTK_Tools.hpp for sphere_rotation, with the
// Green-Lagrange strain besides the stresses: writes one VTU file per time
// step on the deformed mesh, and a PVD file listing them, for ParaView. Its
// vtu_writer.cpp, in this folder, includes it instead of the other one.
//
// Author: Chongran Zhao
// Date: Oct. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef VTK_TOOLS_HPP
#define VTK_TOOLS_HPP

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mfem.hpp>

#include "LocalAssemblyTools.hpp"
#include "Tensor2_3D.hpp"

class VTK_Tools
{
public:
   // Create the output folder.
   VTK_Tools(const std::string &input_dir) : dir(input_dir)
   {
      std::filesystem::create_directories(dir);
   }

   // Write step_XXXX.vtu and update the PVD file. The velocity and the
   // acceleration are in the space of disp, written at the vertices; pres
   // is nodal in H1, or one value per element; PK1 and PK2, the first and
   // second Piola-Kirchhoff stresses, and E, the Green-Lagrange strain,
   // have one value per element.
   void save(int step, double time, const mfem::GridFunction &disp,
             const mfem::GridFunction &velo, const mfem::GridFunction &acce,
             const mfem::GridFunction &pres, const std::vector<Tensor2_3D> &PK1,
             const std::vector<Tensor2_3D> &PK2, const std::vector<Tensor2_3D> &E)
   {
      write_step(step, time, disp, {{"velocity", &velo}, {"acceleration", &acce}},
                 pres, PK1, PK2, E);
   }

   // F at the center of element ee, for the output of the stress.
   static Tensor2_3D get_center_deformation_gradient(const mfem::FiniteElementSpace &fespace,
                                                     const mfem::GridFunction &disp, int ee)
   {
      const mfem::FiniteElement &elem = *fespace.GetFE(ee);
      mfem::ElementTransformation &elem_map = *fespace.GetElementTransformation(ee);
      const mfem::IntegrationPoint &center = mfem::Geometries.GetCenter(elem.GetGeomType());
      elem_map.SetIntPoint(&center);

      const int num_nodes = elem.GetDof();
      mfem::DenseMatrix dN_dxi(num_nodes, 3), dN_dX(num_nodes, 3);
      elem.CalcDShape(center, dN_dxi);
      mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);

      // Element displacement: x of all nodes, then y, then z.
      mfem::Array<int> vdofs;
      mfem::Vector elem_disp;
      fespace.GetElementVDofs(ee, vdofs);
      disp.GetSubVector(vdofs, elem_disp);

      return LocalAssemblyTools::get_deformation_gradient(elem_disp, dN_dX);
   }

   // Path of the PVD file, the one to open in ParaView.
   std::filesystem::path get_pvd_path() const
   {
      return dir / (dir.filename().string() + ".pvd");
   }

private:
   std::filesystem::path dir;
   std::vector<std::pair<double, std::string>> steps; // time and file of each step

   // step_XXXX.vtu of save, with the named nodal_vectors, in the space of
   // disp, written at the vertices after the displacement.
   void write_step(int step, double time, const mfem::GridFunction &disp,
                   const std::vector<std::pair<std::string, const mfem::GridFunction *>> &nodal_vectors,
                   const mfem::GridFunction &pres, const std::vector<Tensor2_3D> &PK1,
                   const std::vector<Tensor2_3D> &PK2, const std::vector<Tensor2_3D> &E)
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

      // Displacement, and the other nodal vectors, at the vertices.
      out << "<PointData Vectors=\"displacement\">\n";
      write_vertex_vector(out, "displacement", disp);
      for (const auto &[name, field] : nodal_vectors)
         write_vertex_vector(out, name, *field);

      // Nodal pressure at the vertices.
      if (is_nodal_pres)
      {
         begin_array(out, "pressure", {"p"});
         for (int vv = 0; vv < mesh.GetNV(); vv++)
            out << pres(vv) << '\n';
         out << "</DataArray>\n";
      }
      out << "</PointData>\n";

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
      out << "</DataArray>\n";

      // Symmetric too: the 6 independent components, in ParaView's order.
      begin_array(out, "Green_Lagrange_strain", {"xx", "yy", "zz", "xy", "yz", "xz"});
      for (int ee = 0; ee < mesh.GetNE(); ee++)
         out << E[ee](0, 0) << ' ' << E[ee](1, 1) << ' ' << E[ee](2, 2) << ' '
             << E[ee](0, 1) << ' ' << E[ee](1, 2) << ' ' << E[ee](0, 2) << '\n';
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

   // A vector field of the space of disp at the vertices.
   static void write_vertex_vector(std::ofstream &out, const std::string &name,
                                   const mfem::GridFunction &field)
   {
      const mfem::FiniteElementSpace &fespace = *field.FESpace();
      begin_array(out, name, {"x", "y", "z"});
      for (int vv = 0; vv < fespace.GetMesh()->GetNV(); vv++)
         out << field(fespace.DofToVDof(vv, 0)) << ' ' << field(fespace.DofToVDof(vv, 1))
             << ' ' << field(fespace.DofToVDof(vv, 2)) << '\n';
      out << "</DataArray>\n";
   }

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

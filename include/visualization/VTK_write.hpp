// ============================================================================
// VTK_write.hpp
//
// Writes one ASCII VTU file per step and a PVD file listing them, for
// ParaView. The mesh is written in the deformed configuration x = X + u, with
// the displacement at the vertices and the first and second Piola-Kirchhoff
// stresses at the element centers. Components are named x, y, z and xx, xy,
// ... instead of 0, 1, 2.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef VTK_WRITE_HPP
#define VTK_WRITE_HPP

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <mfem.hpp>
#include "HyperelasticMaterialModel.hpp"

class VTK_write
{
public:
   // The files are written into input_dir, which is created if needed.
   VTK_write(const std::string &input_dir) : dir(input_dir)
   {
      std::filesystem::create_directories(dir);
   }

   // Writes step_<step>.vtu and rewrites the PVD file with all steps so far.
   void save(int step, double time, mfem::FiniteElementSpace &fespace,
             const mfem::GridFunction &disp, const HyperelasticMaterialModel &material)
   {
      mfem::Mesh &mesh = *fespace.GetMesh();
      MFEM_VERIFY(fespace.GetMaxElementOrder() == 1,
                  "VTK_write writes the vertices only, so it needs order 1.");

      std::ostringstream file_name;
      file_name << "step_" << std::setw(4) << std::setfill('0') << step << ".vtu";

      std::ofstream out(dir / file_name.str());
      out << std::setprecision(10)
          << "<?xml version=\"1.0\"?>\n"
          << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
          << "<UnstructuredGrid>\n"
          << "<Piece NumberOfPoints=\"" << mesh.GetNV()
          << "\" NumberOfCells=\"" << mesh.GetNE() << "\">\n";

      // Displacement at the vertices; with linear elements the dofs of a
      // scalar field are the vertices.
      out << "<PointData Vectors=\"displacement\">\n";
      begin_array(out, "displacement", {"x", "y", "z"});
      for (int vv = 0; vv < mesh.GetNV(); vv++)
         out << disp(fespace.DofToVDof(vv, 0)) << ' ' << disp(fespace.DofToVDof(vv, 1))
             << ' ' << disp(fespace.DofToVDof(vv, 2)) << '\n';
      out << "</DataArray>\n</PointData>\n";

      // Stresses at the element centers.
      std::vector<Tensor2_3D> PK1(mesh.GetNE()), PK2(mesh.GetNE());
      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const Tensor2_3D F = get_center_deformation_gradient(fespace, disp, ee);
         PK2[ee] = material.get_2nd_PK_stress(F);
         PK1[ee] = F * PK2[ee];
      }

      out << "<CellData>\n";
      begin_array(out, "first_PK_stress", {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"});
      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         for (int ii = 0; ii < 3; ii++)
            for (int JJ = 0; JJ < 3; JJ++)
               out << PK1[ee](ii, JJ) << ' ';
         out << '\n';
      }
      out << "</DataArray>\n";

      // Symmetric, so only the 6 independent components, in ParaView's order.
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

      // Elements: vertex numbers, the running end of each element in that
      // list, and the VTK cell type. MFEM and VTK order the vertices of
      // linear hexahedra, tetrahedra and wedges the same way.
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

   // F_kJ = delta_kJ + d_ak N_a,J at the center of element ee.
   static Tensor2_3D get_center_deformation_gradient(mfem::FiniteElementSpace &fespace,
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

      // Element vector ordered by component: x of all nodes, then y, then z.
      mfem::Array<int> vdofs;
      mfem::Vector elem_disp;
      fespace.GetElementVDofs(ee, vdofs);
      disp.GetSubVector(vdofs, elem_disp);

      Tensor2_3D F = Tensor2_3D::identity();
      for (int kk = 0; kk < 3; kk++)
         for (int JJ = 0; JJ < 3; JJ++)
            for (int aa = 0; aa < num_nodes; aa++)
               F(kk, JJ) += elem_disp(aa + kk * num_nodes) * dN_dX(aa, JJ);
      return F;
   }

   // Lists every step with its time; ParaView plays them in this order.
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

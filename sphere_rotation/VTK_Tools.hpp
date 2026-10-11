// ============================================================================
// VTK_Tools.hpp
//
// include/visualization/VTK_Tools.hpp for sphere_rotation, sampled within
// each element: writes one VTU file per time step on the deformed mesh, and
// a PVD file listing them, for ParaView. Each element is cut into refine^3
// linear hexahedra at (refine + 1)^3 evenly spaced points of its reference
// cube, as the visualization of MixPERIGEE does, so that the curved
// geometry and the fields of the higher-order elements show within it; the
// points of an element are its own, not shared with its neighbors. At each
// point it writes the displacement, the velocity, the acceleration, the
// pressure, the first and second Piola-Kirchhoff stresses and the
// Green-Lagrange strain, and for each hexahedron the region of its element,
// the element attribute. The arrays are Float32 and Int32, written in
// binary, base64 encoded. Its vtu_writer.cpp, in this folder, includes it
// instead of the other one.
//
// Author: Chongran Zhao
// Date: Oct. 11, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef VTK_TOOLS_HPP
#define VTK_TOOLS_HPP

#include <cstddef>
#include <cstdint>
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
   // Create the output folder; refine is the number of hexahedra per edge
   // of an element, at least 1.
   VTK_Tools(const std::string &input_dir, int input_refine)
      : dir(input_dir), refine(input_refine)
   {
      MFEM_VERIFY(refine >= 1, "vtu_refine must be at least 1.");
      std::filesystem::create_directories(dir);
   }

   // Write step_XXXX.vtu and update the PVD file. velo and acce are in the
   // space of disp, pres is nodal in H1; first_PK_stress(attribute, F, p)
   // gives P of the material of the element attribute.
   template <typename Stress>
   void save(int step, double time, const mfem::GridFunction &disp,
             const mfem::GridFunction &velo, const mfem::GridFunction &acce,
             const mfem::GridFunction &pres, const Stress &first_PK_stress)
   {
      const mfem::FiniteElementSpace &fespace = *disp.FESpace();
      mfem::Mesh &mesh = *fespace.GetMesh();

      // Point data, Float32, and cell data of the hexahedra.
      std::vector<float> points, disp_out, velo_out, acce_out, pres_out, PK1_out, PK2_out,
         E_out;
      std::vector<std::int32_t> connectivity, offsets, regions;
      std::vector<std::uint8_t> types;

      mfem::Array<int> vdofs;
      mfem::Vector elem_disp, coord(3), value(3);
      mfem::DenseMatrix dN_dxi, dN_dX;
      for (int ee = 0; ee < mesh.GetNE(); ee++)
      {
         const mfem::FiniteElement &elem = *fespace.GetFE(ee);
         mfem::ElementTransformation &elem_map = *fespace.GetElementTransformation(ee);
         const mfem::RefinedGeometry &refined =
            *mfem::GlobGeometryRefiner.Refine(elem.GetGeomType(), refine);
         const int attribute = mesh.GetAttribute(ee);

         const int num_nodes = elem.GetDof();
         dN_dxi.SetSize(num_nodes, 3);
         dN_dX.SetSize(num_nodes, 3);
         fespace.GetElementVDofs(ee, vdofs);
         disp.GetSubVector(vdofs, elem_disp);

         // The points of this element, numbered from first_point.
         const int first_point = static_cast<int>(points.size() / 3);
         for (int qq = 0; qq < refined.RefPts.GetNPoints(); qq++)
         {
            const mfem::IntegrationPoint &ref_pt = refined.RefPts.IntPoint(qq);
            elem_map.SetIntPoint(&ref_pt);

            // Deformed position x = X + u.
            elem_map.Transform(ref_pt, coord);
            disp.GetVectorValue(elem_map, ref_pt, value);
            for (int comp = 0; comp < 3; comp++)
            {
               points.push_back(static_cast<float>(coord(comp) + value(comp)));
               disp_out.push_back(static_cast<float>(value(comp)));
            }
            velo.GetVectorValue(elem_map, ref_pt, value);
            for (int comp = 0; comp < 3; comp++)
               velo_out.push_back(static_cast<float>(value(comp)));
            acce.GetVectorValue(elem_map, ref_pt, value);
            for (int comp = 0; comp < 3; comp++)
               acce_out.push_back(static_cast<float>(value(comp)));
            const double p = pres.GetValue(elem_map, ref_pt);
            pres_out.push_back(static_cast<float>(p));

            // F, then P, S = F^-1 P and E = 1/2 (F^T F - I) at the point.
            elem.CalcDShape(ref_pt, dN_dxi);
            mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
            const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(elem_disp, dN_dX);
            const Tensor2_3D PK1 = first_PK_stress(attribute, F, p);
            const Tensor2_3D PK2 = F.inverse() * PK1;
            const Tensor2_3D E = 0.5 * (F.transpose() * F - Tensor2_3D::identity());
            for (int ii = 0; ii < 3; ii++)
               for (int JJ = 0; JJ < 3; JJ++)
                  PK1_out.push_back(static_cast<float>(PK1(ii, JJ)));
            append_symmetric(PK2_out, PK2);
            append_symmetric(E_out, E);
         }

         // The hexahedra of this element; MFEM and VTK order their vertices
         // the same way.
         for (int hh = 0; hh < refined.RefGeoms.Size() / 8; hh++)
         {
            for (int vv = 0; vv < 8; vv++)
               connectivity.push_back(first_point + refined.RefGeoms[8 * hh + vv]);
            offsets.push_back(static_cast<std::int32_t>(connectivity.size()));
            types.push_back(12);   // VTK_HEXAHEDRON
            regions.push_back(attribute);
         }
      }

      std::ostringstream file_name;
      file_name << "step_" << std::setw(4) << std::setfill('0') << step << ".vtu";

      std::ofstream out(dir / file_name.str());
      out << "<?xml version=\"1.0\"?>\n"
          << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\""
          << " header_type=\"UInt32\">\n"
          << "<UnstructuredGrid>\n"
          << "<Piece NumberOfPoints=\"" << points.size() / 3
          << "\" NumberOfCells=\"" << types.size() << "\">\n";

      out << "<PointData Vectors=\"displacement\" Scalars=\"pressure\">\n";
      write_array(out, "Float32", "displacement", {"x", "y", "z"}, disp_out);
      write_array(out, "Float32", "velocity", {"x", "y", "z"}, velo_out);
      write_array(out, "Float32", "acceleration", {"x", "y", "z"}, acce_out);
      write_array(out, "Float32", "pressure", {"p"}, pres_out);
      write_array(out, "Float32", "first_PK_stress",
                  {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"}, PK1_out);
      // Only the 6 independent components of the symmetric tensors, in
      // ParaView's order.
      write_array(out, "Float32", "second_PK_stress", {"xx", "yy", "zz", "xy", "yz", "xz"},
                  PK2_out);
      write_array(out, "Float32", "Green_Lagrange_strain", {"xx", "yy", "zz", "xy", "yz", "xz"},
                  E_out);
      out << "</PointData>\n";

      out << "<CellData Scalars=\"region\">\n";
      write_array(out, "Int32", "region", {}, regions);
      out << "</CellData>\n";

      out << "<Points>\n";
      write_array(out, "Float32", "", {"x", "y", "z"}, points);
      out << "</Points>\n";

      out << "<Cells>\n";
      write_array(out, "Int32", "connectivity", {}, connectivity);
      write_array(out, "Int32", "offsets", {}, offsets);
      write_array(out, "UInt8", "types", {}, types);
      out << "</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";

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
   const int refine;                                   // hexahedra per edge of an element
   std::vector<std::pair<double, std::string>> steps;  // time and file of each step

   // The 6 independent components of a symmetric tensor, in ParaView's
   // order xx, yy, zz, xy, yz, xz.
   static void append_symmetric(std::vector<float> &data, const Tensor2_3D &AA)
   {
      for (const auto &[ii, jj] : {std::pair{0, 0}, {1, 1}, {2, 2}, {0, 1}, {1, 2}, {0, 2}})
         data.push_back(static_cast<float>(AA(ii, jj)));
   }

   // A DataArray of the given VTK type, in binary: the byte count as UInt32,
   // then the values, each base64 encoded. Without component names, one
   // component; an empty name is left out, as for the points.
   template <typename T>
   static void write_array(std::ofstream &out, const std::string &type, const std::string &name,
                           const std::vector<std::string> &component_names,
                           const std::vector<T> &data)
   {
      const std::size_t num_components = component_names.empty() ? 1 : component_names.size();
      out << "<DataArray type=\"" << type << '"';
      if (!name.empty())
         out << " Name=\"" << name << '"';
      out << " NumberOfComponents=\"" << num_components << '"';
      if (!name.empty())
         for (std::size_t cc = 0; cc < component_names.size(); cc++)
            out << " ComponentName" << cc << "=\"" << component_names[cc] << '"';
      out << " format=\"binary\">\n";

      const std::uint32_t num_bytes = static_cast<std::uint32_t>(data.size() * sizeof(T));
      out << encode_base64(&num_bytes, sizeof(num_bytes))
          << encode_base64(data.data(), num_bytes) << "\n</DataArray>\n";
   }

   // Base64 encoding of size bytes.
   static std::string encode_base64(const void *bytes, std::size_t size)
   {
      static const char table[] =
         "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
      const auto *data = static_cast<const unsigned char *>(bytes);
      std::string out;
      out.reserve(4 * ((size + 2) / 3));
      for (std::size_t ii = 0; ii < size; ii += 3)
      {
         const std::uint32_t chunk = (static_cast<std::uint32_t>(data[ii]) << 16)
            | (ii + 1 < size ? static_cast<std::uint32_t>(data[ii + 1]) << 8 : 0)
            | (ii + 2 < size ? static_cast<std::uint32_t>(data[ii + 2]) : 0);
         out += table[(chunk >> 18) & 63];
         out += table[(chunk >> 12) & 63];
         out += (ii + 1 < size) ? table[(chunk >> 6) & 63] : '=';
         out += (ii + 2 < size) ? table[chunk & 63] : '=';
      }
      return out;
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

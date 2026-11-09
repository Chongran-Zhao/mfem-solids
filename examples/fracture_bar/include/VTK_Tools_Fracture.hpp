// ============================================================================
// VTK_Tools_Fracture.hpp
//
// Writes one VTU file per load step of phase-field fracture, and a PVD file
// listing them, for ParaView; the PVD file is written once, by write_pvd, so
// that the steps may be written by several MPI ranks. Each element,
// hexahedron or tetrahedron, is cut into the linear cells of MFEM's
// refinement of it with refine divisions of each edge, at the points of
// that refinement; the points of an element are its own, not shared with
// its neighbors. At each point it writes
//    displacement   u;
//    alpha          the phase field;
//    strain         eps = (grad u + grad u^T) / 2;
//    stress         g(alpha) sigma_0, the stress of the displacement
//                   problem, with sigma_0 the first Piola-Kirchhoff stress
//                   of the material, lambda tr(eps) I + 2 mu eps for the
//                   linear elastic one;
// and for each cell the region of its element, the element attribute.
//
// The points are at the reference position X when is_reference is set,
// else at the deformed one X + u. In small strain the two differ little,
// but where the material is broken its stiffness is g(alpha), about eta,
// and the displacement across the crack is large: drawn at X + u, the cells
// there are stretched far beyond the body. At X, ParaView's Warp By Vector
// of displacement draws the deformed shape, with any scale factor. The
// arrays are Float32 and Int32, written in binary, base64 encoded.
//
// Author: Chongran Zhao
// Date: Nov. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef VTK_TOOLS_FRACTURE_HPP
#define VTK_TOOLS_FRACTURE_HPP

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

#include "IMaterialModel.hpp"
#include "ISurfaceEnergy.hpp"
#include "LocalAssemblyTools.hpp"
#include "Tensor2_3D.hpp"

class VTK_Tools_Fracture
{
public:
   // Create the output folder; refine is the number of divisions of each
   // edge of an element, at least 1; is_reference puts the points at X.
   VTK_Tools_Fracture(const std::string &input_dir, int input_refine, bool input_is_reference)
      : dir(input_dir), refine(input_refine), is_reference(input_is_reference)
   {
      MFEM_VERIFY(refine >= 1, "vtu_refine must be at least 1.");
      std::filesystem::create_directories(dir);
   }

   // Write step_XXXX.vtu from the displacement and the phase field of a
   // step, on the same mesh; the stress is that of material, degraded by
   // g(alpha) of surface_energy.
   void save(int step, const mfem::GridFunction &disp, const mfem::GridFunction &alpha,
             const IMaterialModel &material, const ISurfaceEnergy &surface_energy)
   {
      const mfem::FiniteElementSpace &fespace = *disp.FESpace();
      mfem::Mesh &mesh = *fespace.GetMesh();

      // Point data, Float32, and cell data.
      std::vector<float> points, disp_out, alpha_out, strain_out, stress_out;
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

            // Reference position X and displacement u; the point is at X,
            // or at X + u.
            elem_map.Transform(ref_pt, coord);
            disp.GetVectorValue(elem_map, ref_pt, value);
            for (int comp = 0; comp < 3; comp++)
            {
               points.push_back(static_cast<float>(
                  is_reference ? coord(comp) : coord(comp) + value(comp)));
               disp_out.push_back(static_cast<float>(value(comp)));
            }

            const double alpha_value = alpha.GetValue(elem_map, ref_pt);
            alpha_out.push_back(static_cast<float>(alpha_value));

            // F, then eps = (F + F^T) / 2 - I and g(alpha) sigma_0(F).
            elem.CalcDShape(ref_pt, dN_dxi);
            mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
            const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(elem_disp, dN_dX);
            const Tensor2_3D strain = 0.5 * (F + F.transpose()) - Tensor2_3D::identity();
            const Tensor2_3D stress =
               surface_energy.get_g(alpha_value) * material.get_1st_PK_stress(F);
            append_symmetric(strain_out, strain);
            append_symmetric(stress_out, stress);
         }

         // The cells of this element, hexahedra or tetrahedra; MFEM and VTK
         // order their vertices the same way.
         const int num_vertices = mfem::Geometry::NumVerts[elem.GetGeomType()];
         const std::uint8_t cell_type = get_cell_type(elem.GetGeomType());
         for (int cc = 0; cc < refined.RefGeoms.Size() / num_vertices; cc++)
         {
            for (int vv = 0; vv < num_vertices; vv++)
               connectivity.push_back(first_point + refined.RefGeoms[num_vertices * cc + vv]);
            offsets.push_back(static_cast<std::int32_t>(connectivity.size()));
            types.push_back(cell_type);
            regions.push_back(attribute);
         }
      }

      const std::vector<std::string> tensor_names = {"xx", "yy", "zz", "xy", "yz", "xz"};
      std::ofstream out(dir / get_vtu_name(step));
      out << "<?xml version=\"1.0\"?>\n"
          << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\""
          << " header_type=\"UInt32\">\n"
          << "<UnstructuredGrid>\n"
          << "<Piece NumberOfPoints=\"" << points.size() / 3
          << "\" NumberOfCells=\"" << types.size() << "\">\n";

      out << "<PointData Vectors=\"displacement\" Scalars=\"alpha\">\n";
      write_array(out, "Float32", "displacement", {"x", "y", "z"}, disp_out);
      write_array(out, "Float32", "alpha", {}, alpha_out);
      write_array(out, "Float32", "strain", tensor_names, strain_out);
      write_array(out, "Float32", "stress", tensor_names, stress_out);
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
   }

   // Write the PVD file, listing step_XXXX.vtu of each step with its time.
   void write_pvd(const std::vector<std::pair<int, double>> &steps) const
   {
      std::ofstream out(get_pvd_path());
      out << std::setprecision(10)
          << "<?xml version=\"1.0\"?>\n"
          << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
          << "<Collection>\n";
      for (const auto &[step, time] : steps)
         out << "<DataSet timestep=\"" << time << "\" file=\"" << get_vtu_name(step) << "\"/>\n";
      out << "</Collection>\n</VTKFile>\n";
   }

   // Path of the PVD file, the one to open in ParaView.
   std::filesystem::path get_pvd_path() const
   {
      return dir / (dir.filename().string() + ".pvd");
   }

private:
   std::filesystem::path dir;
   const int refine;          // divisions of each edge of an element
   const bool is_reference;   // points at X, else at X + u

   // The VTK cell type of the linear cells of an element geometry.
   static std::uint8_t get_cell_type(mfem::Geometry::Type geometry)
   {
      if (geometry == mfem::Geometry::CUBE)
         return 12;   // VTK_HEXAHEDRON
      MFEM_VERIFY(geometry == mfem::Geometry::TETRAHEDRON,
                  "VTK_Tools_Fracture writes hexahedra and tetrahedra only.");
      return 10;      // VTK_TETRA
   }

   // The 6 independent components of a symmetric tensor, in ParaView's
   // order 11, 22, 33, 12, 23, 13.
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

   // Name of the VTU file of a step, step_XXXX.vtu.
   static std::string get_vtu_name(int step)
   {
      std::ostringstream name;
      name << "step_" << std::setw(4) << std::setfill('0') << step << ".vtu";
      return name.str();
   }
};

#endif

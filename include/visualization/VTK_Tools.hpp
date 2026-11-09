// ============================================================================
// VTK_Tools.hpp
//
// Writes one VTU file per load or time step on the deformed mesh, and a PVD
// file listing them, for ParaView; the PVD file is written once, by
// write_pvd, so that the steps may be written by several MPI ranks. Each element is cut into refine^3 linear
// hexahedra at (refine + 1)^3 evenly spaced points of its reference cube, as
// the visualization of MixPERIGEE does, so that the curved geometry and the
// fields of the higher-order elements show within it; the points of an
// element are its own, not shared with its neighbors. At each point it
// writes the displacement, and, if given, the velocity and the
// acceleration; the pressure; the Cauchy and the second Piola-Kirchhoff
// stresses and the Green-Lagrange strain; and for each hexahedron the
// region of its element, the element attribute. The pressure is that of
// the pressure field of the mixed form, and in the displacement form
// p = -tr(sigma) / 3, the pressure of the volumetric model, since the
// isochoric stress has no trace.
//
// The components are Cartesian; a derived class may write them in another
// orthonormal basis Q = [e_1 e_2 e_3] of the point, by get_basis, with its
// own component names and array suffix. Each quantity then takes the basis
// of the configuration it lives in: the current position x for the
// velocity, the acceleration and the Cauchy stress, the reference position
// X for the displacement, S and E,
//    v_Q = Q(x)^T v,   sigma_Q = Q(x)^T sigma Q(x),
//    u_Q = Q(X)^T u,   S_Q = Q(X)^T S Q(X),   E_Q = Q(X)^T E Q(X),
// with sigma = J^-1 P F^T; the displacement is also written in Cartesian
// components, for ParaView's Warp By Vector. The arrays are Float32 and
// Int32, written in binary, base64 encoded.
//
// Author: Chongran Zhao
// Date: Oct. 13, 2026
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

   virtual ~VTK_Tools() = default;

   // Write step_XXXX.vtu. velo and acce, of the
   // dynamics, are in the space of disp, and pres, of the mixed form, is
   // nodal in H1; each may be null. first_PK_stress(attribute, F, p) gives
   // P of the material of the element attribute, at the pressure p of pres,
   // zero without it.
   template <typename Stress>
   void save(int step, const mfem::GridFunction &disp,
             const mfem::GridFunction *velo, const mfem::GridFunction *acce,
             const mfem::GridFunction *pres, const Stress &first_PK_stress)
   {
      const mfem::FiniteElementSpace &fespace = *disp.FESpace();
      mfem::Mesh &mesh = *fespace.GetMesh();
      const bool is_cartesian = get_suffix().empty();

      // Point data, Float32, and cell data of the hexahedra.
      std::vector<float> points, disp_out, disp_basis_out, velo_out, acce_out, pres_out,
         cauchy_out, PK2_out, E_out;
      std::vector<std::int32_t> connectivity, offsets, regions;
      std::vector<std::uint8_t> types;

      mfem::Array<int> vdofs;
      mfem::Vector elem_disp, coord(3), current(3), value(3);
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

            // Reference position X and deformed position x = X + u, and the
            // bases there.
            elem_map.Transform(ref_pt, coord);
            disp.GetVectorValue(elem_map, ref_pt, value);
            add(coord, value, current);
            const Tensor2_3D Q_ref = get_basis(coord);
            const Tensor2_3D Q_cur = get_basis(current);
            for (int comp = 0; comp < 3; comp++)
            {
               points.push_back(static_cast<float>(current(comp)));
               disp_out.push_back(static_cast<float>(value(comp)));
            }
            if (!is_cartesian)
               append_vector(disp_basis_out, Q_ref, value);
            if (velo)
            {
               velo->GetVectorValue(elem_map, ref_pt, value);
               append_vector(velo_out, Q_cur, value);
            }
            if (acce)
            {
               acce->GetVectorValue(elem_map, ref_pt, value);
               append_vector(acce_out, Q_cur, value);
            }
            const double p = pres ? pres->GetValue(elem_map, ref_pt) : 0.0;

            // F, then P, sigma = J^-1 P F^T, S = F^-1 P and
            // E = 1/2 (F^T F - I) at the point.
            elem.CalcDShape(ref_pt, dN_dxi);
            mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
            const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(elem_disp, dN_dX);
            const Tensor2_3D PK1 = first_PK_stress(attribute, F, p);
            const Tensor2_3D cauchy = (1.0 / F.det()) * PK1 * F.transpose();
            const Tensor2_3D PK2 = F.inverse() * PK1;
            const Tensor2_3D E = 0.5 * (F.transpose() * F - Tensor2_3D::identity());
            pres_out.push_back(static_cast<float>(pres ? p : -cauchy.trace() / 3.0));
            append_symmetric(cauchy_out, Q_cur.transpose() * cauchy * Q_cur);
            append_symmetric(PK2_out, Q_ref.transpose() * PK2 * Q_ref);
            append_symmetric(E_out, Q_ref.transpose() * E * Q_ref);
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

      std::ofstream out(dir / get_vtu_name(step));
      out << "<?xml version=\"1.0\"?>\n"
          << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\""
          << " header_type=\"UInt32\">\n"
          << "<UnstructuredGrid>\n"
          << "<Piece NumberOfPoints=\"" << points.size() / 3
          << "\" NumberOfCells=\"" << types.size() << "\">\n";

      out << "<PointData Vectors=\"displacement\" Scalars=\"pressure\">\n";
      const std::string suffix = get_suffix();
      const std::vector<std::string> vector_names = get_vector_names();
      const std::vector<std::string> tensor_names = get_tensor_names();
      write_array(out, "Float32", "displacement", {"x", "y", "z"}, disp_out);
      if (!is_cartesian)
         write_array(out, "Float32", "displacement" + suffix, vector_names, disp_basis_out);
      if (velo)
         write_array(out, "Float32", "velocity" + suffix, vector_names, velo_out);
      if (acce)
         write_array(out, "Float32", "acceleration" + suffix, vector_names, acce_out);
      write_array(out, "Float32", "pressure", {"p"}, pres_out);
      write_array(out, "Float32", "Cauchy_stress" + suffix, tensor_names, cauchy_out);
      write_array(out, "Float32", "second_PK_stress" + suffix, tensor_names, PK2_out);
      write_array(out, "Float32", "Green_Lagrange_strain" + suffix, tensor_names, E_out);
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

protected:
   // The basis Q = [e_1 e_2 e_3] of the components at the point pt, the
   // basis vectors as columns; the Cartesian one, I, here.
   virtual Tensor2_3D get_basis(const mfem::Vector &pt) const { return Tensor2_3D::identity(); }

   // The names of the components of the vectors and, in ParaView's order
   // 11, 22, 33, 12, 23, 13, of the symmetric tensors, in that basis.
   virtual std::vector<std::string> get_vector_names() const { return {"x", "y", "z"}; }
   virtual std::vector<std::string> get_tensor_names() const
   {
      return {"xx", "yy", "zz", "xy", "yz", "xz"};
   }

   // Suffix of the names of the arrays in that basis, empty for the
   // Cartesian one.
   virtual std::string get_suffix() const { return ""; }

private:
   std::filesystem::path dir;
   const int refine;                                   // hexahedra per edge of an element

   // The components Q^T v of a vector in the basis Q.
   static void append_vector(std::vector<float> &data, const Tensor2_3D &Q,
                             const mfem::Vector &vv)
   {
      for (int ii = 0; ii < 3; ii++)
         data.push_back(static_cast<float>(Q(0, ii) * vv(0) + Q(1, ii) * vv(1)
                                           + Q(2, ii) * vv(2)));
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

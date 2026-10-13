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
// point it writes the displacement, also in Cartesian components, and, in
// spherical components (r, theta, phi), with theta from the z axis and phi
// about it, the displacement, the velocity, the acceleration, the Cauchy and
// the second Piola-Kirchhoff stresses and the Green-Lagrange strain; and the
// pressure, and for each hexahedron the region of its element, the element
// attribute. Each quantity takes the spherical basis of the configuration
// it lives in: the current position x for the velocity, the acceleration
// and the Cauchy stress, the reference position X for the displacement,
// S and E,
//    v_sph = Q(x)^T v,   sigma_sph = Q(x)^T sigma Q(x),
//    u_sph = Q(X)^T u,   S_sph = Q(X)^T S Q(X),   E_sph = Q(X)^T E Q(X),
// with Q = [e_r e_theta e_phi], and sigma = J^-1 P F^T. On the z axis phi is
// taken as 0, at the center also theta. The arrays are Float32 and Int32,
// written in binary, base64 encoded. Its vtu_writer.cpp, in this folder,
// includes it instead of the other one.
//
// Author: Chongran Zhao
// Date: Oct. 13, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef VTK_TOOLS_HPP
#define VTK_TOOLS_HPP

#include <cmath>
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
      std::vector<float> points, disp_out, disp_sph_out, velo_sph_out, acce_sph_out, pres_out,
         cauchy_sph_out, PK2_sph_out, E_sph_out;
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
            // spherical bases there.
            elem_map.Transform(ref_pt, coord);
            disp.GetVectorValue(elem_map, ref_pt, value);
            add(coord, value, current);
            const Tensor2_3D Q_ref = get_spherical_basis(coord);
            const Tensor2_3D Q_cur = get_spherical_basis(current);
            for (int comp = 0; comp < 3; comp++)
            {
               points.push_back(static_cast<float>(current(comp)));
               disp_out.push_back(static_cast<float>(value(comp)));
            }
            append_vector(disp_sph_out, Q_ref, value);
            velo.GetVectorValue(elem_map, ref_pt, value);
            append_vector(velo_sph_out, Q_cur, value);
            acce.GetVectorValue(elem_map, ref_pt, value);
            append_vector(acce_sph_out, Q_cur, value);
            const double p = pres.GetValue(elem_map, ref_pt);
            pres_out.push_back(static_cast<float>(p));

            // F, then P, sigma = J^-1 P F^T, S = F^-1 P and
            // E = 1/2 (F^T F - I) at the point.
            elem.CalcDShape(ref_pt, dN_dxi);
            mfem::Mult(dN_dxi, elem_map.InverseJacobian(), dN_dX);
            const Tensor2_3D F = LocalAssemblyTools::get_deformation_gradient(elem_disp, dN_dX);
            const Tensor2_3D PK1 = first_PK_stress(attribute, F, p);
            const Tensor2_3D cauchy = (1.0 / F.det()) * PK1 * F.transpose();
            const Tensor2_3D PK2 = F.inverse() * PK1;
            const Tensor2_3D E = 0.5 * (F.transpose() * F - Tensor2_3D::identity());
            append_symmetric(cauchy_sph_out, Q_cur.transpose() * cauchy * Q_cur);
            append_symmetric(PK2_sph_out, Q_ref.transpose() * PK2 * Q_ref);
            append_symmetric(E_sph_out, Q_ref.transpose() * E * Q_ref);
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
      const std::vector<std::string> sph = {"r", "theta", "phi"};
      // Only the 6 independent components of the symmetric tensors, in
      // ParaView's order, with r, theta, phi for x, y, z.
      const std::vector<std::string> sph_sym = {"rr", "theta_theta", "phi_phi",
                                                "r_theta", "theta_phi", "r_phi"};
      write_array(out, "Float32", "displacement", {"x", "y", "z"}, disp_out);
      write_array(out, "Float32", "displacement_sph", sph, disp_sph_out);
      write_array(out, "Float32", "velocity_sph", sph, velo_sph_out);
      write_array(out, "Float32", "acceleration_sph", sph, acce_sph_out);
      write_array(out, "Float32", "pressure", {"p"}, pres_out);
      write_array(out, "Float32", "Cauchy_stress_sph", sph_sym, cauchy_sph_out);
      write_array(out, "Float32", "second_PK_stress_sph", sph_sym, PK2_sph_out);
      write_array(out, "Float32", "Green_Lagrange_strain_sph", sph_sym, E_sph_out);
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

   // Q = [e_r e_theta e_phi] at the point pt, the basis vectors as columns,
   //    e_r     = (sin theta cos phi, sin theta sin phi, cos theta),
   //    e_theta = (cos theta cos phi, cos theta sin phi, -sin theta),
   //    e_phi   = (-sin phi, cos phi, 0);
   // on the z axis phi = 0, at the center also theta = 0.
   static Tensor2_3D get_spherical_basis(const mfem::Vector &pt)
   {
      const double radius = pt.Norml2();
      const double theta = (radius > 0.0) ? std::acos(pt(2) / radius) : 0.0;
      const double phi = std::atan2(pt(1), pt(0));
      const double sin_theta = std::sin(theta), cos_theta = std::cos(theta);
      const double sin_phi = std::sin(phi), cos_phi = std::cos(phi);
      return Tensor2_3D(sin_theta * cos_phi, cos_theta * cos_phi, -sin_phi,
                        sin_theta * sin_phi, cos_theta * sin_phi, cos_phi,
                        cos_theta, -sin_theta, 0.0);
   }

   // The components Q^T v of a vector in the basis Q.
   static void append_vector(std::vector<float> &data, const Tensor2_3D &Q,
                             const mfem::Vector &vv)
   {
      for (int ii = 0; ii < 3; ii++)
         data.push_back(static_cast<float>(Q(0, ii) * vv(0) + Q(1, ii) * vv(1)
                                           + Q(2, ii) * vv(2)));
   }

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

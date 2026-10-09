// ============================================================================
// read_mesh.cpp
//
// Generates a cubed-sphere hexahedral mesh of a ball: an inner cube and a
// shell of six blocks between the cube and the sphere surface, whose face is
// named "outer".
// Step 1: read the mesh section of config.yaml.
// Step 2: build the straight mesh in the logical coordinates.
// Step 3: curve it and map it onto the ball.
// Step 4: name the surface, report the mesh and save it.
//
// The logical point L lies in the cube [-1, 1]^3; the inner cube is
// |L|_inf <= a, with a = inner_size. The map to the ball of radius R is
//    inner cube   P = R L,
//    shell        P = R [(1 - w) a d + w d / |d|],   d = L / |L|_inf,
// with r = (|L|_inf - a) / (1 - a) in [0, 1] across the shell, and the
// grading w(r) = (q^(m r) - 1) / (q^m - 1), q = ratio^(1/(m-1)), which makes
// the outermost of the m layers ratio times as thick as the innermost one.
//
// Author: Chongran Zhao
// Date: Oct. 9, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <map>
#include <string>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

int main(int argc, char *argv[])
{
   // 1. The mesh section of config.yaml gives the size and the divisions of
   //    the ball and the output file, the space section the order of the
   //    curved geometry. By default the config.yaml of the directory the
   //    program runs in is read, the copy CMake puts in
   //    sphere_rotation/build/.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path("config.yaml");

   const YAML::Node config = YAML::LoadFile(yaml_file.string());
   const YAML::Node paras = config["mesh"];
   const double radius = paras["radius"].as<double>();
   const double inner_size = paras["inner_size"].as<double>();
   const int inner_elements = paras["inner_elements"].as<int>();
   const int radial_elements = paras["radial_elements"].as<int>();
   const double radial_ratio = paras["radial_ratio"].as<double>();
   const std::string output_mesh = paras["output"].as<std::string>();
   const int order = config["space"]["order"].as<int>();

   MFEM_VERIFY(inner_elements % 2 == 0, "inner_elements must be even.");
   MFEM_VERIFY(inner_size > 0.0 && inner_size < 1.0, "inner_size must be in (0, 1).");

   // 2. A vertex is named by its index (ii, jj, kk) on the grid of the inner
   //    cube, with layer 0, or by the index of a point on the cube surface
   //    and its layer ll = 1, ..., m of the shell. Its logical point is that
   //    grid point scaled to |L|_inf = a + (1 - a) ll / m.
   const int num_shell = 6 * inner_elements * inner_elements * radial_elements;
   const int num_elements = inner_elements * inner_elements * inner_elements + num_shell;
   const int num_bdr = 6 * inner_elements * inner_elements;
   mfem::Mesh mesh(3, 0, num_elements, num_bdr);

   std::map<std::array<int, 4>, int> vertex_ids;
   auto vertex = [&](const std::array<int, 3> &index, int ll)
   {
      const auto [it, is_new] = vertex_ids.try_emplace({index[0], index[1], index[2], ll},
                                                       mesh.GetNV());
      if (is_new)
      {
         const double scale = inner_size + (1.0 - inner_size) * ll / radial_elements;
         double coord[3];
         for (int axis = 0; axis < 3; axis++)
            coord[axis] = scale * (2.0 * index[axis] / inner_elements - 1.0);
         mesh.AddVertex(coord);
      }
      return it->second;
   };

   // Adds the hexahedron of corners 0-3 (one face) and 4-7 (the opposite
   // face, in the same order), turned to positive volume.
   auto add_hex = [&](std::array<int, 8> corners)
   {
      const double *origin = mesh.GetVertex(corners[0]);
      double edge[3][3];
      for (int ee = 0; ee < 3; ee++)
         for (int axis = 0; axis < 3; axis++)
            edge[ee][axis] = mesh.GetVertex(corners[std::array<int, 3>{1, 3, 4}[ee]])[axis] - origin[axis];
      const double volume = edge[0][0] * (edge[1][1] * edge[2][2] - edge[1][2] * edge[2][1])
                            - edge[0][1] * (edge[1][0] * edge[2][2] - edge[1][2] * edge[2][0])
                            + edge[0][2] * (edge[1][0] * edge[2][1] - edge[1][1] * edge[2][0]);
      if (volume < 0.0)
      {
         std::swap(corners[1], corners[3]);
         std::swap(corners[5], corners[7]);
      }
      mesh.AddHex(corners.data(), 1);
   };

   // The inner cube.
   for (int ii = 0; ii < inner_elements; ii++)
      for (int jj = 0; jj < inner_elements; jj++)
         for (int kk = 0; kk < inner_elements; kk++)
         {
            std::array<int, 8> corners;
            for (int cc = 0; cc < 8; cc++)
            {
               const int di = ((cc + 1) / 2) % 2, dj = (cc / 2) % 2, dk = cc / 4;
               corners[cc] = vertex({ii + di, jj + dj, kk + dk}, 0);
            }
            add_hex(corners);
         }

   // The shell: over each face of the cube, side 0 or 1 along axis, the
   // columns of hexahedra above its quads, from layer ll to ll + 1. The
   // surface quads of the last layer are the boundary.
   for (int axis = 0; axis < 3; axis++)
      for (int side = 0; side < 2; side++)
         for (int pp = 0; pp < inner_elements; pp++)
            for (int qq = 0; qq < inner_elements; qq++)
            {
               std::array<std::array<int, 3>, 4> quad;
               for (int cc = 0; cc < 4; cc++)
               {
                  quad[cc][axis] = side * inner_elements;
                  quad[cc][(axis + 1) % 3] = pp + ((cc + 1) / 2) % 2;
                  quad[cc][(axis + 2) % 3] = qq + cc / 2;
               }
               for (int ll = 0; ll < radial_elements; ll++)
               {
                  std::array<int, 8> corners;
                  for (int cc = 0; cc < 4; cc++)
                  {
                     corners[cc] = vertex(quad[cc], ll);
                     corners[cc + 4] = vertex(quad[cc], ll + 1);
                  }
                  add_hex(corners);
               }

               std::array<int, 4> face;
               for (int cc = 0; cc < 4; cc++)
                  face[cc] = vertex(quad[cc], radial_elements);
               mesh.AddBdrQuad(face.data(), 1);
            }

   // Finalize also turns the boundary faces outwards.
   mesh.FinalizeTopology();
   mesh.Finalize(false, true);

   // 3. Nodes of the given order hold the geometry; they are first placed
   //    in the logical coordinates, then mapped onto the ball.
   const double qq = std::pow(radial_ratio, 1.0 / (radial_elements - 1));
   auto grading = [&](double rr)
   {
      return (std::abs(qq - 1.0) < 1.0e-12) ? rr
         : (std::pow(qq, radial_elements * rr) - 1.0) / (std::pow(qq, radial_elements) - 1.0);
   };

   mesh.SetCurvature(order, false, 3, mfem::Ordering::byVDIM);
   mesh.Transform([&](const mfem::Vector &logical, mfem::Vector &point)
   {
      const double size = logical.Normlinf();
      if (size <= inner_size)
      {
         point.Set(radius, logical);
         return;
      }
      mfem::Vector dir(logical);
      dir /= size;
      const double ww = grading((size - inner_size) / (1.0 - inner_size));
      point.Set(radius * (1.0 - ww) * inner_size, dir);
      point.Add(radius * ww / dir.Norml2(), dir);
   });

   // 4. The surface is the boundary attribute 1, named "outer". The volume
   //    of the curved elements is compared with 4/3 pi R^3.
   mesh.bdr_attribute_sets.SetAttributeSet("outer", mfem::Array<int>({1}));

   double volume = 0.0;
   for (int ee = 0; ee < mesh.GetNE(); ee++)
      volume += mesh.GetElementVolume(ee);
   const double exact_volume = 4.0 / 3.0 * M_PI * radius * radius * radius;

   mfem::out << "\nMesh\n" << std::string(74, '-') << '\n' << std::left
             << std::setw(20) << "radius" << radius << '\n'
             << std::setw(20) << "inner cube" << inner_elements << "^3, half-size "
             << inner_size << " R\n"
             << std::setw(20) << "shell" << "6 x " << inner_elements << "^2 x "
             << radial_elements << ", ratio " << radial_ratio << '\n'
             << std::setw(20) << "geometry order" << order << '\n'
             << std::setw(20) << "elements" << mesh.GetNE() << '\n'
             << std::setw(20) << "vertices" << mesh.GetNV() << '\n'
             << std::setw(20) << "boundary elements" << mesh.GetNBE() << " (outer)\n"
             << std::setw(20) << "volume error" << std::scientific << std::setprecision(3)
             << (volume - exact_volume) / exact_volume << std::defaultfloat << '\n'
             << std::string(74, '-') << '\n';

   mesh.Save(output_mesh);
   mfem::out << std::left << std::setw(20) << "saved"
             << std::filesystem::absolute(output_mesh).string() << "\n\n";

   return 0;
}

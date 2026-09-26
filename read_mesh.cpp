// ============================================================================
// read_mesh.cpp
//
// Reads the mesh named in config.yaml and labels the six faces of its bounding
// box, so that boundary conditions can refer to faces by name.
// Step 1: read the mesh section of config.yaml.
// Step 2: read and refine the mesh.
// Step 3: label the boundary faces left, right, front, back, bottom, top.
// Step 4: report the faces of each label.
// Step 5: name the labels and save the mesh.
// Step 6: draw the named faces in a web page and open it.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include "mfem.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>

int main(int argc, char *argv[])
{
   // 1. The mesh section of config.yaml gives the mesh file, the number of
   //    uniform refinements and the output file. By default the config.yaml
   //    next to this source file is read. Input paths in it are relative to
   //    the directory of config.yaml; output paths are relative to the
   //    directory the program runs in, normally build/.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path(SOURCE_DIR) / "config.yaml";
   const std::filesystem::path config_dir = yaml_file.parent_path();

   const YAML::Node paras = YAML::LoadFile(yaml_file.string())["mesh"];
   const std::string mesh_file =
      (config_dir / paras["file"].as<std::string>()).lexically_normal().string();
   const int refine_levels = paras["refine_levels"].as<int>();
   const std::string output_mesh = paras["output"].as<std::string>();
   const std::string output_html = paras["output_html"].as<std::string>();
   const bool open_html = paras["open_html"].as<bool>();

   // 2. Each uniform refinement splits a hexahedron into 8.
   mfem::Mesh mesh(mesh_file);
   for (int ll = 0; ll < refine_levels; ll++)
      mesh.UniformRefinement();

   mfem::out << "\nMesh\n" << std::string(74, '-') << '\n' << std::left
             << std::setw(20) << "file" << mesh_file << '\n'
             << std::setw(20) << "refinements" << refine_levels << '\n'
             << std::setw(20) << "elements" << mesh.GetNE() << '\n'
             << std::setw(20) << "vertices" << mesh.GetNV() << '\n'
             << std::setw(20) << "boundary elements" << mesh.GetNBE() << '\n'
             << std::string(74, '-') << '\n';

   // 3. Attribute 2*axis + 1 is the face at the minimum of the bounding box
   //    along that axis, 2*axis + 2 the face at its maximum. A boundary face
   //    belongs to a label when all its vertices lie on that plane; faces on
   //    none of the six planes get attribute 7.
   const std::array<std::string, 7> face_names = {"left", "right", "front", "back",
                                                  "bottom", "top", "other"};

   mfem::Vector box_min, box_max;
   mesh.GetBoundingBox(box_min, box_max);
   const double tol = 1.0e-8 * (box_max.Normlinf() + box_min.Normlinf());

   mfem::Array<int> face_vertices;
   for (int jj = 0; jj < mesh.GetNBE(); jj++)
   {
      mesh.GetBdrElementVertices(jj, face_vertices);

      int label = 7;
      for (int axis = 0; axis < 3 && label == 7; axis++)
      {
         bool on_min = true, on_max = true;
         for (int vv : face_vertices)
         {
            const double coord = mesh.GetVertex(vv)[axis];
            on_min = on_min && std::abs(coord - box_min(axis)) < tol;
            on_max = on_max && std::abs(coord - box_max(axis)) < tol;
         }
         if (on_min)
            label = 2 * axis + 1;
         else if (on_max)
            label = 2 * axis + 2;
      }
      mesh.SetBdrAttribute(jj, label);
   }
   // Rebuild the list mesh.bdr_attributes from the new labels.
   mesh.SetAttributes();

   // 4. For each label: the number of faces and the range of x, y, z of
   //    their vertices, printed as a table.
   std::array<int, 7> num_faces = {0, 0, 0, 0, 0, 0, 0};
   std::array<std::array<double, 3>, 7> coord_min, coord_max;
   for (auto &xx : coord_min)
      xx.fill(std::numeric_limits<double>::max());
   for (auto &xx : coord_max)
      xx.fill(std::numeric_limits<double>::lowest());

   for (int jj = 0; jj < mesh.GetNBE(); jj++)
   {
      const int label = mesh.GetBdrAttribute(jj);
      num_faces[label - 1]++;

      mesh.GetBdrElementVertices(jj, face_vertices);
      for (int vv : face_vertices)
         for (int axis = 0; axis < 3; axis++)
         {
            const double coord = mesh.GetVertex(vv)[axis];
            coord_min[label - 1][axis] = std::min(coord_min[label - 1][axis], coord);
            coord_max[label - 1][axis] = std::max(coord_max[label - 1][axis], coord);
         }
   }

   // "[min, max]" of one coordinate, padded to a fixed width.
   auto range = [](double lo, double hi)
   {
      std::ostringstream out;
      out << std::setprecision(4) << '[' << lo << ", " << hi << ']';
      return out.str();
   };

   // "name  x = value" for the plane of one label, padded to a fixed width.
   auto plane = [&](int label)
   {
      const int axis = (label - 1) / 2;
      const double value = (label % 2 == 1) ? box_min(axis) : box_max(axis);
      std::ostringstream out;
      out << std::setprecision(4) << std::left << std::setw(7) << face_names[label - 1]
          << "xyz"[axis] << " = " << value;
      return out.str();
   };

   mfem::out << "\nBoundary faces\n"
             << "        z                     " << std::left << std::setw(18) << plane(1) << plane(2) << '\n'
             << "        |  y                  " << std::setw(18) << plane(3) << plane(4) << '\n'
             << "        | /                   " << std::setw(18) << plane(5) << plane(6) << '\n'
             << "        |/\n"
             << "        +------ x\n\n"
             << std::left << std::setw(6) << "attr" << std::setw(8) << "name"
             << std::right << std::setw(10) << "bdr elems" << "   "
             << std::left << std::setw(16) << "x" << std::setw(16) << "y" << "z\n"
             << std::string(74, '-') << '\n';
   for (int label = 1; label <= 7; label++)
   {
      if (num_faces[label - 1] == 0)
         continue;
      const auto &lo = coord_min[label - 1];
      const auto &hi = coord_max[label - 1];
      mfem::out << std::left << std::setw(6) << label << std::setw(8) << face_names[label - 1]
                << std::right << std::setw(10) << num_faces[label - 1] << "   "
                << std::left << std::setw(16) << range(lo[0], hi[0])
                << std::setw(16) << range(lo[1], hi[1]) << range(lo[2], hi[2]) << '\n';
   }
   mfem::out << std::string(74, '-') << '\n'
             << std::left << std::setw(14) << "bounding box" << std::right << std::setw(10)
             << mesh.GetNBE() << "   " << std::left << std::setw(16) << range(box_min(0), box_max(0))
             << std::setw(16) << range(box_min(1), box_max(1)) << range(box_min(2), box_max(2))
             << "\n\n";
   if (num_faces[6] > 0)
      mfem::out << "Warning: some boundary faces are not on the bounding box.\n";

   // 5. Give each label that occurs its name as a boundary attribute set, so
   //    that later steps can ask for the face "right" instead of attribute 2.
   //    Mesh files with named sets are written in the MFEM mesh v1.3 format.
   for (int label = 1; label <= 7; label++)
      if (num_faces[label - 1] > 0)
         mesh.bdr_attribute_sets.SetAttributeSet(face_names[label - 1],
                                                 mfem::Array<int>({label}));

   mesh.Save(output_mesh);
   mfem::out << std::left << std::setw(20) << "saved"
             << std::filesystem::absolute(output_mesh).string() << '\n';

   // 6. The boundary faces go into the template boundary_view.html, which
   //    draws them with three.js: one color and one name label per face,
   //    and the x, y, z axes. The page is opened in the browser.
   const double box_size = std::hypot(box_max(0) - box_min(0), box_max(1) - box_min(1),
                                      box_max(2) - box_min(2));

   std::ostringstream json;
   json << std::setprecision(10) << "{\"file\": \"" << mesh_file << "\", \"box\": ["
        << box_min(0) << ", " << box_max(0) << ", " << box_min(1) << ", "
        << box_max(1) << ", " << box_min(2) << ", " << box_max(2) << "], \"faces\": [";

   bool first_face = true;
   for (int label = 1; label <= 7; label++)
   {
      if (num_faces[label - 1] == 0)
         continue;

      // Label position: center of the face group, pushed outwards along the
      // normal of its plane so that it does not sit inside the surface.
      std::array<double, 3> label_position;
      for (int axis = 0; axis < 3; axis++)
         label_position[axis] = 0.5 * (coord_min[label - 1][axis] + coord_max[label - 1][axis]);
      if (label <= 6)
         label_position[(label - 1) / 2] += (label % 2 == 1 ? -0.06 : 0.06) * box_size;

      json << (first_face ? "" : ", ") << "{\"attribute\": " << label
           << ", \"name\": \"" << face_names[label - 1] << "\", \"plane\": \""
           << (label <= 6 ? plane(label).substr(7) : std::string("")) << "\", \"label_position\": ["
           << label_position[0] << ", " << label_position[1] << ", " << label_position[2]
           << "], \"polygons\": [";
      first_face = false;

      bool first_polygon = true;
      for (int jj = 0; jj < mesh.GetNBE(); jj++)
      {
         if (mesh.GetBdrAttribute(jj) != label)
            continue;
         mesh.GetBdrElementVertices(jj, face_vertices);

         json << (first_polygon ? "[" : ", [");
         first_polygon = false;
         for (int aa = 0; aa < face_vertices.Size(); aa++)
         {
            const double *coord = mesh.GetVertex(face_vertices[aa]);
            json << (aa == 0 ? "" : ", ") << coord[0] << ", " << coord[1] << ", " << coord[2];
         }
         json << "]";
      }
      json << "]}";
   }
   json << "]}";

   std::ifstream template_file(std::filesystem::path(SOURCE_DIR) / "boundary_view.html");
   std::string page((std::istreambuf_iterator<char>(template_file)),
                    std::istreambuf_iterator<char>());
   const std::string placeholder = "const data = __DATA__;";
   page.replace(page.find(placeholder), placeholder.size(), "const data = " + json.str() + ";");

   std::ofstream(output_html) << page;
   const std::string html_path = std::filesystem::absolute(output_html).string();
   mfem::out << std::left << std::setw(20) << "saved" << html_path << "\n\n";

   if (open_html)
      std::system(("open \"" + html_path + "\"").c_str());

   return 0;
}

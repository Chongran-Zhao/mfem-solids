// ============================================================================
// read_mesh.cpp
//
// Reads the mesh named in config.yaml and names its boundary attributes, the
// physical tags of a Gmsh file, as mesh.faces of config.yaml gives them, so
// that boundary conditions and the initial crack can refer to faces by name.
// The attributes are those of the mesh file, and a face may lie inside the
// body, as the faces of an initial crack.
// Step 1: read the mesh section of config.yaml.
// Step 2: read and refine the mesh.
// Step 3: check the names of mesh.faces against the attributes of the mesh.
// Step 4: report the faces of each attribute.
// Step 5: name the attributes and save the mesh.
// Step 6: draw the named faces in a web page and open it.
//
// Author: Chongran Zhao
// Date: Sep. 26, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <string>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

int main(int argc, char *argv[])
{
   // 1. The mesh section of config.yaml gives the mesh file, the number of
   //    uniform refinements, the names of the boundary attributes and the
   //    output file. The file given as first
   //    argument is read, else the config.yaml of the directory the program
   //    runs in, the copies CMake puts in the build folder. The mesh file is
   //    relative to the project
   //    directory; output paths are relative to the directory the program
   //    runs in.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path("config.yaml");

   const YAML::Node paras = YAML::LoadFile(yaml_file.string())["mesh"];
   const std::string mesh_file =
      (std::filesystem::path(PROJECT_DIR) / paras["file"].as<std::string>()).lexically_normal().string();
   const int refine_levels = paras["refine_levels"].as<int>();
   // Name -> boundary attribute, e.g. left: 5.
   const std::map<std::string, int> face_map = paras["faces"].as<std::map<std::string, int>>();
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

   // 3. Every name of mesh.faces must be an attribute of the mesh; the
   //    attributes without a name are reported, and no boundary condition
   //    can refer to them.
   mfem::Vector box_min, box_max;
   mesh.GetBoundingBox(box_min, box_max);
   const double tol = 1.0e-8 * (box_max.Normlinf() + box_min.Normlinf());

   std::map<int, std::string> face_names;   // attribute -> name, "-" without one
   for (int attr : mesh.bdr_attributes)
      face_names[attr] = "-";
   for (const auto &[name, attr] : face_map)
   {
      MFEM_VERIFY(face_names.count(attr) == 1,
                  "mesh.faces: " << name << " is attribute " << attr
                  << ", which no boundary face of the mesh has.");
      face_names[attr] = name;
   }

   // 4. For each attribute: the number of faces and the range of x, y, z of
   //    their vertices, printed as a table.
   std::map<int, int> num_faces;
   std::map<int, std::array<double, 3>> coord_min, coord_max;
   for (const auto &[attr, name] : face_names)
   {
      num_faces[attr] = 0;
      coord_min[attr].fill(std::numeric_limits<double>::max());
      coord_max[attr].fill(std::numeric_limits<double>::lowest());
   }

   mfem::Array<int> face_vertices;
   for (int jj = 0; jj < mesh.GetNBE(); jj++)
   {
      const int attr = mesh.GetBdrAttribute(jj);
      num_faces[attr]++;

      mesh.GetBdrElementVertices(jj, face_vertices);
      for (int vv : face_vertices)
         for (int axis = 0; axis < 3; axis++)
         {
            const double coord = mesh.GetVertex(vv)[axis];
            coord_min[attr][axis] = std::min(coord_min[attr][axis], coord);
            coord_max[attr][axis] = std::max(coord_max[attr][axis], coord);
         }
   }

   // "[min, max]" of one coordinate, padded to a fixed width.
   auto range = [](double lo, double hi)
   {
      std::ostringstream out;
      out << std::setprecision(4) << '[' << lo << ", " << hi << ']';
      return out.str();
   };

   // The axis of the plane of the bounding box the faces of attr lie on, and
   // +1 at its maximum or -1 at its minimum; axis -1 when they lie on none,
   // as the faces inside the body.
   auto box_plane = [&](int attr, int &side)
   {
      for (int axis = 0; axis < 3; axis++)
         if (coord_max[attr][axis] - coord_min[attr][axis] < tol)
         {
            if (std::abs(coord_min[attr][axis] - box_min(axis)) < tol)
            {
               side = -1;
               return axis;
            }
            if (std::abs(coord_max[attr][axis] - box_max(axis)) < tol)
            {
               side = 1;
               return axis;
            }
         }
      side = 0;
      return -1;
   };

   mfem::out << "\nBoundary faces\n"
             << std::left << std::setw(6) << "attr" << std::setw(8) << "name"
             << std::right << std::setw(10) << "bdr elems" << "   "
             << std::left << std::setw(16) << "x" << std::setw(16) << "y" << "z\n"
             << std::string(74, '-') << '\n';
   for (const auto &[attr, name] : face_names)
   {
      const auto &lo = coord_min[attr];
      const auto &hi = coord_max[attr];
      mfem::out << std::left << std::setw(6) << attr << std::setw(8) << name
                << std::right << std::setw(10) << num_faces[attr] << "   "
                << std::left << std::setw(16) << range(lo[0], hi[0])
                << std::setw(16) << range(lo[1], hi[1]) << range(lo[2], hi[2]) << '\n';
   }
   mfem::out << std::string(74, '-') << '\n'
             << std::left << std::setw(14) << "bounding box" << std::right << std::setw(10)
             << mesh.GetNBE() << "   " << std::left << std::setw(16) << range(box_min(0), box_max(0))
             << std::setw(16) << range(box_min(1), box_max(1)) << range(box_min(2), box_max(2))
             << "\n\n";
   if (face_map.size() < face_names.size())
      mfem::out << "Warning: some boundary attributes have no name in mesh.faces.\n";

   // 5. Give each name its attribute as a boundary attribute set, so that
   //    later steps can ask for the face "right" instead of its attribute.
   //    Mesh files with named sets are written in the MFEM mesh v1.3 format.
   for (const auto &[name, attr] : face_map)
      mesh.bdr_attribute_sets.SetAttributeSet(name, mfem::Array<int>({attr}));

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
   for (const auto &[attr, name] : face_names)
   {
      // Label position: center of the faces; for faces on a plane of the
      // bounding box, pushed outwards along its normal so that it does not
      // sit inside the surface.
      int side = 0;
      const int axis = box_plane(attr, side);
      std::array<double, 3> label_position;
      for (int aa = 0; aa < 3; aa++)
         label_position[aa] = 0.5 * (coord_min[attr][aa] + coord_max[attr][aa]);
      std::string plane;
      if (axis >= 0)
      {
         label_position[axis] += 0.06 * side * box_size;
         std::ostringstream out;
         out << std::setprecision(4) << "xyz"[axis] << " = " << coord_min[attr][axis];
         plane = out.str();
      }

      json << (first_face ? "" : ", ") << "{\"attribute\": " << attr
           << ", \"name\": \"" << name << "\", \"plane\": \"" << plane
           << "\", \"label_position\": [" << label_position[0] << ", " << label_position[1]
           << ", " << label_position[2] << "], \"polygons\": [";
      first_face = false;

      bool first_polygon = true;
      for (int jj = 0; jj < mesh.GetNBE(); jj++)
      {
         if (mesh.GetBdrAttribute(jj) != attr)
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

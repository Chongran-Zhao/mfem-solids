// ============================================================================
// read_kfile.cpp
//
// Converts the brain of the SIMon model of NHTSA, an LS-Dyna keyword file
// (Simon.k), into an MFEM mesh: the solid elements become the elements, the
// shells the boundary elements, the beams of the vessels are left out.
// Step 1: read the mesh section of config.yaml.
// Step 2: read the nodes, parts, solids and shells of the keyword file.
// Step 3: turn each solid into a hexahedron, wedge, pyramid or tetrahedron
//         by its repeated nodes, with positive volume.
// Step 4: name the faces: the outer faces under the shells of the skull and
//         of the foramen magnum, those left free, and the inner faces under
//         the falx and the tentorium.
// Step 5: name the regions and the surfaces, report the mesh and save it.
//
// Every solid part keeps its own attribute, 1, 2, ... in the order of the
// part IDs, named by its title, e.g. "CEREBRUM-R"; the sets without the
// side, "cerebrum", "cerebellum", "bstem", "ventricle", "csf", join the two
// halves, and "brain" the cerebrum, cerebellum and brain stem. The boundary
// attributes are 1 "skull", 2 "foramen", 3 "falx_tentorium" (inner faces),
// 4 "free" (outer faces under no shell, 12 of the CSF at the top of the
// midline in Simon.k). Only the skull is rigid in SIMon, so only "skull"
// follows the motion of the head; the foramen is an elastic shell.
//
// The quality of the 8-node hexahedra is that of Table 3 of Giudice et al.
// (2019): the characteristic length, volume over the largest face area, and
// the scaled Jacobian, the least over the corners of det[e1 e2 e3] /
// (|e1| |e2| |e3|) with the three edges ee from the corner, in the units of
// the keyword file.
//
// Author: Chongran Zhao
// Date: Nov. 25, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

// A solid or shell of the keyword file: its ID, part and nodes.
struct kfile_element
{
   int id;
   int part;
   std::vector<int> nodes;
};

// The fields of a card: split at the commas of the free format, else cut at
// the given widths of the fixed format.
std::vector<std::string> split_card(const std::string &line, const std::vector<int> &widths)
{
   std::vector<std::string> fields;
   if (line.find(',') != std::string::npos)
   {
      std::stringstream stream(line);
      std::string field;
      while (std::getline(stream, field, ','))
         fields.push_back(field);
      return fields;
   }
   std::size_t start = 0;
   for (const int width : widths)
   {
      if (start >= line.size())
         break;
      fields.push_back(line.substr(start, width));
      start += width;
   }
   return fields;
}

// The integer of a field, 0 if blank.
int to_int(const std::string &field)
{
   return (field.find_first_not_of(" \t") == std::string::npos) ? 0 : std::stoi(field);
}

// The mean and the standard deviation of the values.
std::array<double, 2> get_mean_sd(const std::vector<double> &values)
{
   double sum = 0.0, square = 0.0;
   for (const double value : values)
      sum += value;
   const double mean = sum / values.size();
   for (const double value : values)
      square += (value - mean) * (value - mean);
   return {mean, std::sqrt(square / values.size())};
}

// The determinant of the edges ee[0], ee[1], ee[2].
double get_det(const std::array<std::array<double, 3>, 3> &ee)
{
   return ee[0][0] * (ee[1][1] * ee[2][2] - ee[1][2] * ee[2][1])
          - ee[0][1] * (ee[1][0] * ee[2][2] - ee[1][2] * ee[2][0])
          + ee[0][2] * (ee[1][0] * ee[2][1] - ee[1][1] * ee[2][0]);
}

int main(int argc, char *argv[])
{
   // 1. The mesh section of config.yaml gives the keyword file, the factor
   //    from its lengths to those of the mesh and the output file. The file
   //    given as first argument is read, else the config.yaml of the
   //    directory the program runs in.
   const std::filesystem::path yaml_file =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : std::filesystem::path("config.yaml");

   const YAML::Node paras = YAML::LoadFile(yaml_file.string())["mesh"];
   const std::string kfile = paras["kfile"].as<std::string>();
   const double length_scale = paras["length_scale"].as<double>();
   const std::string output_mesh = paras["output"].as<std::string>();

   // 2. The cards of *NODE, *PART, *ELEMENT_SOLID and *ELEMENT_SHELL; the
   //    lines of $ are comments. A *PART has a title line, then the card of
   //    its ID. The rigid skull is part 8, the foramen magnum part 9, the
   //    falx and the tentorium part 6, the shells of Simon.k.
   const int skull_part = 8, foramen_part = 9, falx_part = 6;

   std::ifstream input(kfile);
   MFEM_VERIFY(input, "Cannot open the keyword file \"" << kfile << "\".");

   std::map<int, std::array<double, 3>> nodes;
   std::map<int, std::string> part_titles;
   std::vector<kfile_element> solids, shells;
   std::string keyword, line, title;
   int part_line = 0;
   while (std::getline(input, line))
   {
      if (!line.empty() && line.back() == '\r')
         line.pop_back();
      if (line.empty() || line[0] == '$')
         continue;
      if (line[0] == '*')
      {
         keyword = line.substr(0, line.find_first_of(" \t"));
         part_line = 0;
         continue;
      }
      if (keyword == "*NODE")
      {
         const std::vector<std::string> fields = split_card(line, {8, 16, 16, 16});
         nodes[to_int(fields[0])] = {std::stod(fields[1]), std::stod(fields[2]),
                                     std::stod(fields[3])};
      }
      else if (keyword == "*PART")
      {
         if (part_line++ == 0)
            title = line.substr(0, line.find_last_not_of(" \t") + 1);
         else if (part_line == 2)
            part_titles[to_int(split_card(line, {10})[0])] = title;
      }
      else if (keyword == "*ELEMENT_SOLID" || keyword == "*ELEMENT_SHELL")
      {
         const bool is_solid = (keyword == "*ELEMENT_SOLID");
         const std::vector<std::string> fields =
            split_card(line, std::vector<int>(is_solid ? 10 : 6, 8));
         kfile_element element{to_int(fields[0]), to_int(fields[1]), {}};
         for (std::size_t ff = 2; ff < fields.size(); ff++)
            element.nodes.push_back(to_int(fields[ff]));
         (is_solid ? solids : shells).push_back(element);
      }
   }
   MFEM_VERIFY(!nodes.empty() && !solids.empty(), "No node or solid in \"" << kfile << "\".");

   // The solid parts, in the order of their IDs, give the attributes; the
   // nodes of the solids, in the order of their IDs, the vertices.
   std::map<int, int> part_attributes;
   std::map<int, int> vertex_ids;
   for (const kfile_element &solid : solids)
   {
      part_attributes[solid.part] = 0;
      MFEM_VERIFY(solid.nodes.size() == 8, "Solid " << solid.id << " has not 8 nodes.");
      for (const int node : solid.nodes)
      {
         MFEM_VERIFY(nodes.count(node), "Node " << node << " of solid " << solid.id
                     << " is not defined.");
         vertex_ids[node] = 0;
      }
   }
   int attribute = 0;
   for (auto &[part, attr] : part_attributes)
      attr = ++attribute;
   int vertex_count = 0;
   for (auto &[node, vertex] : vertex_ids)
      vertex = vertex_count++;

   // 3. A solid of LS-Dyna repeats nodes for the other shapes; its distinct
   //    nodes and their pattern give the shape, in the vertex order of MFEM:
   //    8   hexahedron  n1 ... n8, the same order,
   //    6   wedge       n5 = n6 and n7 = n8: triangles (n1, n2, n5) and
   //                    (n4, n3, n7), the top face shrunk to the edge n5 n7,
   //    5   pyramid     base n1 ... n4 and the fifth node the apex,
   //    4   tetrahedron n1 ... n4.
   //    A negative volume, by the Jacobian at the center, is turned over.
   struct mesh_element
   {
      mfem::Geometry::Type geometry;
      std::vector<int> vertices;
      int attribute;
   };
   std::vector<mesh_element> elements;
   std::array<int, mfem::Geometry::NumGeom> geometry_count{};
   for (const kfile_element &solid : solids)
   {
      const std::vector<int> &nn = solid.nodes;
      const std::set<int> distinct(nn.begin(), nn.end());
      const bool base_distinct = (std::set<int>(nn.begin(), nn.begin() + 4).size() == 4);
      mesh_element element{mfem::Geometry::CUBE, {}, part_attributes.at(solid.part)};
      if (distinct.size() == 8)
         element.vertices = nn;
      else if (distinct.size() == 6 && base_distinct && nn[4] == nn[5] && nn[6] == nn[7])
      {
         element.geometry = mfem::Geometry::PRISM;
         element.vertices = {nn[0], nn[1], nn[4], nn[3], nn[2], nn[6]};
      }
      else if (distinct.size() == 5 && base_distinct)
      {
         element.geometry = mfem::Geometry::PYRAMID;
         element.vertices = {nn[0], nn[1], nn[2], nn[3]};
         for (const int node : distinct)
            if (std::find(nn.begin(), nn.begin() + 4, node) == nn.begin() + 4)
               element.vertices.push_back(node);
      }
      else if (distinct.size() == 4 && base_distinct)
      {
         element.geometry = mfem::Geometry::TETRAHEDRON;
         element.vertices = {nn[0], nn[1], nn[2], nn[3]};
      }
      else
         MFEM_ABORT("Solid " << solid.id << " repeats its nodes in no known pattern.");
      for (int &vertex : element.vertices)
         vertex = vertex_ids.at(vertex);
      geometry_count[element.geometry]++;
      elements.push_back(element);
   }

   // Builds the mesh of the elements, with the boundary elements of
   // bdr_elements, (geometry, vertices, attribute), without topology.
   auto build_mesh = [&](const std::vector<mesh_element> &bdr_elements)
   {
      mfem::Mesh mesh(3, vertex_ids.size(), elements.size(), bdr_elements.size());
      for (const auto &[node, vertex] : vertex_ids)
      {
         double coord[3];
         for (int axis = 0; axis < 3; axis++)
            coord[axis] = length_scale * nodes.at(node)[axis];
         mesh.AddVertex(coord);
      }
      for (const mesh_element &element : elements)
      {
         const int *vv = element.vertices.data();
         if (element.geometry == mfem::Geometry::CUBE)
            mesh.AddHex(vv, element.attribute);
         else if (element.geometry == mfem::Geometry::PRISM)
            mesh.AddWedge(vv, element.attribute);
         else if (element.geometry == mfem::Geometry::PYRAMID)
            mesh.AddPyramid(vv, element.attribute);
         else
            mesh.AddTet(vv, element.attribute);
      }
      for (const mesh_element &bdr : bdr_elements)
      {
         if (bdr.geometry == mfem::Geometry::SQUARE)
            mesh.AddBdrQuad(bdr.vertices.data(), bdr.attribute);
         else
            mesh.AddBdrTriangle(bdr.vertices.data(), bdr.attribute);
      }
      return mesh;
   };

   int flipped = 0;
   {
      mfem::Mesh mesh = build_mesh({});
      mfem::DenseMatrix jacobian(3);
      for (std::size_t ee = 0; ee < elements.size(); ee++)
      {
         mesh.GetElementJacobian(ee, jacobian);
         if (jacobian.Det() > 0.0)
            continue;
         std::vector<int> &vv = elements[ee].vertices;
         if (elements[ee].geometry == mfem::Geometry::CUBE)
         {
            std::swap(vv[1], vv[3]);
            std::swap(vv[5], vv[7]);
         }
         else if (elements[ee].geometry == mfem::Geometry::PRISM)
         {
            std::swap(vv[1], vv[2]);
            std::swap(vv[4], vv[5]);
         }
         else if (elements[ee].geometry == mfem::Geometry::PYRAMID)
            std::swap(vv[1], vv[3]);
         else
            std::swap(vv[0], vv[1]);
         flipped++;
      }
   }

   // 4. The faces of the elements: an outer face, of one element, takes the
   //    attribute of the shell on it, skull or foramen, else "free"; an inner
   //    face under a shell of the falx and the tentorium is kept as a
   //    boundary element too. A shell is found by its distinct vertices; the
   //    shells of the skull also give its center, the mean of the shell
   //    centers weighted by their areas, in the units of the keyword file.
   enum bdr_attribute { skull_bdr = 1, foramen_bdr = 2, falx_bdr = 3, free_bdr = 4 };

   std::map<std::vector<int>, int> shell_parts;
   std::array<double, 3> skull_center{};
   double skull_area = 0.0;
   for (const kfile_element &shell : shells)
   {
      std::set<int> distinct;
      for (const int node : shell.nodes)
      {
         MFEM_VERIFY(vertex_ids.count(node), "Node " << node << " of shell " << shell.id
                     << " is on no solid.");
         distinct.insert(vertex_ids.at(node));
      }
      shell_parts[std::vector<int>(distinct.begin(), distinct.end())] = shell.part;

      if (shell.part != skull_part)
         continue;
      // Triangles (n1, n2, n3) and (n1, n3, n4), the second empty for a
      // triangle, n3 = n4.
      for (int tri = 0; tri < 2; tri++)
      {
         const std::array<double, 3> &p0 = nodes.at(shell.nodes[0]);
         const std::array<double, 3> &p1 = nodes.at(shell.nodes[1 + tri]);
         const std::array<double, 3> &p2 = nodes.at(shell.nodes[2 + tri]);
         std::array<double, 3> cross;
         for (int axis = 0; axis < 3; axis++)
         {
            const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
            cross[axis] = (p1[a1] - p0[a1]) * (p2[a2] - p0[a2])
                          - (p1[a2] - p0[a2]) * (p2[a1] - p0[a1]);
         }
         const double area = 0.5 * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1]
                                              + cross[2] * cross[2]);
         for (int axis = 0; axis < 3; axis++)
            skull_center[axis] += area * (p0[axis] + p1[axis] + p2[axis]) / 3.0;
         skull_area += area;
      }
   }
   for (int axis = 0; axis < 3; axis++)
      skull_center[axis] /= skull_area;

   std::vector<mesh_element> bdr_elements;
   std::array<int, 5> bdr_count{};
   std::set<std::vector<int>> matched_shells;
   {
      mfem::Mesh mesh = build_mesh({});
      mesh.FinalizeTopology(false);
      mfem::Array<int> face_vertices;
      for (int face = 0; face < mesh.GetNumFaces(); face++)
      {
         int elem1, elem2;
         mesh.GetFaceElements(face, &elem1, &elem2);
         mesh.GetFaceVertices(face, face_vertices);
         std::vector<int> key(face_vertices.begin(), face_vertices.end());
         std::sort(key.begin(), key.end());
         const auto shell = shell_parts.find(key);
         const int part = (shell == shell_parts.end()) ? 0 : shell->second;

         int attr = 0;
         if (elem2 < 0)
            attr = (part == skull_part) ? skull_bdr : (part == foramen_part) ? foramen_bdr
                   : free_bdr;
         else if (part == falx_part)
            attr = falx_bdr;
         if (attr == 0)
            continue;
         if (part != 0)
            matched_shells.insert(key);
         bdr_count[attr]++;
         bdr_elements.push_back({mesh.GetFaceGeometry(face),
                                 std::vector<int>(face_vertices.begin(), face_vertices.end()),
                                 attr});
      }
   }
   MFEM_VERIFY(matched_shells.size() == shell_parts.size(),
               shell_parts.size() - matched_shells.size()
               << " shells lie on no face of the solids.");

   // 5. The mesh with its boundary; Finalize turns the outer boundary faces
   //    outwards.
   mfem::Mesh mesh = build_mesh(bdr_elements);
   mesh.FinalizeTopology();
   mesh.Finalize(false, true);
   const int wrong_orientation = mesh.CheckElementOrientation(false);

   // The regions: each part by its title, the two halves by the title
   // without the side, lower case, and "brain".
   std::map<std::string, std::vector<int>> region_sets;
   for (const auto &[part, attr] : part_attributes)
   {
      const std::string part_title =
         part_titles.count(part) ? part_titles.at(part) : "PART-" + std::to_string(part);
      mesh.attribute_sets.SetAttributeSet(part_title, mfem::Array<int>({attr}));
      std::string region = part_title.substr(0, part_title.find('-'));
      std::transform(region.begin(), region.end(), region.begin(),
                     [](unsigned char letter) { return std::tolower(letter); });
      region_sets[region].push_back(attr);
      if (region == "cerebrum" || region == "cerebellum" || region == "bstem")
         region_sets["brain"].push_back(attr);
   }
   for (const auto &[region, attrs] : region_sets)
   {
      mfem::Array<int> set_attrs;
      for (const int attr : attrs)
         set_attrs.Append(attr);
      mesh.attribute_sets.SetAttributeSet(region, set_attrs);
   }

   const std::array<std::string, 5> bdr_names = {"", "skull", "foramen", "falx_tentorium",
                                                 "free"};
   for (int attr = 1; attr <= 4; attr++)
      if (bdr_count[attr] > 0)
         mesh.bdr_attribute_sets.SetAttributeSet(bdr_names[attr], mfem::Array<int>({attr}));

   // The volumes of the regions, and the quality of the 8-node hexahedra in
   // the units of the keyword file.
   std::map<std::string, double> region_volumes;
   for (const auto &[region, attrs] : region_sets)
      for (int ee = 0; ee < mesh.GetNE(); ee++)
         if (std::find(attrs.begin(), attrs.end(), mesh.GetAttribute(ee)) != attrs.end())
            region_volumes[region] += mesh.GetElementVolume(ee);

   const std::array<std::array<int, 3>, 8> corner_edges = {{
      {1, 3, 4}, {2, 0, 5}, {3, 1, 6}, {0, 2, 7}, {7, 5, 0}, {4, 6, 1}, {5, 7, 2}, {6, 4, 3}}};
   const std::array<std::array<int, 4>, 6> hex_faces = {{
      {0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}}};
   std::vector<double> lengths, scaled_jacobians;
   for (std::size_t ee = 0; ee < elements.size(); ee++)
   {
      if (elements[ee].geometry != mfem::Geometry::CUBE)
         continue;
      mfem::Array<int> vv;
      mesh.GetElementVertices(ee, vv);
      auto point = [&](int corner)
      {
         const double *coord = mesh.GetVertex(vv[corner]);
         return std::array<double, 3>{coord[0] / length_scale, coord[1] / length_scale,
                                      coord[2] / length_scale};
      };
      double least = 1.0;
      for (int corner = 0; corner < 8; corner++)
      {
         std::array<std::array<double, 3>, 3> edges;
         double norms = 1.0;
         for (int ed = 0; ed < 3; ed++)
         {
            double norm = 0.0;
            for (int axis = 0; axis < 3; axis++)
            {
               edges[ed][axis] = point(corner_edges[corner][ed])[axis] - point(corner)[axis];
               norm += edges[ed][axis] * edges[ed][axis];
            }
            norms *= std::sqrt(norm);
         }
         least = std::min(least, get_det(edges) / norms);
      }
      scaled_jacobians.push_back(least);

      double largest_area = 0.0;
      for (const std::array<int, 4> &face : hex_faces)
      {
         std::array<double, 3> diag1, diag2, cross;
         for (int axis = 0; axis < 3; axis++)
         {
            diag1[axis] = point(face[2])[axis] - point(face[0])[axis];
            diag2[axis] = point(face[3])[axis] - point(face[1])[axis];
         }
         for (int axis = 0; axis < 3; axis++)
            cross[axis] = diag1[(axis + 1) % 3] * diag2[(axis + 2) % 3]
                          - diag1[(axis + 2) % 3] * diag2[(axis + 1) % 3];
         largest_area = std::max(largest_area, 0.5 * std::sqrt(cross[0] * cross[0]
                                                                + cross[1] * cross[1]
                                                                + cross[2] * cross[2]));
      }
      const double volume = mesh.GetElementVolume(ee) / std::pow(length_scale, 3);
      lengths.push_back(volume / largest_area);
   }
   const std::array<double, 2> length_stats = get_mean_sd(lengths);
   const std::array<double, 2> jacobian_stats = get_mean_sd(scaled_jacobians);

   mfem::out << "\nMesh\n" << std::string(74, '-') << '\n' << std::left
             << std::setw(24) << "keyword file" << kfile << '\n'
             << std::setw(24) << "nodes, solids, shells" << nodes.size() << ", "
             << solids.size() << ", " << shells.size() << '\n'
             << std::setw(24) << "vertices" << mesh.GetNV() << '\n'
             << std::setw(24) << "elements" << mesh.GetNE() << ": "
             << geometry_count[mfem::Geometry::CUBE] << " hexahedra, "
             << geometry_count[mfem::Geometry::PRISM] << " wedges, "
             << geometry_count[mfem::Geometry::PYRAMID] << " pyramids, "
             << geometry_count[mfem::Geometry::TETRAHEDRON] << " tetrahedra\n"
             << std::setw(24) << "turned over" << flipped << ", left inverted "
             << wrong_orientation << '\n'
             << std::setw(24) << "boundary elements" << mesh.GetNBE() << ": skull "
             << bdr_count[skull_bdr] << ", foramen " << bdr_count[foramen_bdr]
             << ", falx_tentorium " << bdr_count[falx_bdr] << ", free "
             << bdr_count[free_bdr] << '\n';
   for (const auto &[part, attr] : part_attributes)
      mfem::out << std::setw(24) << "attribute " + std::to_string(attr)
                << (part_titles.count(part) ? part_titles.at(part) : "") << " (part "
                << part << ")\n";
   for (const auto &[region, volume] : region_volumes)
      mfem::out << std::setw(24) << region + " volume" << std::scientific
                << std::setprecision(4) << volume << std::defaultfloat << '\n';
   mfem::out << std::setw(24) << "skull center" << skull_center[0] << ", "
             << skull_center[1] << ", " << skull_center[2] << " (keyword units)\n"
             << std::setw(24) << "hex length" << std::fixed << std::setprecision(2)
             << length_stats[0] << " +- " << length_stats[1] << " (keyword units)\n"
             << std::setw(24) << "hex scaled Jacobian" << jacobian_stats[0] << " +- "
             << jacobian_stats[1] << ", least "
             << *std::min_element(scaled_jacobians.begin(), scaled_jacobians.end())
             << std::defaultfloat << '\n'
             << std::string(74, '-') << '\n';

   mesh.Save(output_mesh);
   mfem::out << std::left << std::setw(24) << "saved"
             << std::filesystem::absolute(output_mesh).string() << "\n\n";
}

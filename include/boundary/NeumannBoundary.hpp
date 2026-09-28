// ============================================================================
// NeumannBoundary.hpp
//
// Reads the Neumann boundary conditions from config.yaml and adds external
// force integrators with the load data.
//
// Author: Chongran Zhao
// Date: Sep. 28, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef NEUMANN_BOUNDARY_HPP
#define NEUMANN_BOUNDARY_HPP

#include <iomanip>
#include <memory>
#include <string>
#include <vector>
#include <mfem.hpp>
#include <yaml-cpp/yaml.h>
#include "LoadData.hpp"
#include "Vector_3D.hpp"

class NeumannBoundary
{
public:
   // Reads the Neumann section of config.yaml.
   NeumannBoundary(const YAML::Node &paras, mfem::FiniteElementSpace &fespace)
   {
      mfem::Mesh &mesh = *fespace.GetMesh();

      for (const YAML::Node &input_face : paras["faces"])
      {
         const std::string face = input_face.as<std::string>();

         traction_load load;
         load.face = face;
         load.marker = mesh.bdr_attribute_sets.GetAttributeSetMarker(face);
         load.traction = std::make_unique<mfem::VectorFunctionCoefficient>(3,
            [face](const mfem::Vector &pt, double tt, mfem::Vector &T)
            {
               const Vector_3D value =
                  LoadData::surface_traction(pt, tt, face);
               for (int ii = 0; ii < 3; ii++)
                  T(ii) = value(ii);
            });

         traction_load_list.push_back(std::move(load));
      }
   }

   // Whether it is traction loading.
   bool is_traction_load() const { return !traction_load_list.empty(); }

   // Add the external force integrators with the traction of each face.
   void add_traction_integrators(mfem::LinearForm &external_force)
   {
      for (traction_load &load : traction_load_list)
         external_force.AddBoundaryIntegrator(
            new mfem::VectorBoundaryLFIntegrator(*load.traction), load.marker);
   }

   // Set the time of the tractions, before the LinearForm is assembled.
   void set_time(double tt)
   {
      for (traction_load &load : traction_load_list)
         load.traction->SetTime(tt);
   }

   // Print the traction faces.
   void print_traction_load() const
   {
      mfem::out << "Traction loading\n" << std::left
                << std::setw(10) << "face" << '\n'
                << std::string(74, '-') << '\n';

      for (const traction_load &load : traction_load_list)
         mfem::out << load.face << '\n';

      mfem::out << std::string(74, '-') << "\n\n";
   }

   // Print the traction load at each step, as the resultant force.
   void print_traction_load_by_step(const mfem::LinearForm &traction_force) const
   {
      // LinearForm::operator() is the action on a GridFunction, so the
      // entries are read through the Vector.
      const mfem::Vector &force = traction_force;
      const mfem::FiniteElementSpace &fespace = *traction_force.FESpace();
      Vector_3D resultant;
      for (int node = 0; node < fespace.GetNDofs(); node++)
         for (int dir = 0; dir < 3; dir++)
            resultant(dir) += force(fespace.DofToVDof(node, dir));

      mfem::out << "  force"
                << "  Fx = " << resultant(0)
                << "  Fy = " << resultant(1)
                << "  Fz = " << resultant(2) << '\n';
   }

private:

   struct traction_load
   {
      std::string face;          // face name
      mfem::Array<int> marker;   // marker of its boundary attributes
      std::unique_ptr<mfem::VectorFunctionCoefficient> traction;  // T(pt, tt)
   };

   std::vector<traction_load> traction_load_list;  // one per entry of faces
};

#endif

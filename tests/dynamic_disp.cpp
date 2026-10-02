#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <mfem.hpp>
#include <yaml-cpp/yaml.h>

#include "DirichletBoundary.hpp"
#include "GlobalAssembly_Disp.hpp"
#include "LocalAssembly_Disp.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_Dynamic_Disp.hpp"
#include "TimeMethod_GenAlpha.hpp"
#include "TimeSolver_Dynamic_Disp.hpp"

namespace
{
constexpr double density = 1000.0;

void require(bool condition, const std::string &message)
{
   if (!condition) throw std::runtime_error(message);
}

struct Fixture
{
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON);
   mfem::H1_FECollection fec{1, 3};
   mfem::FiniteElementSpace space{&mesh, &fec, 3, mfem::Ordering::byVDIM};
   mfem::GridFunction u{&space}, v{&space}, a{&space};
   YAML::Node config;

   explicit Fixture(const std::string &boundary = "fixed_bc: []\ndisp_bc: []")
   {
      mfem::Array<int> left, right, all;
      for (int i = 0; i < mesh.GetNBE(); ++i)
      {
         auto *tr = mesh.GetBdrElementTransformation(i);
         mfem::Vector x(3);
         tr->Transform(mfem::Geometries.GetCenter(tr->GetGeometryType()), x);
         const int attr = mesh.GetBdrAttribute(i);
         all.Append(attr);
         if (std::abs(x(0)) < 1e-12) left.Append(attr);
         if (std::abs(x(0) - 1.0) < 1e-12) right.Append(attr);
      }
      mesh.bdr_attribute_sets.SetAttributeSet("left", left);
      mesh.bdr_attribute_sets.SetAttributeSet("right", right);
      mesh.bdr_attribute_sets.SetAttributeSet("all", all);
      config = YAML::Load(boundary);
      u = 0.0; v = 0.0; a = 0.0;
   }

   std::unique_ptr<GlobalAssembly_Disp> assembly(const std::string &traction = "faces: []")
   {
      return std::make_unique<GlobalAssembly_Disp>(space,
         std::make_unique<LocalAssembly_Disp>(set_material_model()),
         std::make_unique<DirichletBoundary>(config, space),
         std::make_unique<NeumannBoundary>(YAML::Load(traction), space));
   }

   std::unique_ptr<NonlinearSolver_Dynamic_Disp> solver()
   {
      return std::make_unique<NonlinearSolver_Dynamic_Disp>(assembly(), density,
         YAML::Load("newton_rel_tol: 1e-10\nnewton_abs_tol: 1e-10\nnewton_max_iter: 20"));
   }
};

void tangent_check()
{
   Fixture f("fixed_bc: [{face: left, dir: x}]\ndisp_bc: []");
   auto assembly = f.assembly();
   auto mass = assembly->assemble_mass(density);
   const int n = f.space.GetTrueVSize();
   mfem::Vector un(n), vn(n), an(n), u(n), direction(n);
   for (int i = 0; i < n; ++i)
   {
      un(i) = 0.001 * std::sin(i + 1.0);
      vn(i) = 0.01 * std::cos(i + 1.0);
      an(i) = 0.1 * std::sin(2.0 * i);
      u(i) = un(i) + 0.002 * std::cos(i);
      direction(i) = std::sin(0.7 * i + 0.3);
   }
   for (double rho : {0.0, 0.5, 1.0})
   {
      const TimeMethod_GenAlpha method(rho);
      NonlinearSolver_Dynamic_Disp::StepOperator op(*assembly, *mass, method,
                                                   0.01, un, vn, an);
      for (bool constrained : {false, true})
      {
         mfem::Vector d(direction), plus(u), minus(u), rp(n), rm(n), kd(n);
         if (constrained) assembly->set_essential_bdr(d);
         const double eps = 1e-7;
         plus.Add(eps, d); minus.Add(-eps, d);
         if (constrained)
         {
            op.Mult(plus, rp); op.Mult(minus, rm);
            op.GetGradient(u).Mult(d, kd);
         }
         else
         {
            op.assemble_residual(plus, rp); op.assemble_residual(minus, rm);
            op.assemble_tangent(u)->Mult(d, kd);
         }
         rp -= rm; rp /= 2.0 * eps; rp -= kd;
         require(rp.Norml2() / kd.Norml2() < 1e-7, "Effective tangent finite difference");
      }
   }
}

void translation_and_time_loop()
{
   Fixture f;
   mfem::VectorFunctionCoefficient velocity(3, [](const mfem::Vector &, mfem::Vector &v)
   { v.SetSize(3); v(0) = 0.02; v(1) = -0.01; v(2) = 0.03; });
   f.v.ProjectCoefficient(velocity);
   const auto dir = std::filesystem::temp_directory_path() /
      ("mfem-dynamics-test-" + std::to_string(
         std::chrono::steady_clock::now().time_since_epoch().count()));
   TimeSolver_Dynamic_Disp time(f.solver(), 0.003, 0.01, 0.5, dir);
   time.run(f.u, f.v, f.a);
   mfem::GridFunction expected(&f.space);
   expected.ProjectCoefficient(velocity); expected *= 0.01;
   expected -= f.u;
   require(expected.Normlinf() < 1e-12 && f.a.Normlinf() < 1e-8,
           "Rigid translation should have zero acceleration");
   expected.ProjectCoefficient(velocity); expected -= f.v;
   require(expected.Normlinf() < 1e-10, "Rigid translation velocity");
   std::ifstream history(dir / "time.csv");
   std::string line;
   std::getline(history, line);
   for (double expected_time : {0.0, 0.003, 0.006, 0.009, 0.01})
   {
      require(bool(std::getline(history, line)), "Missing time history row");
      const auto begin = line.find(',') + 1;
      require(std::abs(std::stod(line.substr(begin)) - expected_time) < 1e-14,
              "Incorrect physical output time");
   }
   require(!std::getline(history, line), "Extra time history row");
   int gf_count = 0;
   for (const auto &entry : std::filesystem::directory_iterator(dir))
      if (entry.path().extension() == ".gf") ++gf_count;
   require(gf_count == 15, "Displacement/velocity/acceleration output count");
   std::filesystem::remove_all(dir);
}

// A single hex with lateral motion suppressed reduces to a longitudinal
// oscillator: K = (kappa + 4 mu/3) A/L, M = rho A L/3.
double vibration(double dt)
{
   Fixture f("fixed_bc: [{face: left, dir: x}, {face: all, dir: y}, {face: all, dir: z}]\ndisp_bc: []");
   const double amplitude = 1e-6;
   mfem::VectorFunctionCoefficient initial(3, [=](const mfem::Vector &x, mfem::Vector &u)
   { u.SetSize(3); u = 0.0; u(0) = amplitude * x(0); });
   f.u.ProjectCoefficient(initial);
   auto solver = f.solver();
   solver->initialize(0.0, f.u, f.v, f.a);
   const double mu = young / (2.0 * (1.0 + poisson));
   const double kappa = young / (3.0 * (1.0 - 2.0 * poisson));
   const double stiffness = kappa + 4.0 * mu / 3.0;
   const double omega = std::sqrt(3.0 * stiffness / density);
   mfem::VectorFunctionCoefficient mode(3, [](const mfem::Vector &x, mfem::Vector &u)
   { u.SetSize(3); u = 0.0; u(0) = x(0); });
   mfem::GridFunction shape(&f.space); shape.ProjectCoefficient(mode);
   const double shape_norm = shape * shape;
   const double energy0 = 0.5 * stiffness * amplitude * amplitude;
   const TimeMethod_GenAlpha method(1.0);
   const int steps = int(std::lround(0.02 / dt));
   double q = amplitude;
   for (int step = 0; step < steps; ++step)
   {
      solver->solve(step * dt, dt, method, f.u, f.v, f.a);
      q = (shape * f.u) / shape_norm;
      const double expected = amplitude * std::cos((step + 1) * 2.0 * std::atan(omega * dt / 2.0));
      require(std::abs(q - expected) / amplitude < 2e-5, "Free vibration discrete frequency");
      const double energy = solver->get_kinetic_energy(f.v) + 0.5 * stiffness * q * q;
      require(std::abs(energy / energy0 - 1.0) < 2e-5, "Small amplitude vibration energy");
   }
   return std::abs(q / amplitude - std::cos(omega * 0.02));
}

void prescribed_motion()
{
   Fixture f("fixed_bc: [{face: left, dir: x}, {face: left, dir: y}, {face: left, dir: z}]\ndisp_bc: [{face: right, dir: z}]");
   auto solver = f.solver();
   solver->initialize(0.0, f.u, f.v, f.a);
   const TimeMethod_GenAlpha method(0.5);
   solver->solve(0.0, 0.001, method, f.u, f.v, f.a);
   mfem::Array<int> dofs;
   f.space.GetEssentialTrueDofs(f.mesh.bdr_attribute_sets.GetAttributeSetMarker("right"), dofs, 2);
   for (int i : dofs)
      require(std::abs(f.u(i) + 0.0005) < 1e-14 &&
              std::abs(f.v(i) + 0.5) < 1e-12 && std::abs(f.a(i)) < 1e-8,
              "Consistent prescribed displacement/velocity/acceleration");
}

void traction_momentum()
{
   Fixture f;
   auto assembly = f.assembly("faces: [right]");
   auto mass = assembly->assemble_mass(density);
   NonlinearSolver_Dynamic_Disp solver(std::move(assembly), density,
      YAML::Load("newton_rel_tol: 1e-10\nnewton_abs_tol: 1e-10\nnewton_max_iter: 20"));
   solver.initialize(0.0, f.u, f.v, f.a);
   mfem::VectorFunctionCoefficient z(3, [](const mfem::Vector &, mfem::Vector &v)
   { v.SetSize(3); v = 0.0; v(2) = 1.0; });
   mfem::GridFunction translation(&f.space); translation.ProjectCoefficient(z);
   const TimeMethod_GenAlpha method(1.0);
   const double dt = 0.001;
   for (int step = 0; step < 3; ++step)
   {
      solver.solve(step * dt, dt, method, f.u, f.v, f.a);
      mfem::Vector momentum(f.v.Size()); mass->Mult(f.v, momentum);
      const double time = (step + 1) * dt;
      const double expected = -0.5 * 2275.0 * time * time;
      require(std::abs(translation * momentum - expected) < 1e-10,
              "Traction must be evaluated at the intermediate physical time");
   }
}
}

int main()
{
   try
   {
      tangent_check();
      translation_and_time_loop();
      const double coarse = vibration(0.001), fine = vibration(0.0005);
      require(coarse / fine > 3.5 && coarse / fine < 4.5, "Second order time convergence");
      prescribed_motion();
      traction_momentum();
      std::cout << "PASS: tangent, rigid motion, vibration frequency/energy/convergence, prescribed motion, traction momentum, output times\n";
   }
   catch (const std::exception &error)
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
}

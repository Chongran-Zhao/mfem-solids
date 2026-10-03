// ============================================================================
// dynamic_disp.cpp of tests
//
// Checks of the displacement dynamics on one hexahedron, the unit cube, with
// the material and the density of MaterialModelData:
//    tangent            K_eff against central differences of R_dyn, without
//                       and with the constraints, for rho_inf = 0, 0.5, 1;
//    rigid translation  constant velocity, zero acceleration, the output
//                       times and files of the time solver;
//    free vibration     a small longitudinal vibration against the midpoint
//                       rule of its linear oscillator, its energy, and the
//                       second-order convergence in time;
//    prescribed motion  disp_bc on the right face, LoadData::disp_driven;
//    traction impulse   the momentum against the impulse of the traction on
//                       the right face, LoadData::surface_traction.
// Run by CTest; prints PASS, or the failed check and exits with 1.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
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

// Newton settings of every check.
static const char *newton_settings =
   "{newton_rel_tol: 1.0e-10, newton_abs_tol: 1.0e-10, newton_max_iter: 20}";

// Throws the message if the condition fails.
static void check(bool condition, const std::string &message)
{
   if (!condition)
      throw std::runtime_error(message);
}

// The nonlinear solver, with its step functions made public for the
// tangent check.
class NonlinearSolver_Dynamic_Disp_Test : public NonlinearSolver_Dynamic_Disp
{
public:
   using NonlinearSolver_Dynamic_Disp::NonlinearSolver_Dynamic_Disp;
   using NonlinearSolver_Dynamic_Disp::set_step;
   using NonlinearSolver_Dynamic_Disp::assemble_residual;
   using NonlinearSolver_Dynamic_Disp::assemble_tangent;
};

// The unit cube, its faces left (x = 0), right (x = 1) and all, the
// displacement space, and the boundary conditions of a check.
struct UnitCube
{
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON);
   mfem::H1_FECollection fec_u{1, 3};
   mfem::FiniteElementSpace space_u{&mesh, &fec_u, 3, mfem::Ordering::byVDIM};
   YAML::Node dirichlet_config;
   YAML::Node neumann_config;
   const std::filesystem::path results_dir =
      std::filesystem::temp_directory_path() / "mfem-solids-test-dynamic-disp";

   UnitCube(const std::string &dirichlet_input, const std::string &neumann_input)
      : dirichlet_config(YAML::Load(dirichlet_input)),
        neumann_config(YAML::Load(neumann_input))
   {
      mfem::Array<int> left, right, all;
      for (int be = 0; be < mesh.GetNBE(); be++)
      {
         mfem::ElementTransformation &face_map = *mesh.GetBdrElementTransformation(be);
         mfem::Vector center(3);
         face_map.Transform(mfem::Geometries.GetCenter(face_map.GetGeometryType()), center);
         const int attribute = mesh.GetBdrAttribute(be);
         all.Append(attribute);
         if (std::abs(center(0)) < 1.0e-12)
            left.Append(attribute);
         if (std::abs(center(0) - 1.0) < 1.0e-12)
            right.Append(attribute);
      }
      mesh.bdr_attribute_sets.SetAttributeSet("left", left);
      mesh.bdr_attribute_sets.SetAttributeSet("right", right);
      mesh.bdr_attribute_sets.SetAttributeSet("all", all);
   }

   ~UnitCube() { std::filesystem::remove_all(results_dir); }

   std::unique_ptr<GlobalAssembly_Disp> make_global_assembly()
   {
      return std::make_unique<GlobalAssembly_Disp>(space_u,
         std::make_unique<LocalAssembly_Disp>(set_material_model()),
         std::make_unique<DirichletBoundary>(dirichlet_config, space_u),
         std::make_unique<NeumannBoundary>(neumann_config, space_u));
   }

   template <typename Solver = NonlinearSolver_Dynamic_Disp>
   std::unique_ptr<Solver> make_nonlinear_solver(double rho_inf)
   {
      return std::make_unique<Solver>(make_global_assembly(),
                                      std::make_unique<TimeMethod_GenAlpha>(rho_inf),
                                      YAML::Load(newton_settings));
   }

   // M, for the momentum and the kinetic energy.
   std::unique_ptr<mfem::SparseMatrix> make_mass()
   {
      return make_global_assembly()->assemble_mass();
   }

   // A field of the constant vector value.
   mfem::GridFunction make_uniform_field(double value_x, double value_y, double value_z)
   {
      mfem::Vector value(3);
      value(0) = value_x;
      value(1) = value_y;
      value(2) = value_z;
      mfem::VectorConstantCoefficient coeff(value);
      mfem::GridFunction field(&space_u);
      field.ProjectCoefficient(coeff);
      return field;
   }
};

// One step with the nonlinear solver from the state disp, velo, acce into
// itself.
static void solve_step(NonlinearSolver_Dynamic_Disp &nonlinear_solver, double tt, double dt,
                       mfem::GridFunction &disp, mfem::GridFunction &velo,
                       mfem::GridFunction &acce)
{
   const mfem::GridFunction disp_old(disp), velo_old(velo), acce_old(acce);
   nonlinear_solver.solve(tt, dt, disp_old, velo_old, acce_old, disp, velo, acce);
}

// K_eff against the central difference of R_dyn along a direction dd,
//    ( R_dyn(u + eps dd) - R_dyn(u - eps dd) ) / (2 eps) = K_eff dd + O(eps^2),
// at every dof, and through Mult and GetGradient, with the constraints, for
// a dd zero on the constrained dofs, which carry the identity.
static void check_tangent()
{
   UnitCube cube("{fixed_bc: [{face: left, dir: x}], disp_bc: []}", "{faces: []}");
   const int num_dofs = cube.space_u.GetTrueVSize();
   mfem::Vector disp_old(num_dofs), velo_old(num_dofs), acce_old(num_dofs);
   mfem::Vector disp(num_dofs), direction(num_dofs);
   for (int ii = 0; ii < num_dofs; ii++)
   {
      disp_old(ii) = 0.001 * std::sin(ii + 1.0);
      velo_old(ii) = 0.01 * std::cos(ii + 1.0);
      acce_old(ii) = 0.1 * std::sin(2.0 * ii);
      disp(ii) = disp_old(ii) + 0.002 * std::cos(ii);
      direction(ii) = std::sin(0.7 * ii + 0.3);
   }
   const double eps = 1.0e-7;

   for (double rho_inf : {0.0, 0.5, 1.0})
   {
      auto nonlinear_solver =
         cube.make_nonlinear_solver<NonlinearSolver_Dynamic_Disp_Test>(rho_inf);
      nonlinear_solver->set_step(0.0, 0.01, disp_old, velo_old, acce_old);

      mfem::Vector disp_plus(disp), disp_minus(disp);
      disp_plus.Add(eps, direction);
      disp_minus.Add(-eps, direction);
      mfem::Vector residual_plus(num_dofs), residual_minus(num_dofs), tangent_dir(num_dofs);

      // At every dof.
      nonlinear_solver->assemble_residual(disp_plus, residual_plus);
      nonlinear_solver->assemble_residual(disp_minus, residual_minus);
      nonlinear_solver->assemble_tangent(disp)->Mult(direction, tangent_dir);
      residual_plus -= residual_minus;
      residual_plus /= 2.0 * eps;
      residual_plus -= tangent_dir;
      check(residual_plus.Norml2() < 1.0e-7 * tangent_dir.Norml2(),
            "The tangent differs from the central difference of the residual.");

      // With the constraints, along a direction zero on the constrained dofs.
      mfem::Vector direction_free(direction);
      for (int dof : cube.make_global_assembly()->get_dirichlet().get_ess_tdof_list())
         direction_free(dof) = 0.0;
      disp_plus = disp;
      disp_minus = disp;
      disp_plus.Add(eps, direction_free);
      disp_minus.Add(-eps, direction_free);
      nonlinear_solver->Mult(disp_plus, residual_plus);
      nonlinear_solver->Mult(disp_minus, residual_minus);
      nonlinear_solver->GetGradient(disp).Mult(direction_free, tangent_dir);
      residual_plus -= residual_minus;
      residual_plus /= 2.0 * eps;
      residual_plus -= tangent_dir;
      check(residual_plus.Norml2() < 1.0e-7 * tangent_dir.Norml2(),
            "The constrained tangent differs from the central difference of the residual.");
   }
}

// A free cube moving with a constant velocity v: u = v t and a = 0, at the
// times 0, 0.003, 0.006, 0.009 and the shortened last step to 0.01.
static void check_rigid_translation()
{
   UnitCube cube("{fixed_bc: [], disp_bc: []}", "{faces: []}");
   const mfem::GridFunction velocity = cube.make_uniform_field(0.02, -0.01, 0.03);
   mfem::GridFunction disp(&cube.space_u), velo(velocity), acce(&cube.space_u);
   disp = 0.0;
   acce = 0.0;

   TimeSolver_Dynamic_Disp time_solver(cube.make_nonlinear_solver(0.5), 0.003, 0.01,
                                       cube.results_dir);
   time_solver.run(disp, velo, acce);

   mfem::GridFunction error(velocity);
   error *= 0.01;
   error -= disp;
   check(error.Normlinf() < 1.0e-12, "Rigid translation: wrong displacement.");
   error = velocity;
   error -= velo;
   check(error.Normlinf() < 1.0e-10, "Rigid translation: wrong velocity.");
   check(acce.Normlinf() < 1.0e-8, "Rigid translation: nonzero acceleration.");

   std::ifstream time_file(cube.results_dir / "time.csv");
   std::string line;
   std::getline(time_file, line);
   for (double expected_time : {0.0, 0.003, 0.006, 0.009, 0.01})
   {
      check(static_cast<bool>(std::getline(time_file, line)), "time.csv: a step is missing.");
      const double time = std::stod(line.substr(line.find(',') + 1));
      check(std::abs(time - expected_time) < 1.0e-14, "time.csv: wrong time.");
   }
   check(!std::getline(time_file, line), "time.csv: too many steps.");

   int num_gf_files = 0;
   for (const std::filesystem::directory_entry &entry :
        std::filesystem::directory_iterator(cube.results_dir))
      if (entry.path().extension() == ".gf")
         num_gf_files++;
   check(num_gf_files == 15, "Not 5 steps of disp, velo and acce saved.");
}

// The cube with u_y = u_z = 0 everywhere and u_x = 0 on the left face, from
// u_x = q x: one longitudinal mode, the oscillator
//    m q'' + k q = 0,   k = (kappa + 4 mu / 3) A / L,   m = rho A L / 3,
// for a small q, here 1e-6. With rho_inf = 1, the midpoint rule, its
// discrete solution is q_n = q_0 cos(n 2 atan(omega dt / 2)), and its energy
// stays constant. Returns the error at t = 0.02 against cos(omega t), for
// the convergence check.
static double check_free_vibration(double dt)
{
   UnitCube cube("{fixed_bc: [{face: left, dir: x}, {face: all, dir: y}, {face: all, dir: z}],"
                 " disp_bc: []}", "{faces: []}");
   const double amplitude = 1.0e-6;
   mfem::VectorFunctionCoefficient mode_value(3, [](const mfem::Vector &pt, mfem::Vector &value)
   {
      value = 0.0;
      value(0) = pt(0);
   });
   mfem::GridFunction mode(&cube.space_u);
   mode.ProjectCoefficient(mode_value);
   const double mode_norm = mode * mode;

   mfem::GridFunction disp(mode), velo(&cube.space_u), acce(&cube.space_u);
   disp *= amplitude;
   velo = 0.0;

   const double mu = young / (2.0 * (1.0 + poisson));
   const double kappa = young / (3.0 * (1.0 - 2.0 * poisson));
   const double stiffness = kappa + 4.0 * mu / 3.0;
   const double omega = std::sqrt(3.0 * stiffness / density);
   const double initial_energy = 0.5 * stiffness * amplitude * amplitude;

   const std::unique_ptr<mfem::SparseMatrix> mass = cube.make_mass();
   auto nonlinear_solver = cube.make_nonlinear_solver(1.0);
   nonlinear_solver->initialize(0.0, disp, velo, acce);

   const int num_steps = static_cast<int>(std::lround(0.02 / dt));
   double mode_amplitude = amplitude;
   mfem::Vector momentum(velo.Size());
   for (int step = 0; step < num_steps; step++)
   {
      solve_step(*nonlinear_solver, step * dt, dt, disp, velo, acce);
      mode_amplitude = (mode * disp) / mode_norm;
      const double discrete_amplitude =
         amplitude * std::cos((step + 1) * 2.0 * std::atan(omega * dt / 2.0));
      check(std::abs(mode_amplitude - discrete_amplitude) < 2.0e-5 * amplitude,
            "Free vibration: not the discrete midpoint solution.");

      mass->Mult(velo, momentum);
      const double energy =
         0.5 * (velo * momentum) + 0.5 * stiffness * mode_amplitude * mode_amplitude;
      check(std::abs(energy / initial_energy - 1.0) < 2.0e-5,
            "Free vibration: the energy is not conserved.");
   }
   return std::abs(mode_amplitude / amplitude - std::cos(omega * 0.02));
}

// The cube fixed on the left face and driven along z on the right one by
// LoadData::disp_driven, -0.5 t: after one step, u_z = -0.5 t, v_z = -0.5
// and a_z = 0 there.
static void check_prescribed_motion()
{
   UnitCube cube("{fixed_bc: [{face: left, dir: x}, {face: left, dir: y}, {face: left, dir: z}],"
                 " disp_bc: [{face: right, dir: z}]}", "{faces: []}");
   mfem::GridFunction disp(&cube.space_u), velo(&cube.space_u), acce(&cube.space_u);
   disp = 0.0;
   velo = 0.0;

   auto nonlinear_solver = cube.make_nonlinear_solver(0.5);
   nonlinear_solver->initialize(0.0, disp, velo, acce);
   const double dt = 0.001;
   solve_step(*nonlinear_solver, 0.0, dt, disp, velo, acce);

   mfem::Array<int> right_dofs;
   cube.space_u.GetEssentialTrueDofs(
      cube.mesh.bdr_attribute_sets.GetAttributeSetMarker("right"), right_dofs, 2);
   for (int dof : right_dofs)
      check(std::abs(disp(dof) + 0.5 * dt) < 1.0e-14 && std::abs(velo(dof) + 0.5) < 1.0e-12 &&
            std::abs(acce(dof)) < 1.0e-8,
            "Prescribed motion: wrong displacement, velocity or acceleration.");
}

// A free cube loaded on the right face by LoadData::surface_traction. The
// internal forces sum to zero, so with rho_inf = 1 the momentum along z
// grows by the impulse of the traction at the midpoints,
//    P_{n+1} - P_n = dt F_z(t_n + dt / 2),
// with F_z the total traction force along z.
static void check_traction_impulse()
{
   UnitCube cube("{fixed_bc: [], disp_bc: []}", "{faces: [right]}");
   mfem::GridFunction disp(&cube.space_u), velo(&cube.space_u), acce(&cube.space_u);
   disp = 0.0;
   velo = 0.0;

   const std::unique_ptr<mfem::SparseMatrix> mass = cube.make_mass();
   const mfem::GridFunction unit_z = cube.make_uniform_field(0.0, 0.0, 1.0);

   // F_z(t), from the nodal traction forces.
   NeumannBoundary neumann(cube.neumann_config, cube.space_u);
   mfem::LinearForm traction_force(&cube.space_u);
   neumann.add_traction_integrators(traction_force);
   auto get_force_z = [&](double tt)
   {
      neumann.set_time(tt);
      traction_force.Assemble();
      return unit_z * traction_force;
   };

   auto nonlinear_solver = cube.make_nonlinear_solver(1.0);
   nonlinear_solver->initialize(0.0, disp, velo, acce);
   const double dt = 0.001;
   double expected_momentum = 0.0;
   mfem::Vector momentum(velo.Size());
   for (int step = 0; step < 3; step++)
   {
      solve_step(*nonlinear_solver, step * dt, dt, disp, velo, acce);
      expected_momentum += dt * get_force_z((step + 0.5) * dt);
      mass->Mult(velo, momentum);
      check(std::abs(unit_z * momentum - expected_momentum) < 1.0e-10,
            "Traction impulse: the momentum differs from the impulse.");
   }
}

int main()
{
   try
   {
      check_tangent();
      check_rigid_translation();
      const double coarse_error = check_free_vibration(0.001);
      const double fine_error = check_free_vibration(0.0005);
      check(coarse_error / fine_error > 3.5 && coarse_error / fine_error < 4.5,
            "Free vibration: the time convergence is not of second order.");
      check_prescribed_motion();
      check_traction_impulse();
      std::cout << "PASS: tangent, rigid translation, free vibration, prescribed motion, "
                   "traction impulse\n";
   }
   catch (const std::exception &error)
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
   return 0;
}

// ============================================================================
// dynamic_mixed.cpp of tests
//
// Checks of the mixed displacement-pressure dynamics on one Q2/Q1 hexahedron,
// the unit cube, with the material and the density of MaterialModelData:
//    tangent            K_eff against central differences of R_dyn, through
//                       Mult and GetGradient, without and with constraints,
//                       for rho_inf = 0, 0.5, 1;
//    rigid translation  constant velocity, zero acceleration and pressure,
//                       the output times and files of the time solver;
//    free vibration     a small longitudinal vibration: its energy, the
//                       pressure constraint at t_{n+1}, and the second-order
//                       self-convergence in time;
//    prescribed motion  disp_bc on the right face, LoadData::disp_loading and
//                       velo_loading;
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
#include "GlobalAssembly_Mixed.hpp"
#include "LoadData.hpp"
#include "LocalAssembly_Mixed.hpp"
#include "MaterialModelData.hpp"
#include "NeumannBoundary.hpp"
#include "NonlinearSolver_Dynamic_Mixed.hpp"
#include "TimeMethod_GenAlpha.hpp"
#include "TimeSolver_Dynamic_Mixed.hpp"

// Newton settings of every check.
static const char *newton_settings =
   "{newton_rel_tol: 1.0e-10, newton_abs_tol: 1.0e-10, newton_max_iter: 20}";

// Throws the message if the condition fails.
static void check(bool condition, const std::string &message)
{
   if (!condition)
      throw std::runtime_error(message);
}

// The unit cube, its faces left (x = 0), right (x = 1) and all, the
// displacement and pressure spaces, and the boundary conditions of a check.
struct UnitCube
{
   mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON);
   mfem::H1_FECollection fec_u{2, 3}, fec_p{1, 3};
   mfem::FiniteElementSpace space_u{&mesh, &fec_u, 3, mfem::Ordering::byVDIM};
   mfem::FiniteElementSpace space_p{&mesh, &fec_p};
   YAML::Node dirichlet_config;
   YAML::Node neumann_config;
   const std::filesystem::path results_dir =
      std::filesystem::temp_directory_path() / "mfem-solids-test-dynamic-mixed";

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

   std::unique_ptr<GlobalAssembly_Mixed> make_global_assembly()
   {
      return std::make_unique<GlobalAssembly_Mixed>(space_u, space_p,
         std::make_unique<LocalAssembly_Mixed>(set_material_model()),
         std::make_unique<DirichletBoundary>(dirichlet_config, space_u),
         std::make_unique<NeumannBoundary>(neumann_config, space_u));
   }

   std::unique_ptr<NonlinearSolver_Dynamic_Mixed> make_nonlinear_solver(double rho_inf)
   {
      return std::make_unique<NonlinearSolver_Dynamic_Mixed>(make_global_assembly(),
         std::make_unique<TimeMethod_GenAlpha>(rho_inf), YAML::Load(newton_settings));
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

// One step with the nonlinear solver from the state disp, velo, acce, pres
// into itself.
static void solve_step(NonlinearSolver_Dynamic_Mixed &nonlinear_solver, double tt, double dt,
                       mfem::GridFunction &disp, mfem::GridFunction &velo,
                       mfem::GridFunction &acce, mfem::GridFunction &pres)
{
   const mfem::GridFunction disp_old(disp), velo_old(velo), acce_old(acce), pres_old(pres);
   nonlinear_solver.solve(tt, dt, disp_old, velo_old, acce_old, pres_old,
                          disp, velo, acce, pres);
}

// K_eff against the central difference of R_dyn along a direction dd,
//    ( R_dyn(u + eps dd) - R_dyn(u - eps dd) ) / (2 eps) = K_eff dd + O(eps^2),
// through Mult and GetGradient, which keep the step of the last solve, with
// u and p together: on a free cube, at every dof, and on a cube fixed along x
// on the left face, for a dd zero on the constrained dofs, which carry the
// identity.
static void check_tangent()
{
   for (const char *dirichlet_input : {"{fixed_bc: [], disp_bc: []}",
                                       "{fixed_bc: [{face: left, dir: x}], disp_bc: []}"})
   {
      UnitCube cube(dirichlet_input, "{faces: []}");
      const int num_dofs_u = cube.space_u.GetTrueVSize();
      const int num_dofs = num_dofs_u + cube.space_p.GetTrueVSize();
      mfem::GridFunction disp_n(&cube.space_u), velo_n(&cube.space_u), acce_n(&cube.space_u);
      mfem::GridFunction pres_n(&cube.space_p);
      mfem::Vector shift(num_dofs), direction(num_dofs);
      for (int ii = 0; ii < num_dofs_u; ii++)
      {
         disp_n(ii) = 0.001 * std::sin(ii + 1.0);
         velo_n(ii) = 0.01 * std::cos(ii + 1.0);
         acce_n(ii) = 0.1 * std::sin(2.0 * ii);
      }
      for (int ii = 0; ii < pres_n.Size(); ii++)
         pres_n(ii) = 100.0 * std::cos(0.5 * ii);
      for (int ii = 0; ii < num_dofs; ii++)
      {
         shift(ii) = (ii < num_dofs_u) ? 0.002 * std::cos(ii) : 50.0 * std::sin(ii);
         direction(ii) = std::sin(0.7 * ii + 0.3);
      }
      for (int dof : cube.make_global_assembly()->get_dirichlet().get_ess_tdof_list())
         direction(dof) = 0.0;
      const double eps = 1.0e-7;

      for (double rho_inf : {0.0, 0.5, 1.0})
      {
         // One step from the state n sets the step of Mult and GetGradient.
         auto nonlinear_solver = cube.make_nonlinear_solver(rho_inf);
         mfem::GridFunction disp(disp_n), velo(velo_n), acce(acce_n), pres(pres_n);
         nonlinear_solver->solve(0.0, 0.01, disp_n, velo_n, acce_n, pres_n,
                                 disp, velo, acce, pres);

         // A state [u; p] away from the solution of the step.
         mfem::Vector state(num_dofs);
         for (int ii = 0; ii < num_dofs; ii++)
            state(ii) = (ii < num_dofs_u) ? disp(ii) : pres(ii - num_dofs_u);
         state += shift;
         mfem::Vector state_plus(state), state_minus(state);
         state_plus.Add(eps, direction);
         state_minus.Add(-eps, direction);

         mfem::Vector residual_plus(num_dofs), residual_minus(num_dofs), tangent_dir(num_dofs);
         nonlinear_solver->Mult(state_plus, residual_plus);
         nonlinear_solver->Mult(state_minus, residual_minus);
         nonlinear_solver->GetGradient(state).Mult(direction, tangent_dir);
         residual_plus -= residual_minus;
         residual_plus /= 2.0 * eps;
         residual_plus -= tangent_dir;
         check(residual_plus.Norml2() < 1.0e-7 * tangent_dir.Norml2(),
               "The tangent differs from the central difference of the residual.");
      }
   }
}

// A free cube moving with a constant velocity v: u = v t, a = 0 and p = 0,
// at the times 0, 0.003, 0.006, 0.009 and the shortened last step to 0.01.
static void check_rigid_translation()
{
   UnitCube cube("{fixed_bc: [], disp_bc: []}", "{faces: []}");
   const mfem::GridFunction velocity = cube.make_uniform_field(0.02, -0.01, 0.03);
   mfem::GridFunction disp(&cube.space_u), velo(velocity), acce(&cube.space_u);
   mfem::GridFunction pres(&cube.space_p);
   disp = 0.0;
   acce = 0.0;
   pres = 0.0;

   TimeSolver_Dynamic_Mixed time_solver(cube.make_nonlinear_solver(0.5), 0.003, 0.01,
                                        cube.results_dir);
   time_solver.run(disp, velo, acce, pres);

   mfem::GridFunction error(velocity);
   error *= 0.01;
   error -= disp;
   check(error.Normlinf() < 1.0e-12, "Rigid translation: wrong displacement.");
   error = velocity;
   error -= velo;
   check(error.Normlinf() < 1.0e-10, "Rigid translation: wrong velocity.");
   check(acce.Normlinf() < 1.0e-8, "Rigid translation: nonzero acceleration.");
   check(pres.Normlinf() < 1.0e-6, "Rigid translation: nonzero pressure.");

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
   check(num_gf_files == 20, "Not 5 steps of disp, velo, acce and pres saved.");
}

// The cube with u_y = u_z = 0 everywhere and u_x = 0 on the left face, from
// u_x = q x, small, here q = 1e-6, at rest, with the uniform pressure
// p_0 = p(1 + q) of J = 1 + q: a longitudinal vibration of the two layers of
// nodes of Q2, at x = 1/2 and x = 1. It is linear up to round-off, so with
// rho_inf = 1, the midpoint rule, the constraint R_p = 0 holds at every
// t_{n+1}, and the energy
//    E = 1/2 v^T M v + 1/2 u^T R_u(u,p),
// the kinetic and the strain energy of the condensed stiffness, stays
// constant. Returns the displacement at t = 0.02, for the convergence check.
static mfem::Vector check_free_vibration(double dt)
{
   UnitCube cube("{fixed_bc: [{face: left, dir: x}, {face: all, dir: y}, {face: all, dir: z}],"
                 " disp_bc: []}", "{faces: []}");
   const double amplitude = 1.0e-6;
   mfem::VectorFunctionCoefficient mode_value(3, [](const mfem::Vector &pt, mfem::Vector &value)
   {
      value = 0.0;
      value(0) = pt(0);
   });
   mfem::GridFunction disp(&cube.space_u), velo(&cube.space_u), acce(&cube.space_u);
   mfem::GridFunction pres(&cube.space_p);
   disp.ProjectCoefficient(mode_value);
   disp *= amplitude;
   velo = 0.0;
   pres = set_vol_model()->get_p(1.0 + amplitude);

   const std::unique_ptr<GlobalAssembly_Mixed> global_assembly = cube.make_global_assembly();
   const std::unique_ptr<mfem::SparseMatrix> mass = global_assembly->assemble_mass();
   mfem::BlockVector sol(global_assembly->get_offsets()), residual(global_assembly->get_offsets());
   mfem::Vector momentum(velo.Size());
   auto get_energy = [&]()
   {
      sol.GetBlock(0) = disp;
      sol.GetBlock(1) = pres;
      global_assembly->assemble_residual(sol, residual);
      mass->Mult(velo, momentum);
      return 0.5 * (velo * momentum) + 0.5 * (disp * residual.GetBlock(0));
   };
   const double initial_energy = get_energy();

   auto nonlinear_solver = cube.make_nonlinear_solver(1.0);
   nonlinear_solver->initialize(0.0, disp, velo, acce, pres);

   const int num_steps = static_cast<int>(std::lround(0.02 / dt));
   for (int step = 0; step < num_steps; step++)
   {
      solve_step(*nonlinear_solver, step * dt, dt, disp, velo, acce, pres);
      check(std::abs(get_energy() / initial_energy - 1.0) < 1.0e-5,
            "Free vibration: the energy is not conserved.");
      check(residual.GetBlock(1).Normlinf() < 1.0e-8 * amplitude,
            "Free vibration: the pressure constraint fails at t_{n+1}.");
   }
   return disp;
}

// The cube fixed on the left face and driven along z on the right one by
// LoadData::disp_loading, linear in time, with the velocity
// LoadData::velo_loading: after one step, u_z and v_z there are those of
// LoadData, and a_z = 0.
static void check_prescribed_motion()
{
   UnitCube cube("{fixed_bc: [{face: left, dir: x}, {face: left, dir: y}, {face: left, dir: z}],"
                 " disp_bc: [{face: right, dir: z}]}", "{faces: []}");
   mfem::GridFunction disp(&cube.space_u), velo(&cube.space_u), acce(&cube.space_u);
   mfem::GridFunction pres(&cube.space_p);
   disp = 0.0;
   velo = 0.0;
   pres = 0.0;

   auto nonlinear_solver = cube.make_nonlinear_solver(0.5);
   nonlinear_solver->initialize(0.0, disp, velo, acce, pres);
   const double dt = 0.001;
   solve_step(*nonlinear_solver, 0.0, dt, disp, velo, acce, pres);

   // The vertices of the right face, x = 1, are its first z dofs in Q2.
   for (int vv = 0; vv < cube.mesh.GetNV(); vv++)
   {
      mfem::Vector pt(cube.mesh.GetVertex(vv), 3);
      if (std::abs(pt(0) - 1.0) > 1.0e-12)
         continue;
      const int dof = cube.space_u.DofToVDof(vv, 2);
      const double disp_z = LoadData::disp_loading(pt, dt, "right")(2);
      const double velo_z = LoadData::velo_loading(pt, dt, "right")(2);
      check(std::abs(disp(dof) - disp_z) < 1.0e-14 && std::abs(velo(dof) - velo_z) < 1.0e-12 &&
            std::abs(acce(dof)) < 1.0e-8,
            "Prescribed motion: wrong displacement, velocity or acceleration.");
   }
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
   mfem::GridFunction pres(&cube.space_p);
   disp = 0.0;
   velo = 0.0;
   pres = 0.0;

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
   nonlinear_solver->initialize(0.0, disp, velo, acce, pres);
   const double dt = 0.001;
   double expected_momentum = 0.0;
   mfem::Vector momentum(velo.Size());
   for (int step = 0; step < 3; step++)
   {
      solve_step(*nonlinear_solver, step * dt, dt, disp, velo, acce, pres);
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
      // Self-convergence: the differences of the displacements at t = 0.02
      // with dt, dt / 2 and dt / 4 fall by 4.
      const mfem::Vector disp_coarse = check_free_vibration(0.001);
      const mfem::Vector disp_medium = check_free_vibration(0.0005);
      const mfem::Vector disp_fine = check_free_vibration(0.00025);
      mfem::Vector coarse_error(disp_coarse), fine_error(disp_medium);
      coarse_error -= disp_medium;
      fine_error -= disp_fine;
      const double ratio = coarse_error.Norml2() / fine_error.Norml2();
      check(ratio > 3.5 && ratio < 4.5,
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

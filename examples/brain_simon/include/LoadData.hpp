// ============================================================================
// LoadData.hpp
//
// The loading of brain_simon: the face "skull" moves as the rigid skull of
// SIMon, while the body starts at rest, without body force or traction. The
// face "foramen", the shell of the foramen magnum, of E = 6.9 GPa against
// the kPa of the brain, may move with it too, as a rigid closure.
//
// SIMon drives its rigid skull by *BOUNDARY_PRESCRIBED_MOTION_RIGID of
// Simon.k: the velocity of its center of mass c along the global axes, load
// curves 3, 4, 5, and its angular velocity omega about the global axes,
// curves 6, 7, 8, read by read_motion from a keyword file of
// *DEFINE_CURVE, e.g. the loadcurv_thor9042P.txt of NHTSA. A curve is
//    LCID, SIDR, SFA, SFO, ...
//    t_1  f_1
//    t_2  f_2  ...
// in the time SFA t_i and the value SFO f_i, piecewise linear between the
// points, as in LS-Dyna; the velocities are then multiplied by
// length_scale, from the lengths of the keyword file to those of the mesh.
//
// The motion of a point X of the skull is
//    x(t) = c(t) + R(t) (X - c_0),        c(t) = c_0 + int_0^t v_c dt,
//    R' = [omega] R, R(0) = I,            v(t) = v_c(t) + omega(t) x R (X - c_0),
// with c_0 the center of mass of the skull in the reference configuration.
// c(t) and R(t) are integrated on the points of all curves, each interval
// cut into substeps; on a substep h the curves are linear, so that
//    c += h v_c(t + h/2),   R <- exp(h [omega(t + h/2)]) R,
// exact for c and of order h^3 for R, by unit quaternions. A time between
// substeps is reached by one such step from the substep before it.
//
// Author: Chongran Zhao
// Date: Nov. 25, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "Vector_3D.hpp"

class LoadData
{
public:
   // Reads the curves of the velocity of the center, translation_curves for
   // x, y, z, and of the angular velocity, rotation_curves, from the keyword
   // file, and integrates the motion; center is c_0 in the units of the
   // mesh. Needed before the loading functions are called.
   static void read_motion(const std::string &file, const std::array<int, 3> &translation_curves,
                           const std::array<int, 3> &rotation_curves, double length_scale,
                           const std::array<double, 3> &center, int substeps = 10)
   {
      const std::map<int, curve> curves = read_curves(file);
      for (int axis = 0; axis < 3; axis++)
      {
         MFEM_VERIFY(curves.count(translation_curves[axis]) && curves.count(rotation_curves[axis]),
                     "A curve of the motion is not in \"" << file << "\".");
         velocity[axis] = curves.at(translation_curves[axis]);
         for (double &value : velocity[axis].values)
            value *= length_scale;
         angular_velocity[axis] = curves.at(rotation_curves[axis]);
      }
      center_0 = center;

      // The motion holds from the latest start to the earliest end of the
      // curves; the points of all curves in it, each interval in substeps.
      start_time = -1.0e300;
      end_time = 1.0e300;
      std::vector<double> points;
      for (const std::array<curve, 3> *curves_3 : {&velocity, &angular_velocity})
         for (const curve &cc : *curves_3)
         {
            start_time = std::max(start_time, cc.times.front());
            end_time = std::min(end_time, cc.times.back());
            points.insert(points.end(), cc.times.begin(), cc.times.end());
         }
      MFEM_VERIFY(start_time < end_time, "The curves of the motion do not overlap in time.");
      std::sort(points.begin(), points.end());
      points.erase(std::unique(points.begin(), points.end()), points.end());

      times = {start_time};
      for (const double point : points)
      {
         if (point <= start_time || point > end_time)
            continue;
         const double begin = times.back();
         for (int ss = 1; ss <= substeps; ss++)
            times.push_back(begin + (point - begin) * ss / substeps);
      }

      quaternions = {{1.0, 0.0, 0.0, 0.0}};
      centers = {center_0};
      for (std::size_t kk = 1; kk < times.size(); kk++)
      {
         const auto [quaternion, center_k] =
            step(quaternions.back(), centers.back(), times[kk - 1], times[kk]);
         quaternions.push_back(quaternion);
         centers.push_back(center_k);
      }
   }

   // The start and the end of the motion; the time of the driver is that of
   // the curves, starting from get_start_time.
   static double get_start_time() { return start_time; }
   static double get_end_time() { return end_time; }

   // The body starts at rest, as SIMon has no initial velocity.
   static Vector_3D initial_velo(const mfem::Vector &pt)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // No body force.
   static Vector_3D body_force(const mfem::Vector &pt, double tt)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // No traction.
   static Vector_3D surface_traction(const mfem::Vector &pt, double tt,
                                     const std::string &face)
   {
      return Vector_3D(0.0, 0.0, 0.0);
   }

   // The rigid motion of the skull, u = c(tt) + R(tt) (X - c_0) - X.
   static Vector_3D disp_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      check_face(face);
      const auto [rotation, center] = get_motion(tt);
      const std::array<double, 3> arm = rotate(rotation, get_arm(pt));
      return Vector_3D(center[0] + arm[0] - pt(0), center[1] + arm[1] - pt(1),
                       center[2] + arm[2] - pt(2));
   }

   // Its velocity, v = v_c(tt) + omega(tt) x R(tt) (X - c_0).
   static Vector_3D velo_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      check_face(face);
      const auto [rotation, center] = get_motion(tt);
      const std::array<double, 3> arm = rotate(rotation, get_arm(pt));
      const std::array<double, 3> omega = evaluate(angular_velocity, tt);
      const std::array<double, 3> velo_c = evaluate(velocity, tt);
      return Vector_3D(velo_c[0] + omega[1] * arm[2] - omega[2] * arm[1],
                       velo_c[1] + omega[2] * arm[0] - omega[0] * arm[2],
                       velo_c[2] + omega[0] * arm[1] - omega[1] * arm[0]);
   }

   // The rotation R(tt), by rows, and the center c(tt); for checks and output.
   static std::pair<std::array<double, 9>, std::array<double, 3>> get_motion(double tt)
   {
      MFEM_VERIFY(!times.empty(), "LoadData::read_motion was not called.");
      const double tolerance = 1.0e-12 * std::max(1.0, std::abs(end_time));
      MFEM_VERIFY(tt >= start_time - tolerance && tt <= end_time + tolerance,
                  "The time " << tt << " is outside the motion, [" << start_time << ", "
                  << end_time << "].");
      tt = std::clamp(tt, start_time, end_time);
      const std::size_t kk =
         std::upper_bound(times.begin(), times.end(), tt) - times.begin() - 1;
      const auto [quaternion, center] = step(quaternions[kk], centers[kk], times[kk], tt);
      return {get_rotation(quaternion), center};
   }

private:
   // The faces that move with the skull.
   static void check_face(const std::string &face)
   {
      MFEM_VERIFY(face == "skull" || face == "foramen", "Unknown face \"" << face << "\".");
   }

   // A load curve, its points scaled by SFA and SFO.
   struct curve
   {
      std::vector<double> times;
      std::vector<double> values;
   };

   // The curves of *DEFINE_CURVE in the keyword file, by LCID; the lines of
   // $ are comments, the cards free (commas) or fixed (20 wide) format.
   static std::map<int, curve> read_curves(const std::string &file)
   {
      std::ifstream input(file);
      MFEM_VERIFY(input, "Cannot open the curve file \"" << file << "\".");
      std::map<int, curve> curves;
      std::string line;
      bool in_curve = false;
      int lcid = 0;
      double sfa = 1.0, sfo = 1.0;
      while (std::getline(input, line))
      {
         if (!line.empty() && line.back() == '\r')
            line.pop_back();
         if (line.empty() || line[0] == '$')
            continue;
         if (line[0] == '*')
         {
            in_curve = (line.rfind("*DEFINE_CURVE", 0) == 0);
            lcid = 0;
            continue;
         }
         if (!in_curve)
            continue;
         std::replace(line.begin(), line.end(), ',', ' ');
         std::stringstream stream(line);
         if (lcid == 0)
         {
            double sidr = 0.0;
            stream >> lcid >> sidr;
            sfa = 1.0;
            sfo = 1.0;
            double value;
            if (stream >> value && value != 0.0)
               sfa = value;
            if (stream >> value && value != 0.0)
               sfo = value;
            curves[lcid] = curve();
            continue;
         }
         double time, value;
         MFEM_VERIFY(stream >> time >> value, "A point of curve " << lcid << " is not read.");
         curves[lcid].times.push_back(sfa * time);
         curves[lcid].values.push_back(sfo * value);
      }
      return curves;
   }

   // The three curves at tt, linear between their points.
   static std::array<double, 3> evaluate(const std::array<curve, 3> &curves_3, double tt)
   {
      std::array<double, 3> out;
      for (int axis = 0; axis < 3; axis++)
      {
         const curve &cc = curves_3[axis];
         const std::size_t upper = std::clamp<std::size_t>(
            std::upper_bound(cc.times.begin(), cc.times.end(), tt) - cc.times.begin(), 1,
            cc.times.size() - 1);
         const double weight = (tt - cc.times[upper - 1]) / (cc.times[upper] - cc.times[upper - 1]);
         out[axis] = (1.0 - weight) * cc.values[upper - 1] + weight * cc.values[upper];
      }
      return out;
   }

   // One step from time_0 to time_1 of the unit quaternion (w, x, y, z) of
   // R and of the center, by the velocities at the midpoint.
   static std::pair<std::array<double, 4>, std::array<double, 3>>
   step(const std::array<double, 4> &quaternion, const std::array<double, 3> &center,
        double time_0, double time_1)
   {
      const double hh = time_1 - time_0;
      const double mid = 0.5 * (time_0 + time_1);
      const std::array<double, 3> velo_c = evaluate(velocity, mid);
      const std::array<double, 3> omega = evaluate(angular_velocity, mid);

      std::array<double, 3> center_1;
      for (int axis = 0; axis < 3; axis++)
         center_1[axis] = center[axis] + hh * velo_c[axis];

      // exp(h [omega]) is the quaternion of the angle h |omega| about omega.
      const double norm = std::sqrt(omega[0] * omega[0] + omega[1] * omega[1]
                                    + omega[2] * omega[2]);
      const double half = 0.5 * hh * norm;
      const double factor = (norm > 0.0) ? std::sin(half) / norm : 0.5 * hh;
      const std::array<double, 4> turn = {std::cos(half), factor * omega[0], factor * omega[1],
                                          factor * omega[2]};
      const std::array<double, 4> &qq = quaternion;
      std::array<double, 4> product = {
         turn[0] * qq[0] - turn[1] * qq[1] - turn[2] * qq[2] - turn[3] * qq[3],
         turn[0] * qq[1] + turn[1] * qq[0] + turn[2] * qq[3] - turn[3] * qq[2],
         turn[0] * qq[2] - turn[1] * qq[3] + turn[2] * qq[0] + turn[3] * qq[1],
         turn[0] * qq[3] + turn[1] * qq[2] - turn[2] * qq[1] + turn[3] * qq[0]};
      const double length = std::sqrt(product[0] * product[0] + product[1] * product[1]
                                      + product[2] * product[2] + product[3] * product[3]);
      for (double &entry : product)
         entry /= length;
      return {product, center_1};
   }

   // The rotation of the unit quaternion (w, x, y, z), by rows.
   static std::array<double, 9> get_rotation(const std::array<double, 4> &qq)
   {
      const double ww = qq[0], xx = qq[1], yy = qq[2], zz = qq[3];
      return {1.0 - 2.0 * (yy * yy + zz * zz), 2.0 * (xx * yy - ww * zz), 2.0 * (xx * zz + ww * yy),
              2.0 * (xx * yy + ww * zz), 1.0 - 2.0 * (xx * xx + zz * zz), 2.0 * (yy * zz - ww * xx),
              2.0 * (xx * zz - ww * yy), 2.0 * (yy * zz + ww * xx), 1.0 - 2.0 * (xx * xx + yy * yy)};
   }

   // X - c_0.
   static std::array<double, 3> get_arm(const mfem::Vector &pt)
   {
      return {pt(0) - center_0[0], pt(1) - center_0[1], pt(2) - center_0[2]};
   }

   // R a, R by rows.
   static std::array<double, 3> rotate(const std::array<double, 9> &rotation,
                                       const std::array<double, 3> &arm)
   {
      std::array<double, 3> out;
      for (int row = 0; row < 3; row++)
         out[row] = rotation[3 * row] * arm[0] + rotation[3 * row + 1] * arm[1]
                    + rotation[3 * row + 2] * arm[2];
      return out;
   }

   inline static std::array<curve, 3> velocity;           // v_c along x, y, z
   inline static std::array<curve, 3> angular_velocity;   // omega about x, y, z
   inline static std::array<double, 3> center_0{};        // c_0
   inline static double start_time = 0.0;                 // start of the motion
   inline static double end_time = 0.0;                   // end of the motion
   inline static std::vector<double> times;               // substeps of the integration
   inline static std::vector<std::array<double, 4>> quaternions;   // R at the substeps
   inline static std::vector<std::array<double, 3>> centers;       // c at the substeps
};

#endif

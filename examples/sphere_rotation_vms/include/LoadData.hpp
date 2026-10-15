// ============================================================================
// LoadData.hpp
//
// The loading of sphere_rotation: the surface of the ball, the face "outer",
// turns rigidly about the z axis through the origin by the measured angle,
// while the body starts at rest, without body force or traction. The angle
// theta(t) is a piecewise quintic polynomial fitted to the measurements,
// read by read_rotation from a CSV file of one row per interval,
//    t_start, t_end, c5, c4, c3, c2, c1, c0,
// in seconds, with theta(t) = sum_k c_k (t - t_start)^k on
// t_start <= t <= t_end. The surface turns by Omega(t) = theta(t) - theta(0),
// so that it starts from the reference configuration, at the angular
// velocity omega(t) = theta'(t); t must stay within the intervals of the file.
//
// Author: Chongran Zhao
// Date: Oct. 15, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "Vector_3D.hpp"

class LoadData
{
public:
   // Reads the intervals of theta(t) from the CSV file, its first line the
   // header; needed before the loading functions are called.
   static void read_rotation(const std::string &file)
   {
      std::ifstream input(file);
      MFEM_VERIFY(input, "Cannot open the rotation file \"" << file << "\".");
      std::string line;
      std::getline(input, line);
      intervals.clear();
      while (std::getline(input, line))
      {
         if (line.find_first_not_of(" \t\r") == std::string::npos)
            continue;
         std::stringstream stream(line);
         std::string entry;
         std::array<double, 8> values;
         for (double &value : values)
         {
            MFEM_VERIFY(std::getline(stream, entry, ','),
                        "A row of \"" << file << "\" has fewer than 8 columns.");
            value = std::stod(entry);
         }
         intervals.push_back({values[0], values[1],
                              {values[2], values[3], values[4], values[5], values[6], values[7]}});
      }
      MFEM_VERIFY(!intervals.empty(), "The rotation file \"" << file << "\" has no interval.");
      theta_0 = get_theta(intervals.front().t_start, 0);
   }

   // The end of the last interval, the last time of the loading.
   static double get_end_time() { return intervals.back().t_end; }

   // The body starts at rest.
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

   // Rigid rotation by Omega(tt) about z: u = R X - X.
   static Vector_3D disp_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      check_face(face);
      const double Omega = get_Omega(tt);
      const double cos_Omega = std::cos(Omega), sin_Omega = std::sin(Omega);
      return Vector_3D(cos_Omega * pt(0) - sin_Omega * pt(1) - pt(0),
                       sin_Omega * pt(0) + cos_Omega * pt(1) - pt(1), 0.0);
   }

   // Its velocity, v = omega(tt) e_z x R X.
   static Vector_3D velo_loading(const mfem::Vector &pt, double tt,
                                 const std::string &face)
   {
      check_face(face);
      const double Omega = get_Omega(tt);
      const double omega = get_omega(tt);
      const double cos_Omega = std::cos(Omega), sin_Omega = std::sin(Omega);
      return Vector_3D(-omega * (sin_Omega * pt(0) + cos_Omega * pt(1)),
                       omega * (cos_Omega * pt(0) - sin_Omega * pt(1)), 0.0);
   }

private:
   // The only face of the ball.
   static void check_face(const std::string &face)
   {
      MFEM_VERIFY(face == "outer", "Unknown face \"" << face << "\".");
   }

   // An interval of theta(t): its times and the coefficients c5, ..., c0
   // of the descending powers of t - t_start.
   struct interval
   {
      double t_start;
      double t_end;
      std::array<double, 6> coefficients;
   };

   // The derivative of order 0 or 1 of theta at tt, by Horner's rule on the
   // interval that holds tt.
   static double get_theta(double tt, int derivative)
   {
      MFEM_VERIFY(!intervals.empty(), "LoadData::read_rotation was not called.");
      const double tolerance = 1.0e-12 * intervals.back().t_end;
      MFEM_VERIFY(tt >= intervals.front().t_start - tolerance
                  && tt <= intervals.back().t_end + tolerance,
                  "The time " << tt << " is outside the rotation data.");
      std::size_t index = 0;
      while (index + 1 < intervals.size() && tt > intervals[index].t_end)
         index++;
      const interval &segment = intervals[index];
      const double tau = tt - segment.t_start;

      double out = 0.0;
      for (int kk = 0; kk < 6; kk++)
      {
         const int power = 5 - kk;
         if (power < derivative)
            break;
         const double factor = (derivative == 1) ? power : 1.0;
         out = out * tau + factor * segment.coefficients[kk];
      }
      return out;
   }

   // omega(tt) = theta'(tt).
   static double get_omega(double tt) { return get_theta(tt, 1); }

   // Omega(tt) = theta(tt) - theta(0).
   static double get_Omega(double tt) { return get_theta(tt, 0) - theta_0; }

   inline static std::vector<interval> intervals;   // the intervals of theta(t)
   inline static double theta_0 = 0.0;              // theta at the start of the data
};

#endif

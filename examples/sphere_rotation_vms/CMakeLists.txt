# ============================================================================
# sphere_rotation_vms: the gelatin ball in its skull shell of sphere_rotation_TaylorHood,
# in the rate form of the mixed formulation, equal-order elements with the VMS
# stabilization, solved by GMRES
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh && mpirun -np 4 ./driver && ./vtu_writer
#
# Post-processing of the VTU series, by the pvpython of ParaView, in build/
# (scripts/ describes the options in each file):
#    pvpython ../scripts/probe_strain.py results_vtu/results_vtu.pvd E_rphi.csv
#    pvpython ../scripts/plot_strain.py E_rtheta.png E_rphi.csv
#    pvpython ../scripts/wave_speed.py results_vtu/results_vtu.pvd wave_speed.png
#    pvpython ../scripts/probe_disp.py results_vtu/results_vtu.pvd U_phi.csv
# probe_strain.py writes E_r_phi at 20, 40, 60, 80 mm on the measured ray,
# 60 degrees from x in the equatorial plane; plot_strain.py draws such CSV
# files, one panel each; wave_speed.py fits the arrival of the first negative
# peak from 10 to 80 mm, for VTU output every 1 ms or so; probe_disp.py
# writes u_phi in mm at the measured radii 11.12, 32.31, 53.38 mm on that ray.
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(sphere_rotation_vms LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/mfem-solids.cmake)

# The headers of this case, in its include/; like the shared ones, listed
# for the record, not needed to build, and checked against the folder.
set(case_headers
  include/LoadData.hpp
  include/MaterialModelData.hpp
  include/NonlinearSolver.hpp
  include/TimeSolver.hpp)
check_header_list(${CMAKE_CURRENT_SOURCE_DIR}/include ${case_headers})

# config.yaml is copied into the build folder, again whenever it changes;
# the programs read the copy of the directory they run in.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)
configure_file(angle_coefficients.csv ${CMAKE_CURRENT_BINARY_DIR}/angle_coefficients.csv COPYONLY)

# The mesh generation of sphere_rotation_TaylorHood.
add_executable(read_mesh ../sphere_rotation_TaylorHood/read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})

add_executable(driver driver.cpp ${case_headers})
target_include_directories(driver PRIVATE include)
target_link_libraries(driver PRIVATE mfem_solids)

# vtu_writer writes spherical components, with VTK_Tools_Sphere.hpp.
add_executable(vtu_writer vtu_writer.cpp ${case_headers})
target_include_directories(vtu_writer PRIVATE include)
target_link_libraries(vtu_writer PRIVATE mfem_solids)

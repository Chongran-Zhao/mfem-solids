# ============================================================================
# sphere_rotation: a gelatin ball in a skull shell, spun up about z by its
# surface, in the mixed displacement-pressure form, one driver and input per
# formulation:
#    driver_TaylorHood_dynamic   config_TaylorHood_dynamic.yaml   Taylor-Hood, generalized-alpha
#    driver_vms_dynamic          config_vms_dynamic.yaml          VMS, rate form, GMRES
#
# Build and run in this folder, e.g. with Taylor-Hood:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh config_TaylorHood_dynamic.yaml && mpirun -np 4 ./driver_TaylorHood_dynamic
#    mpirun -np 4 ./vtu_writer config_TaylorHood_dynamic.yaml
#    ./probe_disp U_phi.csv config_TaylorHood_dynamic.yaml
#
# Post-processing of the VTU series, by the pvpython of ParaView, in build/
# (scripts/ describes the options in each file), e.g. of the VMS results:
#    pvpython ../scripts/probe_strain.py results_vms_dynamic_vtu/results_vms_dynamic_vtu.pvd E_rphi.csv
#    pvpython ../scripts/plot_strain.py E_rtheta.png E_rphi.csv
#    pvpython ../scripts/wave_speed.py results_vms_dynamic_vtu/results_vms_dynamic_vtu.pvd wave_speed.png
#    pvpython ../scripts/probe_disp.py results_vms_dynamic_vtu/results_vms_dynamic_vtu.pvd U_phi.csv
# probe_strain.py writes E_r_phi at 20, 40, 60, 80 mm on the measured ray,
# 60 degrees from x in the equatorial plane; plot_strain.py draws such CSV
# files, one panel each; wave_speed.py fits the arrival of the first negative
# peak from 10 to 80 mm, for VTU output every 1 ms or so; probe_disp.py
# writes u_phi in mm at the measured radii 11.12, 32.31, 53.38 mm on that ray.
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(sphere_rotation LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/mfem-solids.cmake)

# The headers of this case, in its include/; like the shared ones, listed
# for the record, not needed to build, and checked against the folder.
set(case_headers
  include/LoadData.hpp
  include/MaterialModelData.hpp
  include/NonlinearSolver.hpp
  include/NonlinearSolver_TaylorHood_Dynamic.hpp
  include/NonlinearSolver_VMS_Dynamic.hpp
  include/TimeSolver.hpp)
check_header_list(${CMAKE_CURRENT_SOURCE_DIR}/include ${case_headers})

# The inputs are copied into the build folder, again whenever they change;
# each driver reads its own by default, the other programs the one given as
# their first argument, probe_disp as its second.
configure_file(config_TaylorHood_dynamic.yaml ${CMAKE_CURRENT_BINARY_DIR}/config_TaylorHood_dynamic.yaml COPYONLY)
configure_file(config_vms_dynamic.yaml ${CMAKE_CURRENT_BINARY_DIR}/config_vms_dynamic.yaml COPYONLY)
configure_file(angle_coefficients.csv ${CMAKE_CURRENT_BINARY_DIR}/angle_coefficients.csv COPYONLY)

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})

add_executable(driver_TaylorHood_dynamic driver_TaylorHood_dynamic.cpp ${case_headers})
target_include_directories(driver_TaylorHood_dynamic PRIVATE include)
target_link_libraries(driver_TaylorHood_dynamic PRIVATE mfem_solids)

add_executable(driver_vms_dynamic driver_vms_dynamic.cpp ${case_headers})
target_include_directories(driver_vms_dynamic PRIVATE include)
target_link_libraries(driver_vms_dynamic PRIVATE mfem_solids)

# vtu_writer writes spherical components, with VTK_Tools_Sphere.hpp.
add_executable(vtu_writer vtu_writer.cpp ${case_headers})
target_include_directories(vtu_writer PRIVATE include)
target_link_libraries(vtu_writer PRIVATE mfem_solids)

# probe_disp writes u_phi of material points, from the fields the driver saved.
add_executable(probe_disp probe_disp.cpp)
target_link_libraries(probe_disp PRIVATE mfem_solids)

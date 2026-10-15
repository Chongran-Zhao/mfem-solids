# ============================================================================
# sphere_rotation_vms: the gelatin ball in its skull shell of sphere_rotation,
# in the rate form of the mixed formulation, equal-order elements with the VMS
# stabilization, solved by MUMPS
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh && mpirun -np 4 ./driver && ./vtu_writer
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

# The mesh generation and the VTU writer of sphere_rotation.
add_executable(read_mesh ../sphere_rotation/read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})

add_executable(driver driver.cpp ${case_headers})
target_include_directories(driver PRIVATE include)
target_link_libraries(driver PRIVATE mfem_solids)

# vtu_writer writes spherical components, with VTK_Tools_Sphere.hpp.
add_executable(vtu_writer ../sphere_rotation/vtu_writer.cpp ${case_headers})
target_include_directories(vtu_writer PRIVATE include)
target_link_libraries(vtu_writer PRIVATE mfem_solids)

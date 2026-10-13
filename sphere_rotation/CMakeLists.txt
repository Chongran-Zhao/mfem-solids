# ============================================================================
# sphere_rotation: a gelatin ball in a skull shell, spun up about z by its
# surface, in the mixed displacement-pressure form
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh && mpirun -np 4 ./driver && ./vtu_writer
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(sphere_rotation LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/mfem-solids.cmake)

# The headers of this case come first: its LoadData.hpp, which
# DirichletBoundary and NeumannBoundary include, and its MaterialModelData.hpp.
set(case_include_dirs ${CMAKE_CURRENT_SOURCE_DIR} ${include_dirs})

# config.yaml is copied into the build folder, again whenever it changes;
# the programs read the copy of the directory they run in.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})

add_executable(driver driver.cpp)
target_include_directories(driver PRIVATE ${case_include_dirs})
target_link_libraries(driver PRIVATE ${mfem_libraries})

# vtu_writer also writes the Green-Lagrange strain, with the VTK_Tools.hpp of
# this folder.
add_executable(vtu_writer vtu_writer.cpp)
target_include_directories(vtu_writer PRIVATE ${case_include_dirs})
target_link_libraries(vtu_writer PRIVATE ${mfem_libraries})

# ============================================================================
# sphere_rotation: an incompressible sphere spun up about z by its surface,
# in the mixed displacement-pressure form
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(sphere_rotation LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/mfem-solids.cmake)

# config.yaml is copied into the build folder, again whenever it changes;
# the programs read the copy of the directory they run in.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})

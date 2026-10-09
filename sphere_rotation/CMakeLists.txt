# ============================================================================
# sphere_rotation: an incompressible sphere spun up about z by its surface,
# in the mixed displacement-pressure form
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh && mpirun -np 2 ./driver && ./vtu_writer && mpirun -np 2 ./csv_writer
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

add_executable(driver driver.cpp)
target_include_directories(driver PRIVATE ${include_dirs})
target_link_libraries(driver PRIVATE ${mfem_libraries})

# vtu_writer also writes the Green-Lagrange strain, with the VTK_Tools.hpp of
# this folder.
add_executable(vtu_writer vtu_writer.cpp)
target_include_directories(vtu_writer PRIVATE ${include_dirs})
target_link_libraries(vtu_writer PRIVATE ${mfem_libraries})

# csv_writer is that of dynamic_mixed: it needs neither the loading nor the
# fixed pressure, and this folder, first on the include path, gives it the
# material of the sphere.
add_executable(csv_writer ../dynamic_mixed/csv_writer.cpp)
target_include_directories(csv_writer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${include_dirs})
target_link_libraries(csv_writer PRIVATE ${mfem_libraries})

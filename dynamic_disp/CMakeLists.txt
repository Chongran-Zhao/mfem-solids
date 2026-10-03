# ============================================================================
# dynamic_disp: hyperelastodynamics in the displacement form
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh && ./driver && ./vtu_writer && ./csv_writer
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(dynamic_disp LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/mfem-solids.cmake)

# config.yaml is copied into the build folder, again whenever it changes;
# the programs read the copy of the directory they run in.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})
# The mesh file is found from the project directory, the HTML template from
# this folder.
target_compile_definitions(read_mesh PRIVATE PROJECT_DIR="${project_dir}"
                                             SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(driver driver.cpp)
target_include_directories(driver PRIVATE ${include_dirs})
target_link_libraries(driver PRIVATE ${mfem_libraries})

add_executable(vtu_writer vtu_writer.cpp)
target_include_directories(vtu_writer PRIVATE ${include_dirs})
target_link_libraries(vtu_writer PRIVATE ${mfem_libraries})

add_executable(csv_writer csv_writer.cpp)
target_include_directories(csv_writer PRIVATE ${include_dirs})
target_link_libraries(csv_writer PRIVATE ${mfem_libraries})

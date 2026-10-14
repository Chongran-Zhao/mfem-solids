# ============================================================================
# static_mixed: hyperelastostatics in the mixed displacement-pressure form
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh && mpirun -np 2 ./driver && ./vtu_writer && mpirun -np 2 ./csv_writer
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(static_mixed LANGUAGES CXX)

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

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})
# The mesh file is found from the project directory, the HTML template from
# this folder.
target_compile_definitions(read_mesh PRIVATE PROJECT_DIR="${project_dir}"
                                             SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(driver driver.cpp ${case_headers})
target_include_directories(driver PRIVATE include)
target_link_libraries(driver PRIVATE mfem_solids)

add_executable(vtu_writer vtu_writer.cpp ${case_headers})
target_include_directories(vtu_writer PRIVATE include)
target_link_libraries(vtu_writer PRIVATE mfem_solids)

add_executable(csv_writer csv_writer.cpp ${case_headers})
target_include_directories(csv_writer PRIVATE include)
target_link_libraries(csv_writer PRIVATE mfem_solids)

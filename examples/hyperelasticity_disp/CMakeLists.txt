# ============================================================================
# hyperelasticity_disp: hyperelasticity in the displacement form, static or
# dynamic by config_static.yaml or config_dynamic.yaml
#
# Build and run in this folder, with config_static.yaml or config_dynamic.yaml:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh config_static.yaml && mpirun -np 2 ./driver config_static.yaml
#    mpirun -np 2 ./vtu_writer config_static.yaml && mpirun -np 2 ./csv_writer config_static.yaml
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(hyperelasticity_disp LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/mfem-solids.cmake)

# The headers of this case, in its include/; like the shared ones, listed
# for the record, not needed to build, and checked against the folder.
set(case_headers
  include/LoadData.hpp
  include/MaterialModelData.hpp
  include/NonlinearSolver.hpp
  include/NonlinearSolver_Dynamic.hpp
  include/NonlinearSolver_Static.hpp
  include/TimeSolver.hpp)
check_header_list(${CMAKE_CURRENT_SOURCE_DIR}/include ${case_headers})

# The two inputs are copied into the build folder, again whenever they
# change; the programs read the one given as their first argument.
configure_file(config_static.yaml ${CMAKE_CURRENT_BINARY_DIR}/config_static.yaml COPYONLY)
configure_file(config_dynamic.yaml ${CMAKE_CURRENT_BINARY_DIR}/config_dynamic.yaml COPYONLY)

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

# ============================================================================
# fracture_bar: quasi-static phase-field fracture by alternate minimization,
# the bar in traction of Bourdin, Francfort and Marigo (J. Elasticity 2008,
# Section 8.3.1), by driver and config.yaml
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh config.yaml && mpirun -np 2 ./driver config.yaml
#    mpirun -np 2 ./vtu_writer config.yaml
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(fracture_bar LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/mfem-solids.cmake)

# The headers of this case, in its include/; like the shared ones, listed
# for the record, not needed to build, and checked against the folder.
set(case_headers
  include/LoadData.hpp
  include/MaterialModelData.hpp
  include/NonlinearSolver_Disp.hpp
  include/NonlinearSolver_Fracture.hpp
  include/TimeSolver.hpp
  include/VTK_Tools_Fracture.hpp)
check_header_list(${CMAKE_CURRENT_SOURCE_DIR}/include ${case_headers})

# The input is copied into the build folder, again whenever it changes; the
# programs read the one given as their first argument.
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

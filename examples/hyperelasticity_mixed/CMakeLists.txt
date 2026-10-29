# ============================================================================
# hyperelasticity_mixed: hyperelasticity in the mixed displacement-pressure
# form, one driver and input per formulation and time method:
#    driver_TaylorHood_static    config_TaylorHood_static.yaml    Taylor-Hood, statics
#    driver_TaylorHood_dynamic   config_TaylorHood_dynamic.yaml   Taylor-Hood, generalized-alpha
#    driver_vms_dynamic          config_vms_dynamic.yaml          VMS, rate form
#
# Build and run in this folder, e.g. the Taylor-Hood statics:
#    cmake -B build && cmake --build build
#    cd build && ./read_mesh config_TaylorHood_static.yaml && mpirun -np 2 ./driver_TaylorHood_static
#    mpirun -np 2 ./vtu_writer config_TaylorHood_static.yaml
#    mpirun -np 2 ./csv_writer config_TaylorHood_static.yaml
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(hyperelasticity_mixed LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/mfem-solids.cmake)

# The headers of this case, in its include/; like the shared ones, listed
# for the record, not needed to build, and checked against the folder.
set(case_headers
  include/LoadData.hpp
  include/MaterialModelData.hpp
  include/NonlinearSolver.hpp
  include/NonlinearSolver_TaylorHood_Dynamic.hpp
  include/NonlinearSolver_TaylorHood_Static.hpp
  include/NonlinearSolver_VMS_Dynamic.hpp
  include/TimeSolver.hpp)
check_header_list(${CMAKE_CURRENT_SOURCE_DIR}/include ${case_headers})

# The inputs are copied into the build folder, again whenever they change;
# each driver reads its own by default, the other programs the one given as
# their first argument.
configure_file(config_TaylorHood_static.yaml ${CMAKE_CURRENT_BINARY_DIR}/config_TaylorHood_static.yaml COPYONLY)
configure_file(config_TaylorHood_dynamic.yaml ${CMAKE_CURRENT_BINARY_DIR}/config_TaylorHood_dynamic.yaml COPYONLY)
configure_file(config_vms_dynamic.yaml ${CMAKE_CURRENT_BINARY_DIR}/config_vms_dynamic.yaml COPYONLY)

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})
# The mesh file is found from the project directory, the HTML template from
# this folder.
target_compile_definitions(read_mesh PRIVATE PROJECT_DIR="${project_dir}"
                                             SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(driver_TaylorHood_static driver_TaylorHood_static.cpp ${case_headers})
target_include_directories(driver_TaylorHood_static PRIVATE include)
target_link_libraries(driver_TaylorHood_static PRIVATE mfem_solids)

add_executable(driver_TaylorHood_dynamic driver_TaylorHood_dynamic.cpp ${case_headers})
target_include_directories(driver_TaylorHood_dynamic PRIVATE include)
target_link_libraries(driver_TaylorHood_dynamic PRIVATE mfem_solids)

add_executable(driver_vms_dynamic driver_vms_dynamic.cpp ${case_headers})
target_include_directories(driver_vms_dynamic PRIVATE include)
target_link_libraries(driver_vms_dynamic PRIVATE mfem_solids)

add_executable(vtu_writer vtu_writer.cpp ${case_headers})
target_include_directories(vtu_writer PRIVATE include)
target_link_libraries(vtu_writer PRIVATE mfem_solids)

add_executable(csv_writer csv_writer.cpp ${case_headers})
target_include_directories(csv_writer PRIVATE include)
target_link_libraries(csv_writer PRIVATE mfem_solids)

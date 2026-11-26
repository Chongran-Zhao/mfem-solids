# ============================================================================
# brain_simon: the brain of the SIMon model of NHTSA, the benchmark of
# Giudice et al. (2019), Ann. Biomed. Eng. 47:1855-1872, incompressible
# Neo-Hookean in place of its viscoelastic materials:
#    read_kfile                  config.yaml   Simon.k (LS-Dyna) to simon.mesh (MFEM)
#    driver_TaylorHood_dynamic   config.yaml   Taylor-Hood, generalized-alpha, GMRES,
#                                              the skull moved by the load curves
#                                              of loading.motion_file
#    vtu_writer                  config.yaml   the saved fields to VTU, for ParaView
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_kfile config.yaml && mpirun -np 8 ./driver_TaylorHood_dynamic
#    mpirun -np 8 ./vtu_writer
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(brain_simon LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/mfem-solids.cmake)

# The headers of this case, in its include/; listed for the record, not
# needed to build, and checked against the folder.
set(case_headers
  include/LoadData.hpp
  include/MaterialModelData.hpp
  include/NonlinearSolver.hpp
  include/NonlinearSolver_TaylorHood_Dynamic.hpp
  include/TimeSolver.hpp)
check_header_list(${CMAKE_CURRENT_SOURCE_DIR}/include ${case_headers})

# The input is copied into the build folder, again whenever it changes.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(read_kfile read_kfile.cpp)
target_include_directories(read_kfile PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_kfile PRIVATE ${mfem_libraries})

add_executable(driver_TaylorHood_dynamic driver_TaylorHood_dynamic.cpp ${case_headers})
target_include_directories(driver_TaylorHood_dynamic PRIVATE include)
target_link_libraries(driver_TaylorHood_dynamic PRIVATE mfem_solids)

add_executable(vtu_writer vtu_writer.cpp ${case_headers})
target_include_directories(vtu_writer PRIVATE include)
target_link_libraries(vtu_writer PRIVATE mfem_solids)

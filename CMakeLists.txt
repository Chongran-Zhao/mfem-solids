# ============================================================================
# mfem-solids
#
# Build:
#   cmake -B build
#   cmake --build build
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(mfem-solids LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Libraries installed in MFEM/lib.
set(lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../lib")

find_package(MFEM REQUIRED HINTS "${lib_dir}/mfem/build")
message(STATUS "MFEM ${MFEM_VERSION} at ${MFEM_DIR}")

if(MFEM_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

find_package(yaml-cpp REQUIRED CONFIG HINTS "${lib_dir}/yaml-cpp/lib/cmake/yaml-cpp")
message(STATUS "yaml-cpp at ${yaml-cpp_DIR}")

set(mfem_libraries ${MFEM_LIBRARIES})
if(MFEM_USE_MPI)
  list(APPEND mfem_libraries MPI::MPI_CXX)
endif()

# Headers of this project, one folder per kind.
set(include_dirs
  include/assembly
  include/boundary
  include/material
  include/solver
  include/system
  include/visualization)

# config.yaml is copied into the build directory, again whenever it changes;
# the programs read the copy of the directory they run in.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)
target_compile_definitions(read_mesh PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(driver_static_disp driver_static_disp.cpp)
target_include_directories(driver_static_disp PRIVATE ${include_dirs} ${MFEM_INCLUDE_DIRS})
target_link_libraries(driver_static_disp PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)

add_executable(driver_static_mixed driver_static_mixed.cpp)
target_include_directories(driver_static_mixed PRIVATE ${include_dirs} ${MFEM_INCLUDE_DIRS})
target_link_libraries(driver_static_mixed PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)

add_executable(vtu_writer vtu_writer.cpp)
target_include_directories(vtu_writer PRIVATE ${include_dirs} ${MFEM_INCLUDE_DIRS})
target_link_libraries(vtu_writer PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)

add_executable(csv_writer csv_writer.cpp)
target_include_directories(csv_writer PRIVATE ${include_dirs} ${MFEM_INCLUDE_DIRS})
target_link_libraries(csv_writer PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)

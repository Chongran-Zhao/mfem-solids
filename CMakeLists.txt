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
  ${PROJECT_SOURCE_DIR}/include/assembly
  ${PROJECT_SOURCE_DIR}/include/boundary
  ${PROJECT_SOURCE_DIR}/include/material
  ${PROJECT_SOURCE_DIR}/include/solver
  ${PROJECT_SOURCE_DIR}/include/system
  ${PROJECT_SOURCE_DIR}/include/visualization)

# One folder per formulation, each with its own programs and CMakeLists.txt;
# the programs of a folder are built into the same folder of the build tree.
add_subdirectory(static_disp)
add_subdirectory(static_mixed)

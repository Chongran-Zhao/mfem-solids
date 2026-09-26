# ============================================================================
# Hyperelastostatics
#
# Build:
#   cmake -B build
#   cmake --build build
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(Hyperelastostatics LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Libraries installed in MFEM/lib.
set(lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../lib")

find_package(MFEM REQUIRED HINTS "${lib_dir}/mfem/build")
message(STATUS "MFEM ${MFEM_VERSION} at ${MFEM_DIR}")

if(MFEM_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

find_package(yaml-cpp REQUIRED CONFIG HINTS "${lib_dir}/yaml-cpp/lib/cmake/yaml-cpp")
message(STATUS "yaml-cpp at ${yaml-cpp_DIR}")

find_package(HDF5 REQUIRED CONFIG COMPONENTS C CXX HINTS "${lib_dir}/hdf5/cmake")
message(STATUS "HDF5 ${HDF5_VERSION} at ${HDF5_DIR}")

set(mfem_libraries ${MFEM_LIBRARIES})
if(MFEM_USE_MPI)
  list(APPEND mfem_libraries MPI::MPI_CXX)
endif()

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)
target_compile_definitions(read_mesh PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

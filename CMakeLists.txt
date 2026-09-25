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

find_package(MFEM REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/mfem/build")
message(STATUS "MFEM ${MFEM_VERSION} at ${MFEM_DIR}")

if(MFEM_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

set(mfem_solids_libraries ${MFEM_LIBRARIES})
if(MFEM_USE_MPI)
  list(APPEND mfem_solids_libraries MPI::MPI_CXX)
endif()

add_executable(1d-elastostatics 1d-elastostatics.cpp)
target_include_directories(1d-elastostatics PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(1d-elastostatics PRIVATE ${mfem_solids_libraries})

add_executable(3d-elastostatics 3d-elastostatics.cpp)
target_include_directories(3d-elastostatics PRIVATE include ${MFEM_INCLUDE_DIRS})
target_link_libraries(3d-elastostatics PRIVATE ${mfem_solids_libraries})

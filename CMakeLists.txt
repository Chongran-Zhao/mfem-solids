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

find_package(MFEM REQUIRED HINTS "$ENV{HOME}/.local/mfem/lib/cmake/mfem")
message(STATUS "MFEM ${MFEM_VERSION} at ${MFEM_DIR}")

add_executable(1d-elastostatics 1d-elastostatics.cpp)
target_include_directories(1d-elastostatics PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(1d-elastostatics PRIVATE ${MFEM_LIBRARIES})

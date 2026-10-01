# ============================================================================
# mfem-solids.cmake
#
# Settings shared by the folders of this project; each folder's
# CMakeLists.txt includes it after its own project() call:
#    project_dir     the root of mfem-solids
#    mfem_libraries  MFEM, with MPI when MFEM uses it, and yaml-cpp
#    include_dirs    the header folders under include/
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The root of mfem-solids, the parent of this file's folder.
get_filename_component(project_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Libraries installed in MFEM/lib.
set(lib_dir "${project_dir}/../../lib")

find_package(MFEM REQUIRED HINTS "${lib_dir}/mfem/build")
message(STATUS "MFEM ${MFEM_VERSION} at ${MFEM_DIR}")

if(MFEM_USE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

find_package(yaml-cpp REQUIRED CONFIG HINTS "${lib_dir}/yaml-cpp/lib/cmake/yaml-cpp")
message(STATUS "yaml-cpp at ${yaml-cpp_DIR}")

set(mfem_libraries ${MFEM_LIBRARIES} yaml-cpp::yaml-cpp)
if(MFEM_USE_MPI)
  list(APPEND mfem_libraries MPI::MPI_CXX)
endif()

# Headers of this project, one folder per kind.
set(include_dirs
  ${project_dir}/include/assembly
  ${project_dir}/include/boundary
  ${project_dir}/include/material
  ${project_dir}/include/solver
  ${project_dir}/include/system
  ${project_dir}/include/visualization
  ${MFEM_INCLUDE_DIRS})

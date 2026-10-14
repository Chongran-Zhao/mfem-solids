# ============================================================================
# mfem-solids.cmake
#
# Settings shared by the folders of this project; each folder's
# CMakeLists.txt includes it after its own project() call:
#    project_dir     the root of mfem-solids
#    mfem_libraries  MFEM, built with MPI and MUMPS, MPI, and yaml-cpp
#    mfem_solids     the target of the shared headers of include/, with
#                    their folders and mfem_libraries; a program links it
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The root of mfem-solids, the parent of this file's folder.
get_filename_component(project_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# Libraries installed in MFEM/lib, next to mfem-solids.
set(lib_dir "${project_dir}/../lib")

find_package(MFEM REQUIRED HINTS "${lib_dir}/mfem/build")
message(STATUS "MFEM ${MFEM_VERSION} at ${MFEM_DIR}")

# The programs run on MPI ranks and solve with MUMPS.
if(NOT MFEM_USE_MPI OR NOT MFEM_USE_MUMPS)
  message(FATAL_ERROR "mfem-solids needs MFEM built with MPI and MUMPS.")
endif()
find_package(MPI REQUIRED COMPONENTS CXX)

find_package(yaml-cpp REQUIRED CONFIG HINTS "${lib_dir}/yaml-cpp/lib/cmake/yaml-cpp")
message(STATUS "yaml-cpp at ${yaml-cpp_DIR}")

set(mfem_libraries ${MFEM_LIBRARIES} yaml-cpp::yaml-cpp MPI::MPI_CXX)

# Warn unless the headers listed are those in folder and its subfolders:
# the lists of headers are not needed to build, so nothing else would tell
# that one is out of date.
function(check_header_list folder)
  file(GLOB_RECURSE found RELATIVE ${folder} ${folder}/*.hpp)
  set(listed)
  foreach(header ${ARGN})
    get_filename_component(header ${header} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    file(RELATIVE_PATH header ${folder} ${header})
    list(APPEND listed ${header})
  endforeach()
  set(unlisted ${found})
  set(missing ${listed})
  if(listed)
    list(REMOVE_ITEM unlisted ${listed})
  endif()
  if(found)
    list(REMOVE_ITEM missing ${found})
  endif()
  if(unlisted OR missing)
    message(WARNING "The headers listed are not those of ${folder}:\n"
                    "  not listed: ${unlisted}\n  not found:  ${missing}")
  endif()
endfunction()

# The shared headers of include/, one folder per kind. The project is
# header-only, so mfem_solids compiles nothing: a program that links it
# gets these folders, MFEM, MPI and yaml-cpp, and these headers among its
# sources.
add_library(mfem_solids INTERFACE)
target_include_directories(mfem_solids INTERFACE
  ${project_dir}/include/assembly
  ${project_dir}/include/boundary
  ${project_dir}/include/material
  ${project_dir}/include/solver
  ${project_dir}/include/system
  ${project_dir}/include/visualization
  ${MFEM_INCLUDE_DIRS})
target_link_libraries(mfem_solids INTERFACE ${mfem_libraries})
set(shared_headers
  ${project_dir}/include/assembly/GlobalAssembly_Disp.hpp
  ${project_dir}/include/assembly/GlobalAssembly_Mixed.hpp
  ${project_dir}/include/assembly/IGlobalAssembly.hpp
  ${project_dir}/include/assembly/ILocalAssembly.hpp
  ${project_dir}/include/assembly/LocalAssemblyTools.hpp
  ${project_dir}/include/assembly/LocalAssembly_Disp.hpp
  ${project_dir}/include/assembly/LocalAssembly_Mixed.hpp
  ${project_dir}/include/assembly/ParBlockNonlinearForm_External.hpp
  ${project_dir}/include/boundary/DirichletBoundary.hpp
  ${project_dir}/include/boundary/NeumannBoundary.hpp
  ${project_dir}/include/material/IMaterialModel_ich.hpp
  ${project_dir}/include/material/IMaterialModel_vol.hpp
  ${project_dir}/include/material/MaterialModel.hpp
  ${project_dir}/include/material/MaterialModel_ich_NeoHookean.hpp
  ${project_dir}/include/material/MaterialModel_vol_Incompressible.hpp
  ${project_dir}/include/material/MaterialModel_vol_Quadratic.hpp
  ${project_dir}/include/material/MaterialModel_vol_SimoPister.hpp
  ${project_dir}/include/solver/INonlinearSolver.hpp
  ${project_dir}/include/solver/ITimeSolver.hpp
  ${project_dir}/include/solver/TimeMethod_GenAlpha.hpp
  ${project_dir}/include/system/SystemTools.hpp
  ${project_dir}/include/system/Tensor2_3D.hpp
  ${project_dir}/include/system/Tensor4_3D.hpp
  ${project_dir}/include/system/Vector_3D.hpp
  ${project_dir}/include/visualization/VTK_Tools.hpp)
target_sources(mfem_solids INTERFACE ${shared_headers})
check_header_list(${project_dir}/include ${shared_headers})

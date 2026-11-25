# ============================================================================
# brain_simon: the brain of the SIMon model of NHTSA, the benchmark of
# Giudice et al. (2019), Ann. Biomed. Eng. 47:1855-1872. So far the mesh:
#    read_kfile   config.yaml   Simon.k (LS-Dyna) to simon.mesh (MFEM)
#
# Build and run in this folder:
#    cmake -B build && cmake --build build
#    cd build && ./read_kfile config.yaml
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(brain_simon LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/mfem-solids.cmake)

# The input is copied into the build folder, again whenever it changes.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(read_kfile read_kfile.cpp)
target_include_directories(read_kfile PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_kfile PRIVATE ${mfem_libraries})

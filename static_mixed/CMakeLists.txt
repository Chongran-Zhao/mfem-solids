# ============================================================================
# static_mixed: hyperelastostatics in the mixed displacement-pressure form
#
# Build and run in this folder, with the beam.mesh of read_mesh copied into
# build/:
#    cmake -B build && cmake --build build
#    cd build && ./driver
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(static_mixed LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/mfem-solids.cmake)

# config.yaml is copied into the build folder, again whenever it changes;
# the driver reads the copy of the directory it runs in.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(driver driver.cpp)
target_include_directories(driver PRIVATE ${include_dirs})
target_link_libraries(driver PRIVATE ${mfem_libraries})

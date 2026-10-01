# ============================================================================
# static_mixed: hyperelastostatics in the mixed displacement-pressure form
#
# The driver is built into build/static_mixed/ and runs there.
# ============================================================================

# config.yaml is copied into the build folder, again whenever it changes;
# the driver reads the copy of the directory it runs in.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(driver_static_mixed driver_static_mixed.cpp)
target_include_directories(driver_static_mixed PRIVATE ${include_dirs} ${MFEM_INCLUDE_DIRS})
target_link_libraries(driver_static_mixed PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)

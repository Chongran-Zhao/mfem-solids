# ============================================================================
# static_disp: hyperelastostatics in the displacement form
#
# The programs are built into build/static_disp/ and run there, in the order
# read_mesh, driver_static_disp, vtu_writer, csv_writer.
# ============================================================================

# config.yaml is copied into the build folder, again whenever it changes;
# the programs read the copy of the directory they run in.
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(read_mesh read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${MFEM_INCLUDE_DIRS})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)
# The mesh file is found from the project directory, the HTML template from
# this folder.
target_compile_definitions(read_mesh PRIVATE PROJECT_DIR="${PROJECT_SOURCE_DIR}"
                                             SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(driver_static_disp driver_static_disp.cpp)
target_include_directories(driver_static_disp PRIVATE ${include_dirs} ${MFEM_INCLUDE_DIRS})
target_link_libraries(driver_static_disp PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)

add_executable(vtu_writer vtu_writer.cpp)
target_include_directories(vtu_writer PRIVATE ${include_dirs} ${MFEM_INCLUDE_DIRS})
target_link_libraries(vtu_writer PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)

add_executable(csv_writer csv_writer.cpp)
target_include_directories(csv_writer PRIVATE ${include_dirs} ${MFEM_INCLUDE_DIRS})
target_link_libraries(csv_writer PRIVATE ${mfem_libraries} yaml-cpp::yaml-cpp)

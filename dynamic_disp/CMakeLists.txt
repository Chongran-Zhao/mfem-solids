cmake_minimum_required(VERSION 3.20)
project(dynamic_disp LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/mfem-solids.cmake)
configure_file(config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml COPYONLY)

add_executable(read_mesh ../static_disp/read_mesh.cpp)
target_include_directories(read_mesh PRIVATE ${include_dirs})
target_link_libraries(read_mesh PRIVATE ${mfem_libraries})
target_compile_definitions(read_mesh PRIVATE PROJECT_DIR="${project_dir}"
                                             SOURCE_DIR="${project_dir}/static_disp")

add_executable(driver driver.cpp)
target_include_directories(driver PRIVATE ${include_dirs})
target_link_libraries(driver PRIVATE ${mfem_libraries})

include(CTest)
if(BUILD_TESTING)
  add_executable(test_dynamic_disp ../tests/dynamic_disp.cpp)
  target_include_directories(test_dynamic_disp PRIVATE ${include_dirs})
  target_link_libraries(test_dynamic_disp PRIVATE ${mfem_libraries})
  add_test(NAME dynamic_disp COMMAND test_dynamic_disp)
endif()

add_library(eigenValueSolver eigenValueSolver.cc utils.cc)
add_executable(main main.cc)

target_include_directories(eigenValueSolver
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(eigenValueSolver
  PUBLIC
  El
  MPI::MPI_CXX)

target_link_libraries(main
  PUBLIC
  eigenValueSolver)

if (ENABLE_TESTING)
  add_subdirectory(tests)
endif()

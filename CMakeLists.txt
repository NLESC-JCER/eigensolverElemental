cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(eigenValueSolver)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

###############################################################################
# Compiler flags
###############################################################################
set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

###############################################################################
# User input options
###############################################################################
option(ENABLE_TESTING "Build and enable tests" ON)
if(ENABLE_TESTING)
  enable_testing()
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
endif(ENABLE_TESTING)

###############################################################################
# Dependencies
###############################################################################

#Require MPI for this project
find_package(MPI REQUIRED)

#Add elemental library
find_package(Elemental REQUIRED)

if(ENABLE_TESTING)
  enable_testing()
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
endif(ENABLE_TESTING)

add_subdirectory(src)

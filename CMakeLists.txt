cmake_minimum_required(VERSION 3.20)
project(paralg_cpp)

set(CMAKE_CXX_STANDARD 17)

set(Eigen3_DIR "C:/Program Files/libs/Eigen")
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(myutils STATIC myutils.hpp myutils.cpp)
target_precompile_headers(myutils PUBLIC <vector> <algorithm> <numeric>)

add_executable(paralg_cpp main.cpp)
target_precompile_headers(paralg_cpp PUBLIC [["mpi.h"]] <cstdio> <Eigen/Dense> <numeric> <string> <vector>)
target_link_libraries (paralg_cpp PUBLIC Eigen3::Eigen)
target_link_libraries(paralg_cpp PUBLIC MPI::MPI_CXX)
target_link_libraries(paralg_cpp PRIVATE myutils)
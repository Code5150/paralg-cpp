cmake_minimum_required(VERSION 3.20)
project(paralg_cpp)

set(CMAKE_CXX_STANDARD 17)

set(Eigen3_DIR "C:/Program Files/libs/Eigen")
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(myutils STATIC myutils.hpp myutils.cpp)
target_precompile_headers(myutils PUBLIC <vector> <algorithm> <numeric> <Eigen/Dense>)
target_link_libraries (myutils PUBLIC Eigen3::Eigen)

IF(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
        #SET(MY_C_FLAGS   ${MY_C_FLAGS}   -g2 -Wall)
        SET(MY_CXX_FLAGS ${MY_CXX_FLAGS} -g2 -Wall)
    ELSE()
        #SET(MY_C_FLAGS   ${MY_C_FLAGS}   -O2 -Wno-error)
        SET(MY_CXX_FLAGS ${MY_CXX_FLAGS} -O2 -Wno-error)
    ENDIF()

    FOREACH(CXX_COMPILE_FLAG ${MY_CXX_FLAGS})
        ADD_COMPILE_OPTIONS($<$<COMPILE_LANGUAGE:CXX>:${CXX_COMPILE_FLAG}>)
    ENDFOREACH()
ENDIF()

add_executable(paralg_cpp main.cpp)
target_precompile_headers(paralg_cpp PUBLIC [["mpi.h"]] <cstdio> <Eigen/Dense> <numeric> <string> <vector> <random>
        <iostream> <chrono>)
target_link_libraries (paralg_cpp PUBLIC Eigen3::Eigen)
target_link_libraries(paralg_cpp PUBLIC MPI::MPI_CXX)
target_link_libraries(paralg_cpp PRIVATE myutils)
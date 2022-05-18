cmake_minimum_required(VERSION 2.8)
project(task3)

set(CMAKE_CXX_FLAGS "-std=c++11")
include_directories("/usr/include/eigen3")

add_executable(1 src.cpp)
add_executable(3 Solving_linear_equations.cpp)

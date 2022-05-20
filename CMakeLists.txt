cmake_minimum_required(VERSION 2.8)
project(task3)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_PREFIX_PATH "/root/opencv-3.4.9lib") 
find_package(OpenCV 3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${CERES_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
)

add_executable(problem1_3 problem1_3.cpp)
add_executable(problem4 problem4.cpp)

target_link_libraries(problem1_3 ${OpenCV_LIBS})
target_link_libraries(problem4 ${OpenCV_LIBS} ${CERES_LIBRARIES} Sophus::Sophus)



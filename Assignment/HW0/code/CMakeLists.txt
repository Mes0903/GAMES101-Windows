cmake_minimum_required (VERSION 3.11)
project (Transformation)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/Eigen)

add_executable (Transformation main.cpp)

cmake_minimum_required(VERSION 3.10)
project(Rasterizer3)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/Eigen)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp rasterizer.hpp rasterizer.cpp Triangle.cpp)
add_dependencies(${PROJECT_NAME} opencv)
set(OpenCV_VERSION 4100)

# Add the directory where OpenCV libraries are located
target_link_directories(${PROJECT_NAME} PRIVATE ${OpenCV_LIB_DIR})

# Link against OpenCV libraries using generator expressions
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${OpenCV_LIB_DIR}/opencv_core${OpenCV_VERSION}$<$<CONFIG:Debug>:d>.lib
  ${OpenCV_LIB_DIR}/opencv_imgproc${OpenCV_VERSION}$<$<CONFIG:Debug>:d>.lib
  ${OpenCV_LIB_DIR}/opencv_highgui${OpenCV_VERSION}$<$<CONFIG:Debug>:d>.lib
  ${OpenCV_LIB_DIR}/opencv_imgcodecs${OpenCV_VERSION}$<$<CONFIG:Debug>:d>.lib
)

# Copy OpenCV DLLs after build
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${OpenCV_BIN_DIR}/opencv_core${OpenCV_VERSION}$<$<CONFIG:Debug>:d>.dll
    ${OpenCV_BIN_DIR}/opencv_imgproc${OpenCV_VERSION}$<$<CONFIG:Debug>:d>.dll
    ${OpenCV_BIN_DIR}/opencv_highgui${OpenCV_VERSION}$<$<CONFIG:Debug>:d>.dll
    ${OpenCV_BIN_DIR}/opencv_imgcodecs${OpenCV_VERSION}$<$<CONFIG:Debug>:d>.dll
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/models
)

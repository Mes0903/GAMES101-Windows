cmake_minimum_required(VERSION 3.11)
project(GAMES101)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

if(WIN32)
  if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    MESSAGE("==================== USING MSVC TO COMILE ====================")
    add_compile_options(/wd4819 /wd4244 /wd4267 /wd4305 "/Zc:__cplusplus")
    set(CMAKE_CXX_FLAGS_DEBUG "/O2")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  else()
    MESSAGE("==================== USING MINGW TO COMILE ====================")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  endif()
endif()

# Build OpenCV from source using ExternalProject_Add
ExternalProject_Add(
  opencv
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/OpenCV
  BINARY_DIR ${CMAKE_BINARY_DIR}/opencv-build
  INSTALL_DIR ${CMAKE_BINARY_DIR}/opencv-install
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/opencv-install
      -DBUILD_TESTS=OFF
      -DBUILD_PERF_TESTS=OFF
      -DBUILD_DOCS=OFF
      -DBUILD_EXAMPLES=OFF
      -DBUILD_opencv_python_bindings_generator=OFF
      -DBUILD_opencv_python3=OFF
      -DBUILD_opencv_world=OFF
      -DBUILD_LIST=core,imgproc,highgui,imgcodecs
)

# Define variables for OpenCV include and library directories
set(OpenCV_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/opencv-install/include)
set(OpenCV_LIB_DIR ${CMAKE_BINARY_DIR}/opencv-install/x64/vc17/lib)
set(OpenCV_BIN_DIR ${CMAKE_BINARY_DIR}/opencv-install/x64/vc17/bin)

# Add subdirectories for homework assignments
add_subdirectory(HW0/code)
add_subdirectory(HW1/code)
add_subdirectory(HW2/code)
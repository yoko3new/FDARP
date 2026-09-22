cmake_minimum_required(VERSION 3.10)
project(fdarp_detection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FDARP_DEBUG "Print per-step debug output" OFF)
if(FDARP_DEBUG)
  add_compile_definitions(FDARP_DEBUG)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

# CFITSIO ships no CMake config; search the usual prefixes (Homebrew, /usr/local, /usr).
find_path(CFITSIO_INCLUDE_DIR fitsio.h
  HINTS /opt/homebrew/include /usr/local/include /usr/include
  PATH_SUFFIXES cfitsio)
find_library(CFITSIO_LIBRARY NAMES cfitsio
  HINTS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
if(NOT CFITSIO_INCLUDE_DIR OR NOT CFITSIO_LIBRARY)
  message(FATAL_ERROR "CFITSIO not found. Install it (brew install cfitsio / apt install libcfitsio-dev).")
endif()

add_executable(fdarp_detect main.cpp src/io.cpp src/utils.cpp)

target_include_directories(fdarp_detect PRIVATE
  include ${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIR})

target_link_libraries(fdarp_detect PRIVATE
  ${OpenCV_LIBS} ${HDF5_LIBRARIES} ${CFITSIO_LIBRARY}
  yaml-cpp::yaml-cpp Threads::Threads)

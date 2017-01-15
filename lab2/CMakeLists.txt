cmake_minimum_required (VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)

project(lab2)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

message(STATUS "OpenCL found: ${OPENCL_FOUND}")
message(STATUS "OpenCL includes: ${OPENCL_INCLUDE_DIRS}")
message(STATUS "OpenCL CXX includes: ${OPENCL_HAS_CPP_BINDINGS}")
message(STATUS "OpenCL libraries: ${OPENCL_LIBRARIES}")


include_directories( ${OPENCL_INCLUDE_DIRS} )

add_executable(prefixsum src/main.cpp)
target_include_directories (prefixsum PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries( prefixsum ${OpenCL_LIBRARY} )
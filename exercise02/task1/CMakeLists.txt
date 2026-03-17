cmake_minimum_required(VERSION 3.16)
project(task1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(OpenMP REQUIRED)

add_executable(dgemv dgemv.cpp)
target_link_libraries(dgemv PRIVATE OpenMP::OpenMP_CXX)
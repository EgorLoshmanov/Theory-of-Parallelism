cmake_minimum_required(VERSION 3.16)
project(task3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(OpenMP REQUIRED)

add_executable(simple_iteration simple_iteration.cpp)
target_link_libraries(simple_iteration PRIVATE OpenMP::OpenMP_CXX)
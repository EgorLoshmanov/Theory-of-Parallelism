cmake_minimum_required(VERSION 3.20)
project(task1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(Threads REQUIRED)

add_executable(dgemv dgemv.cpp)
target_link_libraries(dgemv PRIVATE Threads::Threads)

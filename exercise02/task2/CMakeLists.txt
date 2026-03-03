cmake_minimum_required(VERSION 3.16)
project(task2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(integration integration.cpp)
target_link_libraries(integration PRIVATE OpenMP::OpenMP_CXX)
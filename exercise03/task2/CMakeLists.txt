cmake_minimum_required(VERSION 3.14)
project(task2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_executable(server_client main.cpp)
target_link_libraries(server_client PRIVATE Threads::Threads)

add_executable(test_results test.cpp)
target_link_libraries(test_results PRIVATE Threads::Threads)

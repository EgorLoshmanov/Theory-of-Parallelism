cmake_minimum_required(VERSION 3.22)
project(task2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

find_path(TBB_INCLUDE_DIR tbb/tbb.h HINTS /usr/local/include REQUIRED)
find_library(TBB_LIB NAMES tbb HINTS /usr/local/lib REQUIRED)

add_executable(client_server main.cpp)
target_include_directories(client_server PRIVATE ${TBB_INCLUDE_DIR})
target_link_libraries(client_server PRIVATE ${TBB_LIB} Threads::Threads)
target_compile_options(client_server PRIVATE -O2)

add_executable(test_results test.cpp)
target_link_libraries(test_results PRIVATE Threads::Threads)
target_compile_options(test_results PRIVATE -O2)

cmake_minimum_required(VERSION 3.22)
project(task1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

find_path(TBB_INCLUDE_DIR tbb/tbb.h HINTS /usr/local/include REQUIRED)
find_library(TBB_LIB NAMES tbb HINTS /usr/local/lib REQUIRED)

add_executable(dgemv dgemv_tbb.cpp)
target_include_directories(dgemv PRIVATE ${TBB_INCLUDE_DIR})
target_link_libraries(dgemv PRIVATE ${TBB_LIB} Threads::Threads)
target_compile_options(dgemv PRIVATE -O3)

add_executable(dgemv_stdthread dgemv_stdthread.cpp)
target_link_libraries(dgemv_stdthread PRIVATE Threads::Threads)
target_compile_options(dgemv_stdthread PRIVATE -O3)
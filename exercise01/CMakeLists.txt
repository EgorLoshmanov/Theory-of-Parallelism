cmake_minimum_required(VERSION 3.20)
project(exercise01)
set(CMAKE_CXX_STANDARD 20)

# Выбор данных
option(USE_FLOAT "Use float instead double" OFF)

add_executable(exercise01 exercise01.cpp)

if(USE_FLOAT)
    target_compile_definitions(exercise01 PRIVATE USE_FLOAT)
endif()
cmake_minimum_required(VERSION 3.1)

project(
    CNN_from_scratch 
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O2 -g")

add_executable(train train.cpp)
target_compile_definitions(train PUBLIC NDEBUG)

add_executable(unit_tests unit_tests.cpp)

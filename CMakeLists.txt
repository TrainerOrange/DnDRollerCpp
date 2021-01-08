# cmake_minimum_required(VERSION <specify CMake version here>)
cmake_minimum_required(VERSION 3.17)
project(untitled CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(untitled main.cu)

set_target_properties(
        untitled
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
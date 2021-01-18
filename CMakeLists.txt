# cmake_minimum_required(VERSION <specify CMake version here>)
cmake_minimum_required(VERSION 3.17)
project(DnDRoller CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(DnDRoller main.cu)

set_target_properties(
        DnDRoller
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
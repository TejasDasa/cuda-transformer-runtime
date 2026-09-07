cmake_minimum_required(VERSION 3.28.3)

project(cuda_llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)



add_executable(device_info src/device_info.cu)

add_executable(
    inspect_checkpoint 
    src/inspect_checkpoint.cpp
    src/mapped_file.cpp
)

target_include_directories(
    inspect_checkpoint
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
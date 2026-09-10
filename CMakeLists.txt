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
    src/cpu_ops.cpp
)

target_include_directories(
    inspect_checkpoint
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_rmsnorm src/test_rmsnorm.cpp src/cpu_ops.cpp)
target_include_directories(test_rmsnorm PRIVATE include)

add_executable(test_matvec src/test_matvec.cpp src/cpu_ops.cpp)
target_include_directories(test_matvec PRIVATE include)

add_executable(test_rmsnorm_cuda src/test_rmsnorm_cuda.cu)
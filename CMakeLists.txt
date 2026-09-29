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
    src/checkpoint.cpp
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

add_executable(test_rope src/test_rope.cpp src/cpu_ops.cpp)
target_include_directories(test_rope PRIVATE include)

add_executable(test_attention src/test_attention.cpp src/cpu_ops.cpp)
target_include_directories(test_attention PRIVATE include)

add_executable(test_layer_ops src/test_layer_ops.cpp src/cpu_ops.cpp src/layer_forward.cpp)
target_include_directories(test_layer_ops PRIVATE include)

add_executable(
    test_rmsnorm_cuda 
    src/test_rmsnorm_cuda.cu
    src/cpu_ops.cpp
    src/cuda_ops.cu
)

target_include_directories(test_rmsnorm_cuda PRIVATE include)

add_executable(
    test_matvec_cuda 
    src/test_matvec_cuda.cu
    src/cpu_ops.cpp
    src/cuda_ops.cu
)

target_include_directories(test_matvec_cuda PRIVATE include)

add_executable(
    test_rope_cuda
    src/test_rope_cuda.cu
    src/cpu_ops.cpp
    src/cuda_ops.cu
)

target_include_directories(test_rope_cuda PRIVATE include)

add_executable(
    test_attention_cuda
    src/test_attention_cuda.cu
    src/cpu_ops.cpp
    src/cuda_ops.cu
)

target_include_directories(test_attention_cuda PRIVATE include)

add_executable(
    test_layer_ops_cuda
    src/test_layer_ops_cuda.cu
    src/cpu_ops.cpp
    src/cuda_ops.cu
    src/layer_forward.cpp
    src/layer_forward_cuda.cu
)

target_include_directories(test_layer_ops_cuda PRIVATE include)

add_executable(
    test_layer0_cuda
    src/test_layer0_cuda.cu
    src/checkpoint.cpp
    src/mapped_file.cpp
    src/cpu_ops.cpp
    src/cuda_ops.cu
)

target_include_directories(test_layer0_cuda PRIVATE include)

add_executable(
    test_layer0_attention_cuda
    src/test_layer0_attention_cuda.cu
    src/checkpoint.cpp
    src/mapped_file.cpp
    src/cpu_ops.cpp
    src/cuda_ops.cu
)

target_include_directories(test_layer0_attention_cuda PRIVATE include)

add_executable(
    test_layer0_forward_cuda
    src/test_layer0_forward_cuda.cu
    src/checkpoint.cpp
    src/mapped_file.cpp
    src/cpu_ops.cpp
    src/cuda_ops.cu
    src/layer_forward.cpp
    src/layer_forward_cuda.cu
)

target_include_directories(test_layer0_forward_cuda PRIVATE include)

cmake_minimum_required(VERSION 3.17)
project(cuda_memory_optimizations CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(SOURCES main.cu data_transfers/pinning.cu data_transfers/batching.cu data_transfers/staged_copy_execute.cu)

add_executable(cuda_memory_optimizations ${SOURCES} )

set_target_properties(
        cuda_memory_optimizations
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
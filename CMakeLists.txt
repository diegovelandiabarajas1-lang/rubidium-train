cmake_minimum_required(VERSION 3.18)
project(rubidium-train LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture (sm_60 for P100)
set(CMAKE_CUDA_ARCHITECTURES 60)

# Find cuBLAS (manual path on Kaggle)
find_library(CUBLAS_LIBRARY cublas
    HINTS ${CUDNN_DIR}/lib /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
)
find_path(CUBLAS_INCLUDE cublas_v2.h
    HINTS ${CUDNN_DIR}/include /usr/local/cuda/include /usr/include
)

if(NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "cuBLAS not found")
else()
    message(STATUS "Found cuBLAS: ${CUBLAS_LIBRARY}")
endif()

# Find cuDNN
find_path(CUDNN_INCLUDE cudnn.h
    HINTS ${CUDNN_DIR}/include /usr/local/cuda/include /usr/include)
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDNN_DIR}/lib /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

if(NOT CUDNN_INCLUDE OR NOT CUDNN_LIBRARY)
    message(WARNING "cuDNN not found, building without cuDNN support")
    set(CUDNN_FOUND FALSE)
else()
    set(CUDNN_FOUND TRUE)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/model.cu
    src/cuda_kernels.cu
)

# Create executable
add_executable(rubidium-train ${SOURCES})

# Include directories
target_include_directories(rubidium-train PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CUBLAS_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(rubidium-train PRIVATE
    ${CUBLAS_LIBRARY}
    ${CUDNN_LIBRARY}
)

if(CUDNN_FOUND)
    target_include_directories(rubidium-train PRIVATE ${CUDNN_INCLUDE})
    target_compile_definitions(rubidium-train PRIVATE USE_CUDNN=1)
endif()

# Compiler flags
target_compile_options(rubidium-train PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -lineinfo
        -gencode=arch=compute_60,code=sm_60
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
    >
)

# Create checkpoints directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/checkpoints)

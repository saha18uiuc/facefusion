cmake_minimum_required(VERSION 3.18)
project(facefusion_cuda_kernels LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# CUDA kernel library (static)
add_library(ff_cuda_kernels STATIC
    warp_bilinear.cu
    composite.cu
    color_linear.cu
    mask_hysteresis.cu
    kernel_launchers.cu
)

# Set CUDA architecture (adjust based on your GPU)
# Common values: 75 (Turing), 80/86 (Ampere), 89 (Ada Lovelace)
set_target_properties(ff_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89"
)

# CUDA compile options for deterministic operations + performance
target_compile_options(ff_cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --fmad=false          # Disable FMA contraction for determinism
        -Xptxas -O3           # Optimize PTX
        --use_fast_math       # Fast math for non-critical paths
        --expt-relaxed-constexpr
        -maxrregcount=64      # Increase occupancy by limiting register usage
        --extra-device-vectorization  # Enable device vectorization
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3                   # Maximum C++ optimization
        -march=native         # Target native CPU architecture
        -ffast-math           # Fast math for host code
        -funroll-loops        # Unroll loops
        -finline-functions    # Aggressive inlining
    >
)

# Enable interprocedural optimization (LTO) for release builds
set_target_properties(ff_cuda_kernels PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# Include directories
target_include_directories(ff_cuda_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries
target_link_libraries(ff_cuda_kernels PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

# Python bindings module
pybind11_add_module(ff_cuda_ops
    bindings.cpp
)

target_link_libraries(ff_cuda_ops PRIVATE
    ff_cuda_kernels
    CUDA::cudart
)

# Optimize Python bindings with C++17, LTO, and aggressive optimization
target_compile_options(ff_cuda_ops PRIVATE
    -O3
    -march=native
    -ffast-math
    -funroll-loops
    -finline-functions
)

# Set output directory and enable LTO
set_target_properties(ff_cuda_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    INTERPROCEDURAL_OPTIMIZATION TRUE
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Install targets
install(TARGETS ff_cuda_ops
    LIBRARY DESTINATION ${Python3_SITELIB}/facefusion/cuda
)

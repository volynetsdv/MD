cmake_minimum_required(VERSION 3.20)

# --- Locate CUDA toolkit before project() triggers compiler-ID test ------
if(NOT DEFINED ENV{CUDA_HOME})
    if(EXISTS "/usr/local/cuda")
        set(ENV{CUDA_HOME} "/usr/local/cuda")
    elseif(EXISTS "/usr/lib64/libcudart.so")
        # Flat system install (dnf/yum) — point to /usr
        set(ENV{CUDA_HOME} "/usr")
    endif()
endif()

project(tiling_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_STANDARD 17)

# --- Core library (CPU math) ---
add_library(tiling_math STATIC src/tiling_math.cpp)
target_include_directories(tiling_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- CUDA slicer module ---
add_library(cuda_slicer STATIC src/cuda_slicer.cu)
target_include_directories(cuda_slicer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(cuda_slicer PROPERTIES CUDA_STANDARD 17)

# --- Tests ---
enable_testing()

add_executable(test_tiling_math tests/test_tiling_math.cpp)
target_link_libraries(test_tiling_math PRIVATE tiling_math)
add_test(NAME test_tiling_math COMMAND test_tiling_math)

add_executable(test_cuda_slicer tests/test_cuda_slicer.cpp)
target_link_libraries(test_cuda_slicer PRIVATE cuda_slicer)
add_test(NAME test_cuda_slicer COMMAND test_cuda_slicer)

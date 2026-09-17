cmake_minimum_required(VERSION 3.20)

# --- Ensure Shared CUDA runtime linking to prevent static lib cudart issues ---
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

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
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Find CUDAToolkit (Creates the CUDA::cudart target & sets CUDA includes) ---
find_package(CUDAToolkit REQUIRED)

# --- pybind11 & Python (for pytiling_core module) ---
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED
    PATHS "${Python3_SITELIB}/pybind11/share/cmake/pybind11"
          "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11"
)


# --- Core library (CPU math) ---
add_library(tiling_math STATIC src/tiling_math.cpp)
target_include_directories(tiling_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- CUDA slicer module ---
add_library(cuda_slicer STATIC src/cuda_slicer.cu)
target_include_directories(cuda_slicer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(cuda_slicer PROPERTIES CUDA_STANDARD 17)

# --- TensorRT detector module ---
set(_trt_hints
    $ENV{TENSORRT_HOME}/lib
    $ENV{CONDA_PREFIX}/lib
    /usr/local/TensorRT-*/lib
    /opt/tensorrt/lib
    /usr/lib/x86_64-linux-gnu
)

find_path(TRT_INCLUDE_DIR NvInferRuntime.h HINTS ${_trt_hints} PATH_SUFFIXES include)
find_library(TRT_LIB NAMES nvinfer HINTS ${_trt_hints})
find_library(TRT_PLUGIN_LIB NAMES nvinfer_plugin HINTS ${_trt_hints})

if(NOT TRT_INCLUDE_DIR OR NOT TRT_LIB OR NOT TRT_PLUGIN_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_HOME or install TensorRT.")
endif()

add_library(trt_detector STATIC src/trt_detector.cpp)

# Обов'язково додаємо шляхи include для C++ файлу: заголовочні файли проєкту, TensorRT та CUDA
target_include_directories(trt_detector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(trt_detector PRIVATE ${TRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

# Лінкуємо TensorRT та офіційну ціль CUDA Runtime
target_link_libraries(trt_detector PUBLIC ${TRT_LIB} ${TRT_PLUGIN_LIB})
target_link_libraries(trt_detector PRIVATE CUDA::cudart)

# --- Postprocessing module (Offset Mapping & Cluster-DIoU-NMS) ---
add_library(postprocess STATIC src/postprocess.cpp)
target_include_directories(postprocess PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(postprocess PUBLIC tiling_math)

# --- Tests ---
enable_testing()

add_executable(test_tiling_math tests/test_tiling_math.cpp)
target_link_libraries(test_tiling_math PRIVATE tiling_math)
add_test(NAME test_tiling_math COMMAND test_tiling_math)

# --- Налаштування для test_cuda_slicer ---
add_executable(test_cuda_slicer tests/test_cuda_slicer.cpp)

# Додаємо шляхи до CUDA include для C++ файла тесту:
target_include_directories(test_cuda_slicer PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Лінкуємо бібліотеку slicer та CUDA runtime:
target_link_libraries(test_cuda_slicer PRIVATE 
    cuda_slicer 
    CUDA::cudart
)
add_test(NAME test_cuda_slicer COMMAND test_cuda_slicer)

# --- Налаштування для test_trt_detector ---
add_executable(test_trt_detector tests/test_trt_detector.cpp)

# 1. Додаємо шляхи до include-директорій CUDA та TensorRT:
target_include_directories(test_trt_detector PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)

# 2. Лінкуємо trt_detector та runtime бібліотеку CUDA:
target_link_libraries(test_trt_detector PRIVATE 
    trt_detector
    CUDA::cudart
)

add_test(NAME test_trt_detector COMMAND test_trt_detector)

# --- Налаштування для test_postprocess ---
add_executable(test_postprocess tests/test_postprocess.cpp)
target_include_directories(test_postprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_postprocess PRIVATE postprocess)
add_test(NAME test_postprocess COMMAND test_postprocess)

# --- Головний Pipeline (Inference Engine Orchestrator) ---
add_executable(main_pipeline src/main_pipeline.cpp)
target_include_directories(main_pipeline PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)
target_link_libraries(main_pipeline PRIVATE 
    tiling_math
    cuda_slicer
    trt_detector
    postprocess
    CUDA::cudart
)
add_test(NAME test_main_pipeline COMMAND main_pipeline --width 7680 --height 4320 --altitude 100 --max-tiles 12)

# --- Unit-тест для main_pipeline (test_main_pipeline) ---
add_executable(test_main_pipeline tests/test_main_pipeline.cpp)
target_include_directories(test_main_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)
target_link_libraries(test_main_pipeline PRIVATE
    tiling_math
    cuda_slicer
    trt_detector
    postprocess
    CUDA::cudart
)
add_test(NAME unit_test_main_pipeline COMMAND test_main_pipeline)

# =============================================================================
# pytiling_core — pybind11 Python extension module
# =============================================================================
# Builds a .so that Python can import directly:
#   import pytiling_core
# The module re-uses the same libtiling_math + libpostprocess implementations
# (DRY principle — no duplicate logic).
#
# NOTE: trt_detector is NOT linked here on purpose: the Python binding only
# exposes CPU tiling math and postprocessing. TensorRT inference stays in C++.
# =============================================================================
pybind11_add_module(pytiling_core bindings/bindings.cpp)

target_include_directories(pytiling_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(pytiling_core PRIVATE
    tiling_math
    postprocess
)

# Place the .so next to the build directory root so tests can import it without
# any PYTHONPATH surgery.
set_target_properties(pytiling_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# --- Python Tests (pytiling_core, dataset_slicer, export_profiles) ---
add_test(NAME test_binding COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_binding.py)
add_test(NAME test_dataset_slicer COMMAND ${Python3_EXECUTABLE} -m pytest -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dataset_slicer.py)
add_test(NAME test_export_profiles COMMAND ${Python3_EXECUTABLE} -m pytest -v ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_export_profiles.py)
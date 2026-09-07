cmake_minimum_required(VERSION 3.20)
project(tiling_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Core library ---
add_library(tiling_math STATIC src/tiling_math.cpp)
target_include_directories(tiling_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Tests ---
enable_testing()

add_executable(test_tiling_math tests/test_tiling_math.cpp)
target_link_libraries(test_tiling_math PRIVATE tiling_math)
add_test(NAME test_tiling_math COMMAND test_tiling_math)

cmake_minimum_required(VERSION 3.18)
project(kmamba_cuda LANGUAGES C ASM_NASM CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# NASM configuration
set(CMAKE_ASM_NASM_COMPILE_OBJECT "<CMAKE_ASM_NASM_COMPILER> <INCLUDES> <FLAGS> -o <OBJECT> <SOURCE>")
set(CMAKE_ASM_NASM_FLAGS "-f elf64 -O3")

# Build k-mamba avec backend CUDA (MX450 = sm_75)
set(KMAMBA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(KMAMBA_BUILD_BENCH    OFF CACHE BOOL "" FORCE)
set(KMAMBA_BUILD_CPU      ON  CACHE BOOL "" FORCE)
set(KMAMBA_BUILD_CUDA     ON  CACHE BOOL "" FORCE)
# Force OPTIMATRIX_BUILD_CUDA avant que optimatrix/CMakeLists.txt le lise
set(OPTIMATRIX_BUILD_CUDA ON  CACHE BOOL "" FORCE)
set(OPTIMATRIX_BUILD_CPU  ON  CACHE BOOL "" FORCE)

# CUDA arch pour MX450 (Turing, sm_75)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../k-mamba
    ${CMAKE_CURRENT_BINARY_DIR}/k-mamba
)

add_executable(kmamba_cuda main.c)

target_link_libraries(kmamba_cuda PRIVATE k-mamba::k-mamba m)

target_compile_options(kmamba_cuda PRIVATE
    -O3 -mavx2
    -Wall -Wextra
    -Wno-unused-parameter
)

# No PIE — requis par les ASM k-mamba
target_link_options(kmamba_cuda PRIVATE -no-pie)

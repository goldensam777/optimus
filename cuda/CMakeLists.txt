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

add_executable(kmamba_cuda main.cu)

target_link_libraries(kmamba_cuda PRIVATE k-mamba::k-mamba m cublas)

target_compile_options(kmamba_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -arch=sm_75>
    $<$<COMPILE_LANGUAGE:C>:-O3 -mavx2 -Wall -Wextra -Wno-unused-parameter>
)

# No PIE — requis par les ASM k-mamba
target_link_options(kmamba_cuda PRIVATE -no-pie)

# ─── Run targets ──────────────────────────────────────────────────────────────
set(DATA_FILE  ""                 CACHE STRING "Fichier texte d'entrainement")
set(CKPT_FILE  "kmamba_cuda.bin"  CACHE STRING "Checkpoint (lecture/ecriture)")
set(PROMPT     "Les systemes"     CACHE STRING "Prompt pour la generation")

# make -C build train
add_custom_target(train
    COMMAND $<TARGET_FILE:kmamba_cuda>
    DEPENDS kmamba_cuda
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "CUDA : entrainement sur texte integre"
)

# make -C build train-file
add_custom_target(train-file
    COMMAND $<TARGET_FILE:kmamba_cuda> train "${DATA_FILE}" "${CKPT_FILE}"
    DEPENDS kmamba_cuda
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "CUDA : entrainement sur ${DATA_FILE} -> ${CKPT_FILE}"
)

# make -C build gen
add_custom_target(gen
    COMMAND $<TARGET_FILE:kmamba_cuda> gen "${CKPT_FILE}" "${PROMPT}"
    DEPENDS kmamba_cuda
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "CUDA : generation depuis ${CKPT_FILE}"
)

# make -C build chat
add_custom_target(chat
    COMMAND $<TARGET_FILE:kmamba_cuda> chat "${CKPT_FILE}"
    DEPENDS kmamba_cuda
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "CUDA : chat REPL depuis ${CKPT_FILE}"
)

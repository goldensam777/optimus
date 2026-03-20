cmake_minimum_required(VERSION 3.18)
project(kmamba_cpu LANGUAGES C ASM_NASM)

find_package(OpenMP REQUIRED)

set(CMAKE_C_STANDARD 11)

# NASM configuration (mirrors k-mamba's setup)
set(CMAKE_ASM_NASM_COMPILE_OBJECT "<CMAKE_ASM_NASM_COMPILER> <INCLUDES> <FLAGS> -o <OBJECT> <SOURCE>")
set(CMAKE_ASM_NASM_FLAGS "-f elf64 -O3")

# Build k-mamba as subdirectory — CPU only
set(KMAMBA_BUILD_TESTS   ON  CACHE BOOL "" FORCE)
set(KMAMBA_BUILD_BENCH   OFF CACHE BOOL "" FORCE)
set(KMAMBA_BUILD_CUDA    OFF CACHE BOOL "" FORCE)
set(KMAMBA_BUILD_CPU     ON  CACHE BOOL "" FORCE)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../k-mamba
    ${CMAKE_CURRENT_BINARY_DIR}/k-mamba
)

add_executable(kmamba_cpu main.c)
add_executable(test_grad test_grad.c)
add_executable(test_scan test_scan.c)

target_link_libraries(kmamba_cpu PRIVATE k-mamba::k-mamba m OpenMP::OpenMP_C)
target_link_libraries(test_grad  PRIVATE k-mamba::k-mamba m)
target_link_libraries(test_scan  PRIVATE k-mamba::k-mamba m)

target_compile_options(kmamba_cpu PRIVATE
    -O3 -mavx2 -fopenmp
    -Wall -Wextra
    -Wno-unused-parameter
)

target_compile_options(test_grad PRIVATE
    -O1 -mavx2   # O1 pour garder les variables accessibles en debug
    -Wall -Wextra -Wno-unused-parameter
)

target_link_options(test_grad PRIVATE -no-pie)
target_compile_options(test_scan PRIVATE -O1 -mavx2 -Wall -Wextra -Wno-unused-parameter)
target_link_options(test_scan PRIVATE -no-pie)

# No PIE — required by k-mamba ASM (32-bit reloc incompatible with PIE)
target_link_options(kmamba_cpu PRIVATE -no-pie)

# ─── Run targets ──────────────────────────────────────────────────────────────
# Variables configurables (passer avec -D au cmake ou via make ARGS)
set(DATA_FILE  ""                CACHE STRING "Fichier texte d'entrainement")
set(CKPT_FILE  "kmamba_cpu.bin"  CACHE STRING "Checkpoint (lecture/ecriture)")
set(PROMPT     "Les systemes"    CACHE STRING "Prompt pour la generation")

# make -C build train              → entraine sur texte integre
add_custom_target(train
    COMMAND $<TARGET_FILE:kmamba_cpu>
    DEPENDS kmamba_cpu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "CPU : entrainement sur texte integre"
)

# make -C build train-file         → entraine sur DATA_FILE
add_custom_target(train-file
    COMMAND $<TARGET_FILE:kmamba_cpu> train "${DATA_FILE}" "${CKPT_FILE}"
    DEPENDS kmamba_cpu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "CPU : entrainement sur ${DATA_FILE} -> ${CKPT_FILE}"
)

# make -C build gen                → genere depuis CKPT_FILE
add_custom_target(gen
    COMMAND $<TARGET_FILE:kmamba_cpu> gen "${CKPT_FILE}" "${PROMPT}"
    DEPENDS kmamba_cpu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "CPU : generation depuis ${CKPT_FILE}"
)

# make -C build chat               → REPL interactif
add_custom_target(chat
    COMMAND $<TARGET_FILE:kmamba_cpu> chat "${CKPT_FILE}"
    DEPENDS kmamba_cpu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "CPU : chat REPL depuis ${CKPT_FILE}"
)

cmake_minimum_required(VERSION 3.20)
project(Leibniz)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(ASM_MASM)

file(GLOB_RECURSE LEIBNIZ_PUBLIC    "${CMAKE_CURRENT_SOURCE_DIR}/src/Public/*.h")
file(GLOB_RECURSE LEIBNIZ_PRIVATE   "${CMAKE_CURRENT_SOURCE_DIR}/src/Private/*.cpp")
file(GLOB_RECURSE LEIBNIZ_ASM       "${CMAKE_CURRENT_SOURCE_DIR}/src/Assembly/*.asm")
file(GLOB_RECURSE LEIBNIZ_BACKEND       "${CMAKE_CURRENT_SOURCE_DIR}/src/Backend/*.h")
file(GLOB_RECURSE LEIBNIZ_INTRIN       "${CMAKE_CURRENT_SOURCE_DIR}/src/Intrin/*.h")
file(GLOB_RECURSE LEIBNIZ_COMPILER       "${CMAKE_CURRENT_SOURCE_DIR}/src/Compiler/*.h")
file(GLOB_RECURSE LEIBNIZ_NUMERICS       "${CMAKE_CURRENT_SOURCE_DIR}/src/Numerics/*.h")
file(GLOB_RECURSE LEIBNIZ_KERNELS      "${CMAKE_CURRENT_SOURCE_DIR}/src/Kernels/*.h")
    
add_library(Leibniz SHARED
    ${LEIBNIZ_PRIVATE}
    ${LEIBNIZ_PUBLIC}
    ${LEIBNIZ_BACKEND}
    ${LEIBNIZ_INTRIN}
    ${LEIBNIZ_COMPILER}
    ${LEIBNIZ_NUMERICS}
    ${LEIBNIZ_KERNELS}
)

set_source_files_properties(${LEIBNIZ_ASM} PROPERTIES LANGUAGE ASM_MASM)
target_sources(Leibniz PRIVATE ${LEIBNIZ_ASM})

target_include_directories(Leibniz
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Backend
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Intrin
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Compiler
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Numerics
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Kernels
)

target_precompile_headers(Leibniz PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/Public/Leibniz.h)

set_target_properties(Leibniz PROPERTIES FOLDER "Base")

target_compile_options(Leibniz PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/arch:AVX2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-mavx2>
)

target_compile_definitions(Leibniz PRIVATE        
        LEIBNIZ_SHARED
        LEIBNIZ_BUILDING_RUNTIME)   
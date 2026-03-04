cmake_minimum_required(VERSION 3.20)
project(Leibniz)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE LEIBNIZ_PUBLIC    "src/Public/*.h")
file(GLOB_RECURSE LEIBNIZ_PRIVATE   "src/Private/*.cpp")

add_library(Leibniz SHARED
    ${LEIBNIZ_PRIVATE} ${LEIBNIZ_PUBLIC}
)

target_include_directories(Leibniz PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
)

target_precompile_headers(Leibniz PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/Public/Leibniz.h)

set_target_properties(Leibniz PROPERTIES FOLDER "Base")

target_compile_options(Leibniz PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2>
)

target_compile_definitions(Leibniz PRIVATE        
        LEIBNIZ_SHARED
        LEIBNIZ_BUILDING_RUNTIME)
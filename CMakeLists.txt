cmake_minimum_required(VERSION 3.20)
project(Lebnitz)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE LEBNITZ_PUBLIC    "src/Public/*.h")
file(GLOB_RECURSE LEBNITZ_PRIVATE   "src/Private/*.cpp")

add_library(Lebnitz SHARED
    ${LEBNITZ_PRIVATE} ${LEBNITZ_PUBLIC}
  )

target_include_directories(Lebnitz PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
)

target_precompile_headers(Lebnitz PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/Public/Lebnitz.h)

set_target_properties(Lebnitz PROPERTIES FOLDER "Base")


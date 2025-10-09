cmake_minimum_required(VERSION 3.20)
project(Lebnitz)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(Lebnitz SHARED
    src/Private/Lebnitz.cpp
    src/Public/Lebnitz.h
  )

target_include_directories(Lebnitz PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
)

set_target_properties(Lebnitz PROPERTIES FOLDER "Base")


cmake_minimum_required(VERSION 3.25)
project(NBP_jupyter)

set(CMAKE_CXX_STANDARD 14)
FIND_PACKAGE( OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(.)

add_executable(NBP_jupyter
        simulateFER.cpp
        stabilizerCodes.cpp
        stabilizierCodes.h)

cmake_minimum_required(VERSION 3.25)
project(NBP_jupyter CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

link_libraries(stdc++fs)

link_libraries(stdc++fs)

add_executable(NBP_jupyter
        simulateFER.cpp
        stabilizerCodes.cpp
        fileReader.cpp
        helpers.cpp
				ensembleDecoder.cpp
        )

find_package(OpenMP)
if(OPENMP_FOUND)
    target_link_libraries(NBP_jupyter PUBLIC OpenMP::OpenMP_CXX)
endif()


if(MSVC)
    target_compile_options(NBP_jupyter PRIVATE /W4 /WX)
else()
    target_compile_options(NBP_jupyter PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-ffast-math" fastmath)
if(fastmath)
    target_compile_options(NBP_jupyter PRIVATE -ffast-math)
endif()

check_cxx_compiler_flag("-funsafe-math-optimizations" veryfastmath)
if(veryfastmath)
    target_compile_options(NBP_jupyter PRIVATE -funsafe-math-optimizations)
endif()

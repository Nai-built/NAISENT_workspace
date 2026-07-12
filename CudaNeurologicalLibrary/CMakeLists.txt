cmake_minimum_required(VERSION 3.5...3.18)
project(PybindTesting LANGUAGES CXX CUDA)

find_package (Python3 COMPONENTS Interpreter Development REQUIRED) # the Development component is required, else we get an error: https://github.com/pybind/pybind11/issues/3996

find_package(CUDAToolkit REQUIRED)

set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if (MSVC)
#     add_compile_options($<$<CONFIG:Release>:/O2 /arch:AVX2 /fp:fast /DNDEBUG /GL>)
#     add_link_options($<$<CONFIG:Release>:/LTCG>)
# endif()
# add_compile_options(-O3 -march=native -DNDEBUG -ffast-math)

# if (MSVC)
#     add_compile_options(/O2 /DNDEBUG /fp:fast)
# else()
#     add_compile_options(-O3 -march=native -DNDEBUG -ffast-math)
# endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})

message(STATUS "Using pybind11 found at: ${pybind11_DIR}")
find_package(pybind11 REQUIRED)

# Create Cuda-kernels library
file(GLOB_RECURSE CU_SOURCES
    "library/*.cu"
)
add_library(cuda_kernels SHARED ${CU_SOURCES})
set_target_properties(cuda_kernels PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 50 CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(cuda_kernels PRIVATE CUDA_LIB_EXPORTS)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
    >
)

# Create Python module
file(GLOB_RECURSE CPP_SOURCES
    "library/*.cpp"
)
pybind11_add_module(cuda_module SHARED ${CPP_SOURCES} pybind_wrapper.cpp)

# REMEMBER TO UPDATE THE C++ VERSION WHEN SETTING THE TARGET PROPERTIES: now it'll be from 11 -> 23
set_target_properties(cuda_module PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

target_link_libraries(cuda_module PRIVATE cuda_kernels CUDA::cudart)
add_custom_command(TARGET cuda_module POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:cuda_kernels>
    $<TARGET_FILE_DIR:cuda_module>
)
add_custom_command(TARGET cuda_module POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:CUDA::cudart>
    $<TARGET_FILE_DIR:cuda_module>
)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/MD")

target_compile_options(cuda_module PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/fp:fast>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/arch:AVX2>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/GL>
)

target_link_options(cuda_module PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/LTCG>
)

# Needed for importing the "cuda_module" module from the build directory
set(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
if(NOT EXISTS ${INIT_PY})
    file(WRITE ${INIT_PY} "")
endif()

add_custom_command(TARGET cuda_module POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:cuda_module>/$<TARGET_FILE_NAME:cuda_module>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:cuda_module>
)

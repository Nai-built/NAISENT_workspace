cmake_minimum_required(VERSION 3.5...3.18)
project(PybindTesting)

find_package (Python3 COMPONENTS Interpreter Development REQUIRED) # the Development component is required, else we get an error: https://github.com/pybind/pybind11/issues/3996

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

# Create Python module
file(GLOB_RECURSE CPP_SOURCES
    "library/*.cpp"
)
pybind11_add_module(simple_module SHARED ${CPP_SOURCES} pybind_wrapper.cpp)

# REMEMBER TO UPDATE THE C++ VERSION WHEN SETTING THE TARGET PROPERTIES: now it'll be from 11 -> 23
set_target_properties(simple_module PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

target_compile_options(simple_module PRIVATE
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:Release>:/fp:fast>
    $<$<CONFIG:Release>:/arch:AVX2>
    $<$<CONFIG:Release>:/GL>
)

target_link_options(simple_module PRIVATE
    $<$<CONFIG:Release>:/LTCG>
)

# Needed for importing the "simple_module" module from the build directory
set(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
if(NOT EXISTS ${INIT_PY})
    file(WRITE ${INIT_PY} "")
endif()

add_custom_command(TARGET simple_module POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE_DIR:simple_module>/$<TARGET_FILE_NAME:simple_module>
            ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:simple_module>
)

cmake_minimum_required(VERSION 3.18)
project(simple-filters
        DESCRIPTION "Simple implementation of image filters on CUDA"
        LANGUAGES CXX CUDA)

# Handle CUDA to g++ mapping
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12")
        set(MAX_GCC "11")
        find_program(GCC_EXE NAMES gcc-${MAX_GCC} PATHS /usr/bin /usr/local/bin)
        find_program(GXX_EXE NAMES g++-${MAX_GCC} PATHS /usr/bin /usr/local/bin)

        if(GCC_EXE AND GXX_EXE)
                set(CMAKE_C_COMPILER ${GCC_EXE} CACHE FILEPATH "C compiler" FORCE)
                set(CMAKE_CXX_COMPILER ${GXX_EXE} CACHE FILEPATH "C++ compiler" FORCE)
                message(STATUS "Using GCC: ${CMAKE_C_COMPILER}, G++: ${CMAKE_CXX_COMPILER}")
        else()
                message(WARNING "Could not find gcc-${MAX_GCC}/g++-${MAX_GCC}, using system defaults")
        endif()
endif()

# Sources and executables
file(GLOB_RECURSE sources include/*.h src/*.cpp src/*.cu)

# CUDA flags and standards
execute_process(
        COMMAND bash -c "echo -n $(which nvcc)"
        OUTPUT_VARIABLE nvccLocation
)

set(CMAKE_CUDA_COMPILER "${nvccLocation}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# CPP-related flags
set(CMAKE_CXX_STANDARD 23)

add_executable(simple-filters ${sources})

# Attach include/ directory to the target
target_include_directories(simple-filters PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(simple-filters PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# Package managing (empty for now, must be done via conan or vcpkg)
# find_package()


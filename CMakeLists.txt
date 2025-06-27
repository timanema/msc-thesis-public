cmake_minimum_required(VERSION 3.25.2)

project(gtsst LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

find_package(Threads REQUIRED)
add_executable(gtsst
        src/main.cu
        src/bench/gtsst-bench.cu
        src/fsst/fsst-lib.cu
        src/fsst/gtsst-fsst.cu
        src/compressors/plain/plain-compressor.cu
        src/compressors/stats/stats-compressor.cu
        src/compressors/shared.cu
        src/bench/gtsst-prepare.cu
        src/compressors/compactionv2/compaction-compressor.cu
        src/compressors/compactionv2/compaction-encode.cu
        src/compressors/slidingv1/sliding-compressor.cu
        src/compressors/compactionv2/compaction-decompress.cu
        src/compressors/slidingv1/sliding-decompress.cu
        src/compressors/slidingv1/sliding-encode.cu
        src/compressors/compactionv3t/compaction-compressor.cu
        src/compressors/compactionv3t/compaction-encode.cu
        src/compressors/compactionv3t/compaction-decompress.cu
        src/compressors/compactionv4t/compaction-compressor.cu
        src/compressors/compactionv4t/compaction-encode.cu
        src/compressors/compactionv4t/compaction-decompress.cu
        src/compressors/compactionv5t/compaction-compressor.cu
        src/compressors/compactionv5t/compaction-encode.cu
        src/compressors/compactionv5t/compaction-decompress.cu
        src/compressors/compactionv3/compaction-compressor.cu
        src/compressors/compactionv3/compaction-encode.cu
        src/compressors/compactionv3/compaction-decompress.cu
        src/compressors/compactionv4/compaction-compressor.cu
        src/compressors/compactionv4/compaction-encode.cu
        src/compressors/compactionv4/compaction-decompress.cu
        src/compressors/compactionv1/compaction-compressor.cu
        src/compressors/compactionv1/compaction-encode.cu
        src/compressors/compactionv1/compaction-decompress.cu)
target_link_libraries(gtsst LINK_PUBLIC Threads::Threads)
target_include_directories(gtsst
        PUBLIC ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(gtsst PROPERTIES OUTPUT_NAME gtsst)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(gtsst PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --default-stream per-thread
            >)
else ()
    target_compile_options(gtsst PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -G
            -g
            --default-stream per-thread
            >)
endif ()

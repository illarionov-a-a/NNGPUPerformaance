cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 17)

find_library(TENSORFLOW_LIB tensorflow HINT ${TF_LIB})

if ( "${USE_GPU}" STREQUAL "gpu")
    set(EXE "nn_gpu_performance_test")
else()
    set(EXE "nn_cpu_performance_test")
endif()
 
include_directories(${BOOST_INCLUDE})
include_directories(${TF_INCLUDE})
add_executable(${EXE} main.cpp )
target_link_libraries (${EXE} ${TENSORFLOW_LIB})


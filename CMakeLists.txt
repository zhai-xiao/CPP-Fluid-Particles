cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)

if(NOT "${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
    message("Please use 64-bit cmake generator.")
endif()

project(CPP_Fluid_Particles VERSION 0.1.0 LANGUAGES CXX C CUDA)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCH 5 6)
set(CMAKE_CONFIGURATION_TYPES "Release")

set(CUDASAMPLESROOT $ENV{NVCUDASAMPLES_ROOT})
message("cuda samples root ${CUDASAMPLESROOT}")
include_directories(${CUDASAMPLESROOT}/common/inc)
link_directories(${CUDASAMPLESROOT}/common/lib/x64/)

add_subdirectory(src)

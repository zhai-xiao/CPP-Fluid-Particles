cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)

if(NOT "${CMAKE_GENERATOR}" MATCHES "(Win64|IA64)")
    message([<FATAL_ERROR>] "Please use 64-bit cmake generator.")
	return()
endif()

project(CPP_Fluid_Particles VERSION 0.1.0 LANGUAGES CXX C CUDA)

include_directories($ENV{CUDA_PATH}/include/)
include_directories($ENV{NVCUDASAMPLES_ROOT}/common/inc/)
link_directories($ENV{NVCUDASAMPLES_ROOT}/common/lib/x64/)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Ot /Oi /MP")
	SET(CMAKE_EXE_LINKER_FLAGS "/OPT:REF /OPT:ICF")
endif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")

find_package(CUDA REQUIRED)

set(CMAKE_CONFIGURATION_TYPES "Release")

set(CMAKE_CUDA_ARCH 5 6)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -use_fast_math -Xcompiler \"/wd 4819 /wd 4267\" ")

add_subdirectory(src)

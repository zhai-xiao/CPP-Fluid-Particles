cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(FluidParticles VERSION 0.1.0 LANGUAGES CXX C CUDA)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS 
	"-gencode arch=compute_20,code=sm_20;"
	"-gencode arch=compute_20,code=sm_21;"
	"-gencode arch=compute_30,code=sm_30;"
	"-gencode arch=compute_35,code=sm_35;"
	"-gencode arch=compute_50,code=sm_50;"
	"-gencode arch=compute_52,code=sm_52; "
	"-gencode arch=compute_60,code=sm_60;"
	"-gencode arch=compute_61,code=sm_61;"
	"-gencode;arch=compute_61;code=sm_61;" 
	"-gencode;arch=compute_75;code=sm_75;" # GTX 2080 Ti
	)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

include_directories(${PROJECT_SOURCE_DIR}/external/)
include_directories(${PROJECT_SOURCE_DIR}/external/include)
link_directories(${PROJECT_SOURCE_DIR}/external/lib/)
add_subdirectory(src)
set(shaderfile particles.frag particles.vert)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${shaderfile}
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
	
set(headers
	BaseSolver.h
	BasicSPHSolver.h
	DArray.h
	DFSPHSolver.h
	global.h
	Particles.h
	PBDSolver.h
	ShaderUtility.h
	SPHParticles.h
	SPHSystem.h
	CUDAFunctions.cuh
	)
	
set(source
	BasicSPHSolver.cu
	DFSPHSolver.cu
	Particles.cu
	PBDSolver.cu
	SPHSystem.cu
	vbo.cu
	ShaderUtility.cpp
	main.cpp
	)

add_executable(CPP_Fluid_Particles
	${headers}
	${source}
	)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	# Optimization options for C/C++
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Ot /Oi /Gy /MP")
	# Optimization options for C/C++ Linkers
	set_property(TARGET CPP_Fluid_Particles APPEND PROPERTY LINK_FLAGS "/OPT:REF /OPT:ICF")
endif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")

# Optimization options for CUDA C/C++ 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -use_fast_math -Xcompiler \"/wd 4819 /wd 4267 /FS\"")

set(Libs glew64)
target_link_libraries(CPP_Fluid_Particles ${Libs})

#copy dlls
add_custom_command(
	TARGET CPP_Fluid_Particles
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CUDASAMPLESROOT}/bin/win64/Release"
	$<TARGET_FILE_DIR:CPP_Fluid_Particles>
	)
	
#copy shader files
add_custom_command(
	TARGET CPP_Fluid_Particles
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CUDASAMPLESROOT}/bin/win64/Release"
	$<TARGET_FILE_DIR:CPP_Fluid_Particles>
	COMMAND ${CMAKE_COMMAND} -E copy
	${CMAKE_CURRENT_SOURCE_DIR}/${shaderfile}
	$<TARGET_FILE_DIR:CPP_Fluid_Particles>
	)
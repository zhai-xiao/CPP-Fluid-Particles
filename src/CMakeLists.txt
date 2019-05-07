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

set(default_build_type "Release")

add_executable(fluidParticle
	${headers}
	${source}
	)

set(Libs freeglut glew64)
# Compiler Optimization:
#        1. C/C++ - Optimization - Optimization设为/O2
#        2. C/C++ - Optimization - Inline Function Expansion设为/Ob2
#        3. C/C++ - Optimization - Enable Intrinsic Functions设为/Oi
#        4. C/C++ - Optimization - Favor Size Or Speed设为/Ot
#        5. C/C++ - Code Generation - Enable Function-Level Linking设为Yes(/Gy)
#        6. CUDA C/C++ - Host - Use Fast Math设为Yes(-use_fast_math)
#        7. CUDA C/C++ - Host - Optimization设为/O2
# Linker Optimization
#        1. Linker - Optimization - Reference设为Yes(/OPT:REF)
#        2. Linker - Optimization - Enable COMDAT Folding设为Yes(/OPT:ICF)
#        3. CUDA Linker - General - Optimization设为/O2 (default)
# Acoid Compiler Collision
#        1. C/C++ - General - Multi-processor Compilation设为Yes(/MP)
#        2. CUDA C/C++ - Host - Additional Compiler Options添加/FS (default)
# Optimization options for C/C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Ot /Oi /Gy /MP")
# Optimization options for C/C++ Linkers
set_property(TARGET fluidParticle APPEND PROPERTY LINK_FLAGS "/OPT:REF /OPT:ICF")
# Optimization options for CUDA C/C++ 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-extended-lambda")

target_link_libraries(fluidParticle ${Libs})

#copy dlls
add_custom_command(
	TARGET fluidParticle
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
	"${CUDASAMPLESROOT}/bin/win64/Release"
	$<TARGET_FILE_DIR:fluidParticle>
	)
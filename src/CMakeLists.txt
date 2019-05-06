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

add_executable(fluidParticle
	${headers}
	${source}
	)

set(Libs freeglutd glew32d)
target_link_libraries(fluidParticle ${Libs})
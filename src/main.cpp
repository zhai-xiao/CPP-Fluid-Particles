// Copyright (C) 2019 Xiao Zhai
// 
// This file is part of CPP-Fluid-Particles.
// 
// CPP-Fluid-Particles is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// CPP-Fluid-Particles is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with CPP-Fluid-Particles.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <GL/freeglut.h>
#include <vector>
#include <memory>
#include "ShaderUtility.h"
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"
#include "BaseSolver.h"
#include "BasicSPHSolver.h"
#include "DFSPHSolver.h"
#include "PBDSolver.h"
#include "SPHSystem.h"

// vbo and GL variables
static GLuint particlesVBO;
static GLuint particlesColorVBO;
static GLuint m_particles_program;
static const int m_window_h = 700;
static const int m_fov = 30;
static const float particle_radius = 0.01f;
// view variables
static float rot[2] = { 0.0f, 0.0f };
static int mousePos[2] = { -1,-1 };
static bool mouse_left_down = false;
static float zoom = 0.3f;
// state variables
static int frameId = 0;
static float totalTime = 0.0f;
bool running = false;
// particle system variables
std::shared_ptr<SPHSystem> pSystem;
const float3 spaceSize = make_float3(1.0f);
const float sphSpacing = 0.02f;
const float sphSmoothingRadius = 2.0f * sphSpacing;
const float sphCellLength = 1.01f * sphSmoothingRadius;
const float dt = 0.002f;
const float sphRho0 = 1.0f;
const float sphRhoBoundary = 1.4f * sphRho0;
const float sphM0 = 76.596750762082e-6f;
const float sphStiff = 10.0f;
const float3 sphG = make_float3(0.0f, -9.8f, 0.0f);
const float sphVisc = 5e-4f;
const float sphSurfaceTensionIntensity = 0.0001f;
const float sphAirPressure = 0.0001f;
const int3 cellSize = make_int3(ceil(spaceSize.x / sphCellLength), ceil(spaceSize.y / sphCellLength), ceil(spaceSize.z / sphCellLength));

namespace fluid_solver {
	enum { SPH, DFSPH, PBD};
}

void initSPHSystem(const int solver = fluid_solver::PBD) {
	// initiate fluid particles
	std::vector<float3> pos;
	for (auto i = 0; i < 36; ++i) {
		for (auto j = 0; j < 24; ++j) {
			for (auto k = 0; k < 24; ++k) {
				auto x = make_float3(0.27f + sphSpacing * j,
					0.10f + sphSpacing * i,
					0.27f + sphSpacing * k);
				pos.push_back(x);
			}
		}
	}
	auto fluidParticles = std::make_shared<SPHParticles>(pos);
	// initiate boundary particles
	pos.clear();
	const auto compactSize = 2 * make_int3(ceil(spaceSize.x / sphCellLength), ceil(spaceSize.y / sphCellLength), ceil(spaceSize.z / sphCellLength));
	// front and back
	for (auto i = 0; i < compactSize.x; ++i) {
		for (auto j = 0; j < compactSize.y; ++j) {
			auto x = make_float3(i, j, 0) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(i, j, compactSize.z - 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	// top and bottom
	for (auto i = 0; i < compactSize.x; ++i) {
		for (auto j = 0; j < compactSize.z-2; ++j) {
			auto x = make_float3(i, 0, j+1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(i, compactSize.y - 1, j+1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	// left and right
	for (auto i = 0; i < compactSize.y - 2; ++i) {
		for (auto j = 0; j < compactSize.z - 2; ++j) {
			auto x = make_float3(0, i + 1, j + 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(compactSize.x - 1, i + 1, j + 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	auto boundaryParticles = std::make_shared<SPHParticles>(pos);
	// initiate solver and particle system
	std::shared_ptr<BaseSolver> pSolver;
	switch (solver) {
	case fluid_solver::PBD:
		pSolver = std::make_shared<PBDSolver>(fluidParticles->size());
		break;
	case fluid_solver::DFSPH:
		pSolver = std::make_shared<DFSPHSolver>(fluidParticles->size());
		break;
	default:
		pSolver = std::make_shared<BasicSPHSolver>(fluidParticles->size());
		break;
	}	
	pSystem = std::make_shared<SPHSystem>(fluidParticles, boundaryParticles, pSolver,
		spaceSize, sphCellLength, sphSmoothingRadius, dt, sphM0,
		sphRho0, sphRhoBoundary, sphStiff, sphVisc, 
		sphSurfaceTensionIntensity, sphAirPressure, sphG, cellSize);
}

void createVBO(GLuint* vbo, const unsigned int length) {
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, length, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA
	CUDA_CALL(cudaGLRegisterBufferObject(*vbo));
}

void deleteVBO(GLuint* vbo) {
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	CUDA_CALL(cudaGLUnregisterBufferObject(*vbo));

	*vbo = NULL;
}

void onClose(void) {
	deleteVBO(&particlesVBO);
	deleteVBO(&particlesColorVBO);
	pSystem = nullptr;
	CUDA_CALL(cudaDeviceReset());
	exit(0);
}

namespace particle_attributes {
	enum { POSITION, COLOR, SIZE, };
}

void initGL(void) {
	// create VBOs
	createVBO(&particlesVBO, sizeof(float3) * pSystem->size());
	createVBO(&particlesColorVBO, sizeof(float3) * pSystem->size());
	// initiate shader program
	m_particles_program = glCreateProgram();
	glBindAttribLocation(m_particles_program, particle_attributes::SIZE, "pointSize");
	ShaderUtility::attachAndLinkProgram(m_particles_program, ShaderUtility::loadShaders("particles.vert", "particles.frag"));
	return;
}

void mouseFunc(const int button, const int state, const int x, const int y) {
	if (GLUT_DOWN == state) {
		if (GLUT_LEFT_BUTTON == button) {
			mouse_left_down = true;
			mousePos[0] = x;
			mousePos[1] = y;
		}
		else if (GLUT_RIGHT_BUTTON == button) {}
	}
	else {
		mouse_left_down = false;
	}
	return;
}

void motionFunc(const int x, const int y) {
	int dx, dy;
	if (-1 == mousePos[0] && -1 == mousePos[1])
	{
		mousePos[0] = x;
		mousePos[1] = y;
		dx = dy = 0;
	}
	else
	{
		dx = x - mousePos[0];
		dy = y - mousePos[1];
	}
	if (mouse_left_down)
	{
		rot[0] += (float(dy) * 180.0f) / 720.0f;
		rot[1] += (float(dx) * 180.0f) / 720.0f;
	}

	mousePos[0] = x;
	mousePos[1] = y;

	glutPostRedisplay();
	return;
}

void keyboardFunc(const unsigned char key, const int x, const int y) {
	switch (key) {
	case '1':
		initSPHSystem(fluid_solver::SPH);
		frameId = 0;
		totalTime = 0.0f;
		break;
	case '2':
		initSPHSystem(fluid_solver::DFSPH);
		frameId = 0;
		totalTime = 0.0f;
		break;
	case '3':
		initSPHSystem(fluid_solver::PBD);
		frameId = 0;
		totalTime = 0.0f;
		break;
	case ' ':
		running = !running;
		break;
	case ',':
		zoom *= 1.2f;
		break;
	case '.':
		zoom /= 1.2f;
		break;
	case 'q':
	case 'Q':
		onClose();
		break;
	case 'r':
	case 'R':
		rot[0] = rot[1] = 0;
		zoom = 0.3f;
		break;
	case 'n':
	case 'N':
		void oneStep();
		oneStep();
		break;
	default:
		;
	}
}

extern "C" void generate_dots(float3* dot, float3* color, std::shared_ptr<SPHParticles> particles);

void renderParticles(void) {
	// map OpenGL buffer object for writing from CUDA
	float3 *dptr;
	float3 *cptr;
	CUDA_CALL(cudaGLMapBufferObject((void**)&dptr, particlesVBO));
	CUDA_CALL(cudaGLMapBufferObject((void**)&cptr, particlesColorVBO));

	// calculate the dots' position and color
	generate_dots(dptr, cptr, std::dynamic_pointer_cast<SPHParticles>(pSystem->getFluids()));

	// unmap buffer object
	CUDA_CALL(cudaGLUnmapBufferObject(particlesVBO));
	CUDA_CALL(cudaGLUnmapBufferObject(particlesColorVBO));


	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	glVertexPointer(3, GL_FLOAT, 0, nullptr);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, particlesColorVBO);
	glColorPointer(3, GL_FLOAT, 0, nullptr);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_POINTS, 0, pSystem->size());

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	return;
}

void oneStep() {
	++frameId;
	const auto milliseconds = pSystem->step();
	totalTime += milliseconds;
	printf("Frame %d - %2.2f ms, avg time - %2.2f ms/frame (%3.2f FPS)\r", 
		frameId%10000, milliseconds, totalTime / float(frameId), float(frameId)*1000.0f/totalTime);
}

void displayFunc(void) {
	if (running) {
		oneStep();
	}
	////////////////////
	glClearColor(0.9f, 0.9f, 0.92f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, m_window_h, m_window_h);
	glUseProgram(0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(m_fov, 1.0, 0.01, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 1.0 / zoom, 0, 0, 0, 0, 1, 0);

	glPushMatrix();
	glRotatef(rot[0], 1.0f, 0.0f, 0.0f);
	glRotatef(rot[1], 0.0f, 1.0f, 0.0f);
	glColor4f(0.7f, 0.7f, 0.7f, 1.0f);
	glLineWidth(1.0);
	////////////////////
	glEnable(GL_MULTISAMPLE_ARB);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); 
	glutSolidCube(1.0);
	////////////////////
	glUseProgram(m_particles_program);
	glUniform1f(glGetUniformLocation(m_particles_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*float(PI) / 180.0f));
	glUniform1f(glGetUniformLocation(m_particles_program, "pointRadius"), particle_radius);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glPushMatrix();
	glTranslatef(-.5, -.5, -.5);
	renderParticles();
	glPopMatrix();
	////////////////////
	glPopMatrix();
	glutSwapBuffers();
	glutPostRedisplay(); 
}

int main(int argc, char* argv[]) {
	try{
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE);
		glutInitWindowPosition(400, 0);
		glutInitWindowSize(m_window_h, m_window_h);
		glutCreateWindow("");
		glutDisplayFunc(&displayFunc);
		glutKeyboardFunc(&keyboardFunc);
		glutMouseFunc(&mouseFunc);
		glutMotionFunc(&motionFunc);

		glewInit();
		////////////////////
		initSPHSystem();
		initGL();

		std::cout << "Instructions\n";
		std::cout << "The color indicates the density of a particle.\nMagenta means higher density, navy means lesser density.\n";
		std::cout << "Controls\n";
		std::cout << "Space - Start/Pause\n";
		std::cout << "Key N - One Step Forward\n";
		std::cout << "Key Q - Quit\n";
		std::cout << "Key 1 - Restart Simulation Using SPH Solver\n";
		std::cout << "Key 2 - Restart Simulation Using DFSPH Solver\n";
		std::cout << "Key 3 - Restart Simulation Using PBD Solver\n";
		std::cout << "Key R - Reset Viewpoint\n";
		std::cout << "Key , - Zoom In\n";
		std::cout << "Key . - Zoom Out\n";
		std::cout << "Mouse Drag - Change Viewpoint\n\n";
		////////////////////
		glutMainLoop();
	}
	catch (...) {
		std::cout << "Unknown Exception at " << __FILE__ << ": line " << __LINE__ << "\n";
	}
	return 0;
}
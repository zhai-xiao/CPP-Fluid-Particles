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

#include "ShaderUtility.h"
#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <GL/glew.h>

namespace ShaderUtility {
	void* initGLEW()
	{
		// add for glew tools
		glewInit();
		if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
			fprintf(stderr, "Required OpenGL extensions missing.");
			exit(-1);
		}

#if defined (_WIN32)
		if (wglewIsSupported("WGL_EXT_swap_control")) {
			// disable vertical sync
			wglSwapIntervalEXT(0);
		}
#endif
	}

	char* loadFile(char *fname, GLint &fSize)
	{
		char * memblock;
		std::string text;

		// file read based on example in cplusplus.com tutorial
		std::ifstream file (fname, std::ios::in| std::ios::binary| std::ios::ate);
		if (file.is_open())
		{
			const auto size = file.tellg();
			fSize = GLuint(size);
			memblock = new char [size];
			file.seekg (0, std::ios::beg);
			file.read (memblock, size);
			file.close();
			//cout << "file " << fname << " loaded" << "\n";
			text.assign(memblock);
		}
		else
		{
			std::cout << "Unable to open file " << fname << "\n";
			exit(1);
		}
		return memblock;
	}

	// printShaderInfoLog
	// From OpenGL Shading Language 3rd Edition, p215-216
	// Display (hopefully) useful error messages if shader fails to compile
	void printShaderInfoLog(const GLint shader)
	{
		auto infoLogLen = 0;
		auto charsWritten = 0;

		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

		// should additionally check for OpenGL errors here

		if (infoLogLen > 0)
		{
			const auto infoLog = new GLchar[infoLogLen];
			// error check for fail to allocate memory omitted
			glGetShaderInfoLog(shader,infoLogLen, &charsWritten, infoLog);
			std::cout << "InfoLog:\n" << infoLog << "\n";
			delete [] infoLog;
		}

		// should additionally check for OpenGL errors here
	}

	void printLinkInfoLog(const GLint prog) 
	{
		auto infoLogLen = 0;
		auto charsWritten = 0;

		glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

		// should additionally check for OpenGL errors here

		if (infoLogLen > 0)
		{
			const auto infoLog = new GLchar[infoLogLen];
			// error check for fail to allocate memory omitted
			glGetProgramInfoLog(prog,infoLogLen, &charsWritten, infoLog);
			std::cout << "InfoLog:\n" << infoLog << "\n";
			delete [] infoLog;
		}
	}

	shaders_t loadShaders(char * vert_path, char * frag_path) {
		const auto v = glCreateShader(GL_VERTEX_SHADER);
		const auto f = glCreateShader(GL_FRAGMENT_SHADER);	

		// load shaders & get length of each
		GLint vlen;
		GLint flen;
		const auto vs = loadFile(vert_path, vlen);
		const auto fs = loadFile(frag_path, flen);

		const char * vv = vs;
		const char * ff = fs;

		glShaderSource(v, 1, &vv,&vlen);
		glShaderSource(f, 1, &ff,&flen);

		GLint compiled;

		glCompileShader(v);
		glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);
		if (!compiled)
		{
			std::cout << "Vertex shader not compiled.\n";
			printShaderInfoLog(v);
			system("PAUSE");
		} 

		glCompileShader(f);
		glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);
		if (!compiled)
		{
			std::cout << "Fragment shader not compiled.\n";
			printShaderInfoLog(f);
			system("PAUSE");
		} 
		shaders_t out; out.vertex = v; out.fragment = f;

		delete [] vs; // dont forget to free allocated memory
		delete [] fs; // we allocated this in the loadFile function...

		return out;
	}

	void attachAndLinkProgram(const GLuint program, const shaders_t shaders) {
		glAttachShader(program, shaders.vertex);
		glAttachShader(program, shaders.fragment);

		glLinkProgram(program);
		GLint linked;
		glGetProgramiv(program,GL_LINK_STATUS, &linked);
		if (!linked) 
		{
			std::cout << "Program did not link.\n";
			printLinkInfoLog(program);
		}
	}

}
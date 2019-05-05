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

#version 120


uniform float pointRadius;  // point size in world space

attribute float pointSize;
varying float fs_pointSize;

varying vec3 fs_PosEye;		// center of the viewpoint space
 
uniform float pointScale;
varying mat4 u_Persp;

void main(void) {

	vec3 posEye = (gl_ModelViewMatrix  * vec4(gl_Vertex.xyz, 1.0f)).xyz;
	float dist = length(posEye);

	
	gl_PointSize = pointRadius * pointScale/ dist;

	fs_PosEye = posEye;

	gl_FrontColor = gl_Color;
	
	u_Persp = gl_ProjectionMatrix;
	
	gl_Position = ftransform();
	fs_pointSize = pointSize;
}
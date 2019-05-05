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


varying float fs_pointSize;

varying vec3 fs_PosEye;
varying mat4 u_Persp;

void main(void)
{
    // calculate normal from texture coordinates
    vec3 N;

    N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);
    
    //calculate depth
    vec4 pixelPos = vec4(fs_PosEye + normalize(N)*fs_pointSize,1.0f);
    vec4 clipSpacePos = u_Persp * pixelPos;
    //gl_FragDepth = clipSpacePos.z / clipSpacePos.w;
    
    gl_FragColor = vec4(exp(-mag*mag)*gl_Color.rgb,1.0f);
	//gl_FragColor = vec4(vec3(0.03f),1.0f);
}

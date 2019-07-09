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

#pragma once

#include <helper_math.h>
#include "global.h"

static inline __device__ float cubic_spline_kernel(const float r, const float radius)
{
	const auto q = 2.0f * fabs(r) / radius;
	//return (q < EPSILON) ? 0.0f :
	//	((q) <= 1.0f ? (powf(2.0f - q, 3) - 4.0f * powf(1.0f - q, 3)) :
	//	(q) <= 2.0f ? (powf(2.0f - q, 3)) :
	//		0.0f) / (4.0f * PI * powf(radius, 3));
	if (q > 2.0f || q < EPSILON) return 0.0f;
	else {
		const auto a = 0.25f / (PI * radius * radius * radius);
		return a * ((q > 1.0f) ? (2.0f - q) * (2.0f - q) * (2.0f - q) : ((3.0f * q - 6.0f) * q * q + 4.0f));
	}
}

static inline __device__ float3 cubic_spline_kernel_gradient(const float3 r, const float radius)
{
	const auto q = 2.0f * length(r) / radius;
	//return
	//	((q) <= 1.0f ? -(3.0f * (2.0f - q) * (2.0f - q) - 12.0f * (1.0f - q) * (1.0f - q)) :
	//	(q) <= 2.0f ? -(3.0f * (2.0f - q) * (2.0f - q)) :
	//		0.0f) / (2.0f * PI * powf(radius, 4)) * r / fmaxf(EPSILON, length(r));
	if (q > 2.0f) return make_float3(0.0f);
	else {
		//const auto a = r / ((length(r) + EPSILON) * 2.0f * PI * radius * radius * radius * radius);
		const auto a = r / (PI * (q + EPSILON) * radius * radius * radius * radius * radius);
		return a * ((q > 1.0f) ? ((12.0f - 3.0f * q) * q - 12.0f) : ((9.0f * q - 12.0f) * q));
	}
}

static inline __device__ float viscosity_kernel_laplacian(const float r, const float radius) {
	return (r <= radius) ? (45.0f * (radius - r) / (PI * powf(radius, 6))) : 0.0f;
}

static __global__ void countingInCell_CUDA(int* cellStart, int* particle2cell, const int num)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	atomicAdd(&cellStart[particle2cell[i]], 1);
	return;
}

static inline __device__ int particlePos2cellIdx(const int3 pos, const int3 cellSize)
{
	// return (cellSize.x*cellSize.y*cellSize.z) if the particle is out of the grid
	return (pos.x >= 0 && pos.x < cellSize.x && pos.y >= 0 && pos.y < cellSize.y && pos.z >= 0 && pos.z < cellSize.z) ?
		(((pos.x * cellSize.y) + pos.y) * cellSize.z + pos.z)
		: (cellSize.x * cellSize.y * cellSize.z);
}

static __global__ void mapParticles2Cells_CUDA(int* particles2cells, float3* pos, const float cellLength, const int3 cellSize, const int num)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	particles2cells[i] = particlePos2cellIdx(make_int3(pos[i] / cellLength), cellSize);
	return;
}

// smoothing kernel in [2013][SIGGRAPH ASIA][Versatile Surface Tension and Adhesion for SPH Fluids].
// it's already been 3D spherical normalized.
inline __device__ float3 surface_tension_kernel_gradient(float3 r, const float radius)
{
	const auto x = length(r);
	//return
	//	(x < EPSILON) ? make_float3(0.0f) : (
	//		2.0f * x <= radius ? 2.0f * powf((radius - x), 3) * powf(x, 3) - 0.0156f * powf(radius, 6) :
	//		x <= radius ? powf((radius - x), 3) * powf(x, 3) :
	//		0.0f) * 136.0241f / (PI * powf(radius, 9)) * -r / fmaxf(EPSILON, x);
	if (x > radius || x < EPSILON) return make_float3(0.0f);
	else {
		auto cube = [](float x) {return x * x * x; };
		const float3 a = 136.0241f * -r / (PI * cube(radius) * cube(radius) * cube(radius) * x);
		return a * ((2.0f * x <= radius) ?
			(2.0f * cube(radius - x) * cube(x) - 0.0156f * cube(radius) * cube(radius)) :
			(cube(radius - x) * cube(x)));
	}
}

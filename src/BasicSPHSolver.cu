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

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include "CUDAFunctions.cuh"
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"
#include "BaseSolver.h"
#include "BasicSPHSolver.h"

__device__ void contributeFluidDensity(float* density, int i, float3* pos, float* mass, int cellStart, int cellEnd, float radius)
{
	int j = cellStart;
	while (j < cellEnd)
	{
		*density += mass[j] * cubic_spline_kernel(length(pos[i] - pos[j]), radius);
		j++;
	}
	return;
}

__device__ void contributeBoundaryDensity(float* density, float3 pos_i, float3* pos, float* mass, int cellStart, int cellEnd, float radius)
{
	int j = cellStart;
	while (j < cellEnd)
	{
		*density += mass[j] * cubic_spline_kernel(length(pos_i - pos[j]), radius);
		j++;
	}
	return;
}

__global__ void computeDensity_CUDA(float* density, int num,
	float3* posFluid, float* massFluid, int* cellStartFluid, 
	float3* posBoundary, float* massBoundary, int* cellStartBoundary, 
	int3 cellSize, float cellLength, float radius)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	int cellID;
	__syncthreads();
#pragma unroll
	for (int m = 0; m < 27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeFluidDensity(&density[i], i, posFluid, massFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius);
		contributeBoundaryDensity(&density[i], posFluid[i], posBoundary, massBoundary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius);
	}
	return;
}

void BasicSPHSolver::computeDensity(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, const int3 cellSize, const float cellLength, const float radius) {
	int num = fluids->size();
	thrust::fill(thrust::device, fluids->getDensityPtr(), fluids->getDensityPtr() + num, 0);
	computeDensity_CUDA <<<(num - 1) / block_size + 1, block_size >>> (fluids->getDensityPtr(), num,
		fluids->getPosPtr(), fluids->getMassPtr(), cellStartFluid.addr(),
		boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
		cellSize, cellLength, radius);
}

__global__ void enforceBoundary_CUDA(float3* pos, float3* vel, int num, float3 spaceSize)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	if (pos[i].x <= spaceSize.x * .00f) { pos[i].x = spaceSize.x * .00f;	vel[i].x = fmaxf(vel[i].x, 0.0f); }
	if (pos[i].x >= spaceSize.x * .99f) { pos[i].x = spaceSize.x * .99f;	vel[i].x = fminf(vel[i].x, 0.0f); }
	if (pos[i].y <= spaceSize.y * .00f) { pos[i].y = spaceSize.y * .00f;	vel[i].y = fmaxf(vel[i].y, 0.0f); }
	if (pos[i].y >= spaceSize.y * .99f) { pos[i].y = spaceSize.y * .99f;	vel[i].y = fminf(vel[i].y, 0.0f); }
	if (pos[i].z <= spaceSize.z * .00f) { pos[i].z = spaceSize.z * .00f;	vel[i].z = fmaxf(vel[i].z, 0.0f); }
	if (pos[i].z >= spaceSize.z * .99f) { pos[i].z = spaceSize.z * .99f;	vel[i].z = fminf(vel[i].z, 0.0f); }
	return;
}

void BasicSPHSolver::advect(std::shared_ptr<SPHParticles>& fluids, const float dt, const float3 spaceSize) {
	fluids->advect(dt);
	enforceBoundary_CUDA <<<((fluids->size())-1)/block_size+1, block_size >>> (fluids->getPosPtr(), fluids->getVelPtr(), fluids->size(), spaceSize);
}

__global__ void computePressure_CUDA(float* pressure, float* density, int num, float rho0, float stiff)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	pressure[i] = stiff * (powf((density[i] / rho0), 7) - 1.0f);
	//clamp
	if (pressure[i] < 0.0f) pressure[i] = 0.0f;
	return;
}

__device__ void contributeFluidPressure(float3* a, int i, float3* pos, float* mass, 
	float* density, float* pressure, int cellStart, int cellEnd, float radius)
{
	int j = cellStart;
	while (j < cellEnd)
	{
		if (i != j)
			* a += -mass[j] * 
			(pressure[i] / fmaxf(EPSILON, density[i] * density[i]) + pressure[j] / fmaxf(EPSILON, density[j] * density[j]))
			* cubic_spline_kernel_gradient(pos[i] - pos[j], radius);
		j++;
	}
	return;
}

__device__ void contributeBoundaryPressure(float3* a, float3 pos_i, float3* pos, float* mass, 
	float density, float pressure, int cellStart, int cellEnd, float radius)
{
	int j = cellStart;
	while (j < cellEnd)
	{
		*a += -mass[j] * (pressure / fmaxf(EPSILON, density * density)) * cubic_spline_kernel_gradient(pos_i - pos[j], radius);
		j++;
	}
	return;
}

__global__ void pressureForce_CUDA(float3* velFluid, float3* posFluid, float* massFluid, 
	float* density, float* pressure, int num, int* cellStartFluid, 
	float3* posBoundary, float* massBoundary, int* cellStartBoundary, 
	int3 cellSize, float cellLength, float radius, float dt)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	float3 a = make_float3(0.0f);
	int cellID;
	__syncthreads();
#pragma unroll
	for (int m = 0; m < 27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeFluidPressure(&a, i, posFluid, massFluid, density, pressure, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius);
		contributeBoundaryPressure(&a, posFluid[i], posBoundary, massBoundary, density[i], pressure[i], cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius);
	}

	// dirty trick to prevent blowups in large dt
	if (length(a) > MAX_A)
		a = normalize(a) * MAX_A;

	velFluid[i] += a * dt;
	return;
}

void BasicSPHSolver::project(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, const float rho0, const float stiff,
	const int3 cellSize, const float cellLength, const float radius, const float dt)
{
	int num = fluids->size();
	// step 1:: calculate density
	computeDensity(fluids, boundaries, cellStartFluid, cellStartBoundary, cellSize, cellLength, radius);
	// step 2: calculate pressure from density
	computePressure_CUDA <<<(num - 1) / block_size + 1, block_size >>> (fluids->getPressurePtr(), fluids->getDensityPtr(), num, rho0, stiff);
	// step 3: apply pressure force according to pressure
	pressureForce_CUDA <<<(num - 1) / block_size + 1, block_size >>> (fluids->getVelPtr(), fluids->getPosPtr(), fluids->getMassPtr(),
		fluids->getDensityPtr(), fluids->getPressurePtr(), num, cellStartFluid.addr(), 
		boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(), 
		cellSize, cellLength, radius, dt);
}

__device__ void contributeViscosity(float3* a, int i, float3* pos, float3* vel, 
	float* mass, float* density, int j/*cellStart*/, int cellEnd, float rho0, float radius) {
	while (j < cellEnd) {
		*a += mass[j] * ((vel[j] - vel[i]) / rho0) * viscosity_kernel_laplacian(length(pos[i] - pos[j]), radius);
		j++;
	}
	return;
}

__global__ void viscosity_CUDA(float3* deltaV, float3* vel, float3* pos, 
	float* mass, float* density, int num, int* cellStart, int3 cellSize, 
	float cellLength, float rho0, float radius, float visc, float dt) {
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	float3 a = make_float3(0.0f);
	int cellID;
	__syncthreads();
#pragma unroll
	for (int m = 0; m < 27; __syncthreads(), m++) {
		cellID = particlePos2cellIdx(make_int3(pos[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeViscosity(&a, i, pos, vel, mass, density, cellStart[cellID], cellStart[cellID + 1], rho0, radius);
	}

	deltaV[i] = visc * a * dt;
	return;
}

void BasicSPHSolver::diffuse(std::shared_ptr<SPHParticles>& fluids, const DArray<int>& cellStartFluid,
	const int3 cellSize, const float cellLength, const float rho0, 
	const float radius, const float visc, const float dt)
{
	int num = fluids->size();
	viscosity_CUDA <<<(num - 1) / block_size + 1, block_size >>> (bufferFloat3.addr(), fluids->getVelPtr(), fluids->getPosPtr(),
		fluids->getMassPtr(), fluids->getDensityPtr(), num, cellStartFluid.addr(), cellSize, 
		cellLength, rho0, radius, visc, dt);
	thrust::transform(thrust::device,
		fluids->getVelPtr(), fluids->getVelPtr() + num,
		bufferFloat3.addr(),
		fluids->getVelPtr(),
		thrust::plus<float3>()
	);
}

void BasicSPHSolver::force(std::shared_ptr<SPHParticles>& fluids, const float dt, const float3 G)
{
	const float3 dv = dt * G;
	thrust::transform(thrust::device,
		fluids->getVelPtr(), fluids->getVelPtr() + fluids->size(),
		fluids->getVelPtr(),
		ThrustHelper::plus<float3>(dv)
	);
}

void BasicSPHSolver::step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, const float3 spaceSize,
	const int3 cellSize, const float cellLength, const float radius, const float dt,
	const float rho0, const float rhoB, const float stiff, const float visc, const float3 G,
	const float surfaceTensionIntensity, const float airPressure)
{
	// step 1: non-pressure, non-viscosity force
	force(fluids, dt, G);
	// step 2: viscosity force, surface tension
	diffuse(fluids, cellStartFluid, cellSize,
		cellLength, rho0, radius,
		visc, dt);
	if (surfaceTensionIntensity > EPSILON || airPressure > EPSILON)
		handleSurface(fluids, boundaries,
			cellStartFluid, cellStartBoundary,
			rho0, rhoB, cellSize, cellLength, radius,
			dt, surfaceTensionIntensity, airPressure);
	// step 3: pressure force
	project(fluids, boundaries,
		cellStartFluid, cellStartBoundary, rho0, stiff,
		cellSize, cellLength, radius, dt);
	// step 4:: advection
	advect(fluids, dt, spaceSize);
}

void BasicSPHSolver::handleSurface(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries, 
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, 
	const float rho0, const float rhoB, const int3 cellSize, const float cellLength, const float radius,
	const float dt, const float surfaceTensionIntensity, const float airPressure)
{
	// the free surface handling method is from
	// [2014][TOG][Robust Simulation of Small-Scale Thin Features in SPH-based Free Surface Flows]
	// use bufferFloat3 as color gradient array
	surfaceDetection(bufferFloat3, fluids, boundaries,
		cellStartFluid, cellStartBoundary,
		rho0, rhoB, cellSize, cellLength, radius);
	applySurfaceEffects(fluids, bufferFloat3, cellStartFluid, 
		rho0, cellSize, cellLength,	radius, dt, surfaceTensionIntensity, airPressure);
}

__device__ void contributeColorGrad_fluid(float3& numerator, float& denominator, int i, float3* pos, float* mass, float* density, int j, int cellEnd, float radius, float rho0)
{
	while (j < cellEnd)
	{
		numerator += mass[j] / rho0 * cubic_spline_kernel_gradient(pos[i] - pos[j], radius);
		denominator += mass[j] / rho0 * cubic_spline_kernel(length(pos[i] - pos[j]), radius);
		j++;
	}
	return;
}

__device__ void contributeColorGrad_boundary(float3& numerator, float& denominator, float3* pos_i, float3* pos, float* mass, int j, int cellEnd, float radius, float rhoB)
{
	while (j < cellEnd)
	{
		numerator += mass[j] / rhoB * cubic_spline_kernel_gradient(*pos_i - pos[j], radius);
		denominator += mass[j] / rhoB * cubic_spline_kernel(length(*pos_i - pos[j]), radius);
		j++;
	}
	return;
}

__global__ void computeColorGrad_CUDA(float3* colorGrad, float3* posFluid, float* massFluid, float* density, int num, int* cellStartFluid, int3 cellSize,
	float3* posBoundary, float* massBoudnary, int* cellStartBoundary, float cellLength, float radius, float rho0, float rhoB)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	float3 c_g = make_float3(0.0f);
	float denominator = 0.0f;
	int cellID;
#pragma unroll
	for (int m = 0; m < 27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeColorGrad_fluid(c_g, denominator, i, posFluid, massFluid, density, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius, rho0);
		contributeColorGrad_boundary(c_g, denominator, &posFluid[i], posBoundary, massBoudnary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius, rhoB);
	}

	colorGrad[i] = c_g / fmaxf(EPSILON, denominator);
	return;
}

void BasicSPHSolver::surfaceDetection(DArray<float3>& colorGrad,
	const std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
	const float rho0, const float rhoB, const int3 cellSize, const float cellLength, const float radius)
{
	computeColorGrad_CUDA <<<(fluids->size()-1)/block_size+1, block_size>>> (colorGrad.addr(),
		fluids->getPosPtr(), fluids->getMassPtr(), fluids->getDensityPtr(), fluids->size(), 
		cellStartFluid.addr(), cellSize, 
		boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
		cellLength, radius, rho0, rhoB);
	return;
}

__device__ void contributeSurfaceTensionAndAirPressure(float3& a, int i, int cellID, float3* pos, float* density, float* mass, 
	float3* color_grad, int j, int cellEnd, float radius,
	float rho0, float color_energy_coefficient, float airPressure)
{
	while (j < cellEnd)
	{
		// surface tension
		a += 0.25f * mass[j] / (rho0 * rho0) * color_energy_coefficient
			* (dot(color_grad[i], color_grad[i]) + dot(color_grad[j], color_grad[j]))
			* surface_tension_kernel_gradient(pos[i] - pos[j], radius);
		// air pressure
		a += airPressure * mass[j] / (rho0 * rho0)
			* cubic_spline_kernel_gradient(pos[i] - pos[j], radius)
			/*following terms disable inner particles*/
			* length(color_grad[i]) / fmaxf(EPSILON, length(color_grad[i]));
		j++;
	}
	return;
}

__global__ void surfaceTensionAndAirPressure_CUDA(float3* vel, float3* pos_fluid, float* density, float* mass_fluid, 
	float3* color_grad, int num, int* cellStart, int3 cellSize, float cellLength, float radius, float dt,
	float rho0, float color_energy_coefficient, float airPressure)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	float3 a = make_float3(0.0f);
	int cellID;
#pragma unroll
	for (int m = 0; m < 27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeSurfaceTensionAndAirPressure(a, i, cellID, pos_fluid, density, mass_fluid, color_grad, 
			cellStart[cellID], cellStart[cellID + 1], radius, rho0, color_energy_coefficient, airPressure);
	}
	vel[i] += a * dt;
	return;
}

void BasicSPHSolver::applySurfaceEffects(std::shared_ptr<SPHParticles>& fluids, const DArray<float3>& colorGrad, 
	const DArray<int>& cellStartFluid, const float rho0, const int3 cellSize, const float cellLength, 
	const float radius, const float dt, const float surfaceTensionIntensity, const float airPressure)
{
	int num = fluids->size();
	surfaceTensionAndAirPressure_CUDA <<<(num - 1) / block_size + 1, block_size>>> (fluids->getVelPtr(),
		fluids->getPosPtr(), fluids->getDensityPtr(), fluids->getMassPtr(),
		colorGrad.addr(), num, cellStartFluid.addr(), cellSize, cellLength, radius, dt,
		rho0, surfaceTensionIntensity, airPressure);
}

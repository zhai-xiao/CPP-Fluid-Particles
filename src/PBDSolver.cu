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

#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include "CUDAFunctions.cuh"
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"
#include "BaseSolver.h"
#include "BasicSPHSolver.h"
#include "PBDSolver.h"

void PBDSolver::step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, const float3 spaceSize,
	const int3 cellSize, const float cellLength, const float radius, const float dt,
	const float rho0, const float rhoB, const float stiff, const float visc, const float3 G,
	const float surfaceTensionIntensity, const float airPressure)
{
	// the order of steps is slighted adjusted to accomodate the call from step() in SPHSystem.cu

	// Position-based Fluids need the position in last timestep to calculate velocity.
	// If it is not provided, use the current position as the history position in the next timestep.
	if (!posLastInitialized) {
		initializePosLast(fluids->getPos());
		throw "PBD: The last position of fluids is initialized.";
	}
	//// step 1: update local neighborhood
	updateNeighborhood(fluids);
	// step 2: iteratively correct position
	project(fluids, boundaries,
	        cellStartFluid, cellStartBoundary,
	        rho0, cellSize, spaceSize, cellLength, radius, maxIter);
	// step 3: calculate velocity
	thrust::transform(thrust::device,
		fluids->getPosPtr(), fluids->getPosPtr() + fluids->size(),
		fluidPosLast.addr(),
		fluids->getVelPtr(),
		[dt]__host__ __device__(const float3& lhs, const float3& rhs) { return (lhs - rhs)/dt; }
	);
	// step 4: apply non-pressure forces (gravity, XSPH viscosity and surface tension)
	diffuse(fluids, cellStartFluid, cellSize,
	        cellLength, rho0, radius, xSPH_c);
	if (surfaceTensionIntensity > EPSILON || airPressure > EPSILON)
		handleSurface(fluids, boundaries,
			cellStartFluid, cellStartBoundary,
			rho0, rhoB, cellSize, cellLength, radius,
			dt, surfaceTensionIntensity, airPressure);
	force(fluids, dt, G);

	// step 5: predict position for next timestep
	predict(fluids, dt, spaceSize);
}

void PBDSolver::predict(std::shared_ptr<SPHParticles>& fluids, const float dt, const float3 spaceSize)
{
	CUDA_CALL(cudaMemcpy(fluidPosLast.addr(), fluids->getPosPtr(), sizeof(float3) * fluids->size(), cudaMemcpyDeviceToDevice));
	advect(fluids, dt, spaceSize);
}

void PBDSolver::updateNeighborhood(const std::shared_ptr<SPHParticles>& particles)
{
	const int num = particles->size();
	CUDA_CALL(cudaMemcpy(bufferInt.addr(), particles->getParticle2Cell(), sizeof(int) * num, cudaMemcpyDeviceToDevice));
	thrust::sort_by_key(thrust::device, bufferInt.addr(), bufferInt.addr() + num, fluidPosLast.addr());
	return;
}

__device__ void contributeXSPHViscosity(float3* a, const int i, float3* pos, float3* vel,
	float* mass, int j/*cellStart*/, const int cellEnd, const float radius) {
	while (j < cellEnd) {
		*a += mass[j]*(vel[j] - vel[i]) * cubic_spline_kernel(length(pos[i] - pos[j]), radius);
		++j;
	}
	return;
}

__global__ void XSPHViscosity_CUDA(float3* vel, float3* pos,
	float *mass, const int num, int* cellStart, const int3 cellSize,
	const float cellLength, const float radius, const float visc, const float rho0) {
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	auto a = make_float3(0.0f);
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m) {
		const auto cellID = particlePos2cellIdx(make_int3(pos[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1),
		                                 cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeXSPHViscosity(&a, i, pos, vel, mass, cellStart[cellID], cellStart[cellID + 1], radius);
	}

	vel[i] += visc * a/rho0;
	return;
}

void PBDSolver::diffuse(std::shared_ptr<SPHParticles>& fluids, const DArray<int>& cellStartFluid,
                        const int3 cellSize, const float cellLength, const float rho0,
                        const float radius, const float visc)
{
	int num = fluids->size();
	XSPHViscosity_CUDA <<<(num - 1) / block_size + 1, block_size>>> (fluids->getVelPtr(), fluids->getPosPtr(),
		fluids->getMassPtr(), num, cellStartFluid.addr(), cellSize,
		cellLength, radius, visc, rho0);
}

__device__ void contributeDensityLambda(float& density, float3& gradientSum, float& sampleLambda, const float3 pos_i, float3* pos, 
	float* mass, int j, const int cellEnd, const bool rho0, const float radius)
{
	while (j < cellEnd)
	{
		density += mass[j] * cubic_spline_kernel(length(pos_i - pos[j]), radius);
		const auto gradient = - mass[j] * cubic_spline_kernel_gradient(pos_i - pos[j], radius) / rho0;
		gradientSum -= gradient;
		sampleLambda += dot(gradient, gradient);
		++j;
	}
	return;
}

__global__ void computeDensityLambda_CUDA(float* density, float* lambda,
	float3* posFluid, float* massFluid, const int num, int* cellStartFluid, const int3 cellSize,
	float3* posBoundary, float* massBoundary, int* cellStartBoundary,
	const float cellLength, const float radius, const float rho0, const float relaxation)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	auto gradientSum = make_float3(0.0f);
	auto sampleLambda = 0.0f;
	auto den = 0.0f;
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeDensityLambda(den, gradientSum, sampleLambda, posFluid[i], posFluid, massFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], rho0, radius);
		contributeDensityLambda(den, gradientSum, sampleLambda, posFluid[i], posBoundary, massBoundary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], rho0, radius);
	}

	density[i] = den;
	lambda[i] = (den > rho0) ? 
		(-(den / rho0 - 1.0f) / (dot(gradientSum, gradientSum) + sampleLambda + EPSILON)) 
		: 0.0f;
	lambda[i] *= relaxation;
	return;
}

__device__ void contributeDeltaPos_fluid(float3& a, const int i, float3* pos, float* lambda, float *mass, int j, const int cellEnd, const float radius)
{
	while (j < cellEnd)
	{
		a += mass[j] * (lambda[i] + lambda[j]) * cubic_spline_kernel_gradient(pos[i] - pos[j], radius);
		++j;
	}
	return;
}

__device__ void contributeDeltaPos_boundary(float3& a, const float3 pos_i, float3* pos, const float lambda_i, float* mass, int j, const int cellEnd, const float radius)
{
	while (j < cellEnd)
	{
		a += mass[j] * (lambda_i)* cubic_spline_kernel_gradient(pos_i - pos[j], radius);
		++j;
	}
	return;
}

__global__ void computeDeltaPos_CUDA(float3* deltaPos, float3* posFluid, float3* posBoundary, float *lambda,
                                     float *massFluid, float *massBoundary, const int num, int* cellStartFluid, int* cellStartBoundary, const int3 cellSize,
                                     const float cellLength, const float radius, const float rho0)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	auto dp = make_float3(0.0f);
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeDeltaPos_fluid(dp, i, posFluid, lambda, massFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius);
		contributeDeltaPos_boundary(dp, posFluid[i], posBoundary, lambda[i], massBoundary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius);
	}

	deltaPos[i] = dp/rho0;
	return;
}

__global__ void enforceBoundary_CUDA(float3* pos, const int num, const float3 spaceSize)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	if (pos[i].x <= spaceSize.x * .00f) { pos[i].x = spaceSize.x * .00f; }
	if (pos[i].x >= spaceSize.x * .99f) { pos[i].x = spaceSize.x * .99f; }
	if (pos[i].y <= spaceSize.y * .00f) { pos[i].y = spaceSize.y * .00f; }
	if (pos[i].y >= spaceSize.y * .99f) { pos[i].y = spaceSize.y * .99f; }
	if (pos[i].z <= spaceSize.z * .00f) { pos[i].z = spaceSize.z * .00f; }
	if (pos[i].z >= spaceSize.z * .99f) { pos[i].z = spaceSize.z * .99f; }
	return;
}

int PBDSolver::project(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
                       const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
                       const float rho0, const int3 cellSize, const float3 spaceSize, const float cellLength,
                       const float radius, const int maxIter)
{
	int num = fluids->size();
	auto iter = 0;
	while (iter < maxIter) {
		// step 1: compute lambda
		// use bufferFloat as lambda
		computeDensityLambda_CUDA <<<(num - 1) / block_size + 1, block_size >>> (fluids->getDensityPtr(), bufferFloat.addr(),
			fluids->getPosPtr(), fluids->getMassPtr(), fluids->size(), cellStartFluid.addr(), cellSize,
			boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
			cellLength, radius, rho0, relaxation);
		// step 2: compute Delta pos for density correction
		// use bufferFloat3 as Delta pos
		computeDeltaPos_CUDA << <(num - 1) / block_size + 1, block_size >> > (bufferFloat3.addr(),
		                                                                      fluids->getPosPtr(), boundaries->getPosPtr(), bufferFloat.addr(),
		                                                                      fluids->getMassPtr(), boundaries->getMassPtr(), num,
		                                                                      cellStartFluid.addr(), cellStartBoundary.addr(), cellSize,
		                                                                      cellLength, radius, rho0);
		// step 3: update pos
		thrust::transform(thrust::device,
			fluids->getPosPtr(), fluids->getPosPtr() + num,
			bufferFloat3.addr(),
			fluids->getPosPtr(),
			thrust::plus<float3>());
		enforceBoundary_CUDA <<<(num - 1) / block_size + 1, block_size>>> 
			(fluids->getPosPtr(), num, spaceSize);
		
		++iter;
	}
	return iter;
}
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
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include "CUDAFunctions.cuh"
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"
#include "BaseSolver.h"
#include "BasicSPHSolver.h"
#include "DFSPHSolver.h"

void DFSPHSolver::step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, const float3 spaceSize,
	const int3 cellSize, const float cellLength, const float radius, const float dt,
	const float rho0, const float rhoB, const float stiff, const float visc, const float3 G,
	const float surfaceTensionIntensity, const float airPressure)
{
	// the order of steps is slighted adjusted to accomodate the call from step() in SPHSystem.cu

	// step 1: update neighborhodd
	// done by caller of this function

	// step 2: compute density and alpha
	computeDensityAlpha(fluids, boundaries, cellStartFluid, cellStartBoundary,
		cellSize, cellLength, radius);

	// step 3: correct divergence error
	auto it_div = correctDivergenceError(fluids, boundaries, cellStartFluid, cellStartBoundary,
		rho0, cellSize, cellLength, radius, dt,
		divergenceErrorThreshold, maxIter);

	// step 4: non-pressure forces
	force(fluids, dt, G);
	diffuse(fluids, cellStartFluid, cellSize,
	        cellLength, rho0, radius,
	        visc, dt);
	if (surfaceTensionIntensity > EPSILON || airPressure > EPSILON)
		handleSurface(fluids, boundaries,
			cellStartFluid, cellStartBoundary,
			rho0, rhoB, cellSize, cellLength, radius,
			dt, surfaceTensionIntensity, airPressure);

	// step 5: correct density error
	auto it_den = project(fluids, boundaries,
		cellStartFluid, cellStartBoundary,
		rho0, cellSize, cellLength, radius, dt,
		densityErrorThreshold, maxIter);

	// step 6: advect
	advect(fluids, dt, spaceSize);
}

__device__ auto contributeDensityError_fluid(float& e, const int i, float3* pos, float3* vel, float* mass, int j, const int cellEnd, const float radius) -> void
{
	while (j < cellEnd)
	{
		e += mass[j] * dot((vel[i] - vel[j]), cubic_spline_kernel_gradient(pos[i] - pos[j], radius));
		++j;
	}
	return;
}

__device__ void contributeDensityError_boundary(float& e, const float3 vel_i, const float3 pos_i, float3* pos, float* mass, int j, const int cellEnd, const float radius)
{
	while (j < cellEnd)
	{
		e += mass[j] * dot(vel_i, cubic_spline_kernel_gradient(pos_i - pos[j], radius));
		++j;
	}
	return;
}

__global__ void computeDensityError_CUDA(float* error, float* stiff, float3* posFluid, float3* velFluid, float* massFluid, const int num, 
	float* density, float* alpha, int* cellStartFluid, const int3 cellSize, const float cellLength, const float dt, const float  rho0, const float radius,
	float3* posBoundary, float* massBoundary, int* cellStartBoundary)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	auto e = 0.0f;
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeDensityError_fluid(e, i, posFluid, velFluid, massFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius);
		contributeDensityError_boundary(e, velFluid[i], posFluid[i], posBoundary, massBoundary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius);
	}

	//clamp
	error[i] = fmaxf(0.0f, dt * e + density[i] - rho0);
	stiff[i] = error[i] * alpha[i];
	return;
}

__device__ void contributeAcceleration_fluid(float3& a, const int i, float3* pos, float* mass, float* stiff, int j, const int cellEnd, const float radius)
{
	while (j < cellEnd)
	{
		a += mass[j] * (stiff[i] + stiff[j]) * cubic_spline_kernel_gradient(pos[i] - pos[j], radius);
		++j;
	}
	return;
}

__device__ void contributeAcceleration_boundary(float3& a, const float3 pos_i, float3* pos, float* mass, const float stiff_i, int j, const int cellEnd, const float radius)
{
	while (j < cellEnd)
	{
		a += mass[j] * (stiff_i)* cubic_spline_kernel_gradient(pos_i - pos[j], radius);
		++j;
	}
	return;
}

__global__ void correctDensityError_CUDA(float3* velFluid, float3* posFluid, float* massFluid, float* stiff, const int num, 
	int* cellStartFluid, const int3 cellSize,
	float3* posBoundary, float* massBoundary, int* cellStartBoundary, const float cellLength, const float dt, const float radius)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	auto a = make_float3(0.0f);
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeAcceleration_fluid(a, i, posFluid, massFluid, stiff, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius);
		contributeAcceleration_boundary(a, posFluid[i], posBoundary, massBoundary, stiff[i], cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius);
	}

	velFluid[i] += a / dt;
	return;
}

int DFSPHSolver::project(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
	const float rho0, const int3 cellSize, const float cellLength, const float radius, const float dt,
	const float errorThreshold, const int maxIter)
{
	int num = fluids->size();
	auto totalError = std::numeric_limits<float>::max();
	auto iter = 0;

	// gather warm stiffness from last time step using particle2cell table
	CUDA_CALL(cudaMemcpy(bufferInt.addr(), fluids->getParticle2Cell(), sizeof(int) * num, cudaMemcpyDeviceToDevice));
	thrust::sort_by_key(thrust::device, bufferInt.addr(), bufferInt.addr() + num, denWarmStiff.addr());
	// warm start
	correctDensityError_CUDA << <(num - 1) / block_size + 1, block_size >> > (fluids->getVelPtr(),
		fluids->getPosPtr(), fluids->getMassPtr(), denWarmStiff.addr(), num,
		cellStartFluid.addr(), cellSize,
		boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
		cellLength, dt, radius);

	// bufferFloat act as the stiffness array in the paper
	computeDensityError_CUDA <<<(num-1)/block_size+1, block_size>>> (error.addr(), bufferFloat.addr(), 
		fluids->getPosPtr(), fluids->getVelPtr(), fluids->getMassPtr(), num, 
		fluids->getDensityPtr(), alpha.addr(), cellStartFluid.addr(), cellSize, cellLength, dt, rho0, radius,
		boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr());
	// reset warm stiffness
	CUDA_CALL(cudaMemcpy(denWarmStiff.addr(), bufferFloat.addr(), sizeof(float) * num, cudaMemcpyDeviceToDevice));

	while ((iter<2 || totalError>errorThreshold*num * rho0) && iter < maxIter)
	{
		correctDensityError_CUDA <<<(num - 1) / block_size + 1, block_size >>> (fluids->getVelPtr(), 
			fluids->getPosPtr(), fluids->getMassPtr(), bufferFloat.addr(), num, 
			cellStartFluid.addr(), cellSize, 
			boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(), 
			cellLength, dt, radius);
		computeDensityError_CUDA << <(num - 1) / block_size + 1, block_size >> > (error.addr(), bufferFloat.addr(),
			fluids->getPosPtr(), fluids->getVelPtr(), fluids->getMassPtr(), num,
			fluids->getDensityPtr(), alpha.addr(), cellStartFluid.addr(), cellSize, cellLength, dt, rho0, radius,
			boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr());
		// accumulate warm stiffness
		thrust::transform(thrust::device,
			denWarmStiff.addr(), denWarmStiff.addr() + num,
			bufferFloat.addr(),
			denWarmStiff.addr(),
			thrust::plus<float>());
		++iter;
		if (iter >= 2) {
			totalError = thrust::reduce(thrust::device, error.addr(), error.addr() + num, 0.0f, ThrustHelper::abs_plus<float>());
		}
	}
	return iter;
}

__device__ void contributeDensityAlpha(float& density, float3& gradientSum, float& sampleLambda, const float3 pos_i, float3* pos, float* mass, int j, const int cellEnd, const bool isBoundary, const float radius)
{
	while (j < cellEnd)
	{
		density += mass[j] * cubic_spline_kernel(length(pos_i - pos[j]), radius);
		gradientSum += mass[j] * cubic_spline_kernel_gradient(pos_i - pos[j], radius);
		if (!isBoundary)
			sampleLambda += dot(mass[j] * cubic_spline_kernel_gradient(pos_i - pos[j], radius), mass[j] * cubic_spline_kernel_gradient(pos_i - pos[j], radius));
		++j;
	}
	return;
}

__global__ void computeDensityAlpha_CUDA(float* density, float* alpha, 
	float3* posFluid, float* massFluid, const int num, int* cellStartFluid, const int3 cellSize,
	float3* posBoundary, float* massBoundary, int* cellStartBoundary,
	const float cellLength, const float radius)
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
		contributeDensityAlpha(den, gradientSum, sampleLambda, posFluid[i], posFluid, massFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], false, radius);
		contributeDensityAlpha(den, gradientSum, sampleLambda, posFluid[i], posBoundary, massBoundary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], true, radius);
	}

	density[i] = den;
	alpha[i] = -1.0f / fmaxf(EPSILON, (dot(gradientSum, gradientSum) + sampleLambda));
	return;
}

void DFSPHSolver::computeDensityAlpha(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, const int3 cellSize, const float cellLength, const float radius)
{
	computeDensityAlpha_CUDA <<<(fluids->size()-1)/block_size+1, block_size >>> (fluids->getDensityPtr(), alpha.addr(), 
		fluids->getPosPtr(), fluids->getMassPtr(), fluids->size(), cellStartFluid.addr(), cellSize, 
		boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
		cellLength, radius);
}

__device__ void contributeDivergenceError_fluid(float& e, const int i, float3* pos, float3* vel, float* mass, int j, const int cellEnd, const float radius)
{
	while (j < cellEnd)
	{
		e += mass[j] * dot((vel[i] - vel[j]), cubic_spline_kernel_gradient(pos[i] - pos[j], radius));
		++j;
	}
	return;
}

__device__ void contributeDivergenceError_boundary(float& e, const float3 pos_i, const float3 vel_i, float3* pos, float* mass, int j, const int cellEnd, const float radius)
{
	while (j < cellEnd)
	{
		e += mass[j] * dot(vel_i, cubic_spline_kernel_gradient(pos_i - pos[j], radius));
		++j;
	}
	return;
}

__global__ void computeDivergenceError_CUDA(float* error, float* stiff, 
	float3* posFluid, float3* velFluid, float* massFluid, float* density, const int num, 
	float* alpha, int* cellStartFluid, const int3 cellSize,
	float3* posBoundary, float* massBoundary, int* cellStartBoundary,
	const float cellLength, const float dt, const float rho0, const float radius)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	auto e = 0.0f;
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeDivergenceError_fluid(e, i, posFluid, velFluid, massFluid, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius);
		contributeDivergenceError_boundary(e, posFluid[i], velFluid[i], posBoundary, massBoundary, cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius);
	}
	error[i] = fmaxf(0.0f, e);
	// clamp: if the predicted density is less than the rest density, compress is allowed
	if (density[i] + dt * error[i] < rho0 && density[i] <= rho0)
		error[i] = 0.0f;
	stiff[i] = error[i] * alpha[i];
	return;
}

__global__ void correctDivergenceError_CUDA(float3* velFluid, float3* posFluid, float* massFluid, float* stiff, const int num, 
	int* cellStartFluid, const int3 cellSize,
	float3* posBoundary, float* massBoundary, int* cellStartBoundary, const float cellLength, const float radius)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	auto a = make_float3(0.0f);
	__syncthreads();
#pragma unroll
	for (auto m = 0; m < 27; __syncthreads(), ++m)
	{
		const auto cellID = particlePos2cellIdx(
			make_int3(posFluid[i] / cellLength) + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		if (cellID == (cellSize.x * cellSize.y * cellSize.z)) continue;
		contributeAcceleration_fluid(a, i, posFluid, massFluid, stiff, cellStartFluid[cellID], cellStartFluid[cellID + 1], radius);
		contributeAcceleration_boundary(a, posFluid[i], posBoundary, massBoundary, stiff[i], cellStartBoundary[cellID], cellStartBoundary[cellID + 1], radius);

	}

	velFluid[i] += a; // dt is already included in a
	return;
}

int DFSPHSolver::correctDivergenceError(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
	const float rho0, const int3 cellSize, const float cellLength, const float radius, const float dt, 
	const float errorThreshold, const int maxIter)
{
	int num = fluids->size();
	auto totalError = std::numeric_limits<float>::max();
	auto iter = 0;

	// bufferFloat again act as the stiffness array
	computeDivergenceError_CUDA <<<(num-1)/block_size+1, block_size>>> (error.addr(), bufferFloat.addr(), 
		fluids->getPosPtr(), fluids->getVelPtr(), fluids->getMassPtr(), fluids->getDensityPtr(), num, 
		alpha.addr(), cellStartFluid.addr(), cellSize, 
		boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
		cellLength, dt, rho0, radius);

	while ((iter<1 || totalError>errorThreshold *num* rho0) && iter < maxIter)
	{
		correctDivergenceError_CUDA <<<(num - 1) / block_size + 1, block_size>>> (fluids->getVelPtr(), 
			fluids->getPosPtr(), fluids->getMassPtr(), bufferFloat.addr(), num, 
			cellStartFluid.addr(), cellSize, 
			boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
			cellLength, radius);
		computeDivergenceError_CUDA <<<(num - 1) / block_size + 1, block_size>>> (error.addr(), bufferFloat.addr(),
			fluids->getPosPtr(), fluids->getVelPtr(), fluids->getMassPtr(), fluids->getDensityPtr(), num,
			alpha.addr(), cellStartFluid.addr(), cellSize,
			boundaries->getPosPtr(), boundaries->getMassPtr(), cellStartBoundary.addr(),
			cellLength, dt, rho0, radius);
		++iter;
		totalError = thrust::reduce(thrust::device, error.addr(), error.addr() + num, 0.0f, ThrustHelper::abs_plus<float>());
	}
	return iter;
}

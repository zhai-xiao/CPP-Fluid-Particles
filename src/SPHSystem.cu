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
#include <helper_math.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "CUDAFunctions.cuh"
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"
#include "BaseSolver.h"
#include "SPHSystem.h"

SPHSystem::SPHSystem(
	std::shared_ptr<SPHParticles>& fluidParticles,
	std::shared_ptr<SPHParticles>& boundaryParticles,
	std::shared_ptr<BaseSolver>& solver,
	const float3 spaceSize,
	const float sphCellLength,
	const float sphSmoothingRadius,
	const float dt,
	const float sphM0,
	const float sphRho0,
	const float sphRhoBoundary,
	const float sphStiff,
	const float sphVisc,
	const float sphSurfaceTensionIntensity,
	const float sphAirPressure,
	const float3 sphG,
	const int3 cellSize)
	:_fluids(fluidParticles), _boundaries(boundaryParticles),
	_solver(solver),
	_spaceSize(spaceSize),
	_sphCellLength(sphCellLength),
	_sphSmoothingRadius(sphSmoothingRadius),
	_dt(dt),
	_sphRho0(sphRho0),
	_sphRhoBoundary(sphRhoBoundary),
	_sphStiff(sphStiff),
	_sphVisc(sphVisc),
	_sphSurfaceTensionIntensity(sphSurfaceTensionIntensity),
	_sphAirPressure(sphAirPressure),
	_sphG(sphG),
	_cellSize(cellSize),
	cellStartFluid(cellSize.x* cellSize.y* cellSize.z + 1),
	cellStartBoundary(cellSize.x* cellSize.y* cellSize.z + 1),
	bufferInt(max(totalSize(), cellSize.x* cellSize.y* cellSize.z + 1))
{
	// step 1: init boundary particles
	neighborSearch(_boundaries, cellStartBoundary);
	// step 2: calculate boundary particles' mass
	computeBoundaryMass();
	// step 3: init fluid particles
	thrust::fill(thrust::device, _fluids->getMassPtr(), _fluids->getMassPtr() + _fluids->size(), sphM0);
	neighborSearch(_fluids, cellStartFluid);
	// step 4: fill all fluid particles' properties by calling step()
	step();
}

__device__ void contributeBoundaryKernel(float* sum_kernel, int i, int cellID, float3* pos, int* cellStart, int3 cellSize, float radius)
{
	int j, end;
	if (cellID == (cellSize.x * cellSize.y * cellSize.z)) return;
	j = cellStart[cellID];	end = cellStart[cellID + 1];
	while (j < end)
	{
		*sum_kernel += cubic_spline_kernel(length(pos[i] - pos[j]), radius);
		j++;
	}
	return;
}

__global__ void computeBoundaryMass_CUDA(float* mass, float3* pos, int num, int* cellStart, int3 cellSize, float cellLength, float rhoB, float radius)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;
	int3 cellPos = make_int3(pos[i] / cellLength);
	int cellID;
#pragma unroll
	for (int m = 0; m < 27; m++)
	{
		cellID = particlePos2cellIdx(cellPos + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1), cellSize);
		contributeBoundaryKernel(&mass[i], i, cellID, pos, cellStart, cellSize, radius);
	}
	mass[i] = rhoB / fmaxf(EPSILON, mass[i]);
	return;
}

void SPHSystem::computeBoundaryMass() {
	computeBoundaryMass_CUDA <<<(_boundaries->size() - 1) / block_size + 1, block_size >>> (
		_boundaries->getMassPtr(), _boundaries->getPosPtr(), _boundaries->size(),
		cellStartBoundary.addr(), _cellSize, _sphCellLength, _sphRhoBoundary, _sphSmoothingRadius);
}

void SPHSystem::neighborSearch(const std::shared_ptr<SPHParticles> &particles, DArray<int> &cellStart)
{
	int num = particles->size();
	mapParticles2Cells_CUDA <<<(num - 1) / block_size + 1, block_size >>> (particles->getParticle2Cell(), particles->getPosPtr(), _sphCellLength, _cellSize, num);
	CUDA_CALL(cudaMemcpy(bufferInt.addr(), particles->getParticle2Cell(), sizeof(int) * num, cudaMemcpyDeviceToDevice));
	thrust::sort_by_key(thrust::device, bufferInt.addr(), bufferInt.addr() + num, particles->getPosPtr());
	CUDA_CALL(cudaMemcpy(bufferInt.addr(), particles->getParticle2Cell(), sizeof(int) * num, cudaMemcpyDeviceToDevice));
	thrust::sort_by_key(thrust::device, bufferInt.addr(), bufferInt.addr() + num, particles->getVelPtr());

	thrust::fill(thrust::device, cellStart.addr(), cellStart.addr() + _cellSize.x * _cellSize.y * _cellSize.z + 1, 0);
	countingInCell_CUDA <<<(num - 1) / block_size + 1, block_size >>> (cellStart.addr(), particles->getParticle2Cell(), num);
	thrust::exclusive_scan(thrust::device, cellStart.addr(), cellStart.addr() + _cellSize.x * _cellSize.y * _cellSize.z + 1, cellStart.addr());
	return;
}

float SPHSystem::step()
{
	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	CUDA_CALL(cudaEventRecord(start, 0));

	neighborSearch(_fluids, cellStartFluid);
	try {
		_solver->step(_fluids, _boundaries, cellStartFluid, cellStartBoundary,
			_spaceSize, _cellSize, _sphCellLength, _sphSmoothingRadius,
			_dt, _sphRho0, _sphRhoBoundary, _sphStiff, _sphVisc, _sphG,
			_sphSurfaceTensionIntensity, _sphAirPressure);
		cudaDeviceSynchronize(); CHECK_KERNEL();
	}
	catch (const char* s) {
		std::cout << s << std::endl;
	}
	catch (...) {
		std::cout << "Unknown Exception at "<<__FILE__<<": line "<<__LINE__ << std::endl;
	}

	float milliseconds;
	CUDA_CALL(cudaEventRecord(stop, 0));
	CUDA_CALL(cudaEventSynchronize(stop));
	CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
	CUDA_CALL(cudaEventDestroy(start));
	CUDA_CALL(cudaEventDestroy(stop));
	return milliseconds;
}

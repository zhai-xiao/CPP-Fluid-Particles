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
#include <helper_math.h>
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"

__global__ void generate_dots_CUDA(float3* dot, float3* posColor, float3* pos, float* density, const int num)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;

	dot[i] = pos[i];
	if (density[i] < 0.75f)	{
		posColor[i] = make_float3(0.34f, 0.46f, 0.7f);
	}
	else if (density[i] < 1.0f) {
		const auto w = (density[i] - 0.75f) * 4.0f;
		posColor[i] = w * make_float3(0.9f) + (1 - w) * make_float3(0.34f, 0.46f, 0.7f);
	}
	else {
		auto w = (powf(density[i], 2) - 1.0f)*4.0f;
		w = fminf(w, 1.0f);
		posColor[i] = (1-w)*make_float3(0.9f) + w*make_float3(1.0f, 0.4f, 0.7f);
	}
}

extern "C" void generate_dots(float3* dot, float3* color, const std::shared_ptr<SPHParticles> particles) {
	generate_dots_CUDA <<<(particles->size() - 1) / block_size + 1, block_size >>> 
		(dot, color, particles->getPosPtr(), particles->getDensityPtr(), particles->size());
	cudaDeviceSynchronize(); CHECK_KERNEL();
	return;
}
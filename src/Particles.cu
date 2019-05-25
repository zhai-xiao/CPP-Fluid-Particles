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
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include "DArray.h"
#include "Particles.h"

void Particles::advect(float dt)
{	
	thrust::transform(thrust::device,
		pos.addr(), pos.addr() + size(),
		vel.addr(),
		pos.addr(),
		[dt]__host__ __device__(const float3& lhs, const float3& rhs) { return lhs + dt*rhs; }	
	);
}

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

const int block_size = 256;
#define EPSILON (1e-6f)
#define PI (3.14159265358979323846f)
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { printf("CUDA Error at %s:%d\t Error code = %d\n",__FILE__,__LINE__,x);}} while(0) 
//#define CUDA_CALL(x) do { x ;} while(0) 
#define CHECK_KERNEL(); 	{cudaError_t err = cudaGetLastError();if(err)printf("CUDA Error at %s:%d:\t%s\n",__FILE__,__LINE__,cudaGetErrorString(err));}
#define MAX_A (1000.0f)

namespace ThrustHelper {
	template<typename T>
	struct plus {
		T _a;
		plus(const T a) :_a(a) {}
		__host__ __device__
			T operator()(const T& lhs) const {
			return lhs + _a;
		}
	};

	template <typename T>
	struct abs_plus
	{
		__host__ __device__
			T operator()(const T& lhs, const T& rhs) const {
			return abs(lhs) + abs(rhs);
		}
	};
}
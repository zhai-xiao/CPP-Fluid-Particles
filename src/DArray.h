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
#include "global.h"

template <typename T>
class DArray {
	static_assert(
		std::is_same<T, float3>::value || std::is_same<T, float>::value || 
		std::is_same<T, int>::value, "DArray must be of int, float or float3.");
public:
	explicit DArray(const unsigned int length) :
		_length(length),
 		d_array([length]() {
		T* ptr; 
		CUDA_CALL(cudaMalloc((void**)& ptr, sizeof(T) * length));
		std::shared_ptr<T> t(new(ptr)T[length], [](T* ptr) {CUDA_CALL(cudaFree(ptr)); });
 		return t;
	}()) {
		this->clear();
	}

	DArray(const DArray&) = delete;
	DArray& operator=(const DArray&) = delete;

	T* addr(const int offset= 0) const {
		return d_array.get() + offset;
	}

	unsigned int length() const { return _length; }
	void clear()
	{ CUDA_CALL(cudaMemset(this->addr(), 0, sizeof(T) * this->length())); }
	
	~DArray() noexcept { }

private:
	const unsigned int _length;
	const std::shared_ptr<T> d_array;
};
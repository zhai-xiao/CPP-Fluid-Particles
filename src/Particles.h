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

class Particles {
public:
	explicit Particles::Particles(const std::vector<float3>& p)
		:pos(p.size()), vel(p.size()) {
		CUDA_CALL(cudaMemcpy(pos.addr(), &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));
	}

	Particles(const Particles&) = delete;
	Particles& operator=(const Particles&) = delete;

	unsigned int size() const {
		return pos.length();
	}
	float3* getPosPtr() const {
		return pos.addr();
	}
	float3* getVelPtr() const {
		return vel.addr();
	}
	const DArray<float3>& getPos() const {
		return pos;
	}

	void advect(float dt);

	virtual ~Particles() noexcept { }

protected:
	DArray<float3> pos;
	DArray<float3> vel;
};

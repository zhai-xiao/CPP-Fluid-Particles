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

class SPHParticles final : public Particles {
public:
	explicit SPHParticles::SPHParticles(const std::vector<float3>& p)
		:Particles(p),
		pressure(p.size()),
		density(p.size()),
		mass(p.size()),
		particle2Cell(p.size()) {
		CUDA_CALL(cudaMemcpy(pos.addr(), &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));
	}

	SPHParticles(const SPHParticles&) = delete;
	SPHParticles& operator=(const SPHParticles&) = delete;

	float* getPressurePtr() const {
		return pressure.addr();
	}
	const DArray<float>& getPressure() const {
		return pressure;
	}
	float* getDensityPtr() const {
		return density.addr();
	}
	const DArray<float>& getDensity() const {
		return density;
	}
	int* getParticle2Cell() const {
		return particle2Cell.addr();
	}
	float* getMassPtr() const {
		return mass.addr();
	}

	virtual ~SPHParticles() noexcept { }

protected:
	DArray<float> pressure;
	DArray<float> density;
	DArray<float> mass;
	DArray<int> particle2Cell; // lookup key
};

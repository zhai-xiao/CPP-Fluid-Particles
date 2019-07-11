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

class SPHSystem {
public:
	SPHSystem(
		std::shared_ptr<SPHParticles>& fluidParticles,
		std::shared_ptr<SPHParticles>& boundaryParticles,
		std::shared_ptr<BaseSolver>& solver,
		float3 spaceSize,
		float sphCellLength,
		float sphSmoothingRadius,
		float dt,
		float sphM0,
		float sphRho0,
		float sphRhoBoundary,
		float sphStiff,
		float sphVisc,
		float sphSurfaceTensionIntensity,
		float sphAirPressure,
		float3 sphG,
		int3 cellSize);
	SPHSystem(const SPHSystem&) = delete;
	SPHSystem& operator=(const SPHSystem&) = delete;

	float step();

	int size() const {
		return fluidSize();
	}
	int fluidSize() const {
		return (*_fluids).size();
	}
	int boundarySize() const {
		return (*_boundaries).size();
	}
	int totalSize() const {
		return (*_fluids).size() + (*_boundaries).size();
	}
	auto getFluids() const {
		return static_cast<const std::shared_ptr<SPHParticles>>(_fluids);
	}
	auto getBoundaries() const {
		return static_cast<const std::shared_ptr<SPHParticles>>(_boundaries);
	}
	~SPHSystem() noexcept { }
private:
	std::shared_ptr<SPHParticles> _fluids;
	const std::shared_ptr<SPHParticles> _boundaries;
	std::shared_ptr<BaseSolver> _solver;
	DArray<int> cellStartFluid;
	DArray<int> cellStartBoundary;
	const float3 _spaceSize;
	const float _sphSmoothingRadius;
	const float _sphCellLength;
	const float _dt;
	const float _sphRho0;
	const float _sphRhoBoundary;
	const float _sphStiff;
	const float3 _sphG;
	const float _sphVisc;
	const float _sphSurfaceTensionIntensity;
	const float _sphAirPressure;
	const int3 _cellSize;
	DArray<int> bufferInt;
	void computeBoundaryMass();
	void neighborSearch(const std::shared_ptr<SPHParticles>& particles, DArray<int>& cellStart);
};
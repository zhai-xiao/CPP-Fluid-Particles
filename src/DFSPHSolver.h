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

class DFSPHSolver final : public BasicSPHSolver {
public:
	virtual void step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, const float3 spaceSize,
		const int3 cellSize, const float cellLength, const float radius, const float dt,
		const float rho0, const float rhoB, const float stiff, const float visc, const float3 G,
		const float surfaceTensionIntensity, const float airPressure) override;
	explicit DFSPHSolver(const int num,
		const float defaultDensityErrorThreshold = 1e-3f,
		const float defaultDivergenceErrorThreshold = 1e-3f,
		const int defaultMaxIter = 20)
		:BasicSPHSolver(num),
		alpha(num),
		bufferFloat(num),
		bufferInt(num),
		error(num),
		denWarmStiff(num),
		densityErrorThreshold(defaultDensityErrorThreshold),
		divergenceErrorThreshold(defaultDivergenceErrorThreshold), 
		maxIter(defaultMaxIter){}
	virtual ~DFSPHSolver() {
		alpha.~DArray();
		bufferFloat.~DArray();
		error.~DArray();
		denWarmStiff.~DArray();
	}
protected:
	// overwrite and hide the project function in BasicSPHSolver
	// in project, correct density error from alpha
	virtual int project(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		const float rho0, const int3 cellSize, const float cellLength, const float radius, const float dt,
		const float errorThreshold, const int maxIter);
	virtual void computeDensityAlpha(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, 
		const int3 cellSize, const float cellLength, const float radius);
	virtual int correctDivergenceError(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		const float rho0, const int3 cellSize, const float cellLength, const float radius, const float dt,
		const float errorThreshold, const int maxIter);

private:
	DArray<float> alpha;
	DArray<float> bufferFloat;
	DArray<int> bufferInt;
	DArray<float> error;
	DArray<float> denWarmStiff;
	const float densityErrorThreshold;
	const float divergenceErrorThreshold;
	const int maxIter;
};
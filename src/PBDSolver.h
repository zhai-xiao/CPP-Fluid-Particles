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

class PBDSolver final : public BasicSPHSolver {
public:
	virtual void step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, float3 spaceSize,
		int3 cellSize, float cellLength, float radius, float dt,
		float rho0, float rhoB, float stiff, float visc, float3 G,
		float surfaceTensionIntensity, float airPressure) override;
	explicit PBDSolver(int num,
		int defaultMaxIter = 20,
		float defaultXSPH_c = 0.05f,
		float defaultRelaxation = 0.75f)
		:BasicSPHSolver(num),
		maxIter(defaultMaxIter),
		xSPH_c(defaultXSPH_c),
		relaxation(defaultRelaxation),
		bufferInt(num),
		fluidPosLast(num),
		bufferFloat3(num),
		bufferFloat(num) {}

	explicit PBDSolver(const std::shared_ptr<SPHParticles>& particles,
		int defaultMaxIter = 20,
		float defaultXSPH_c = 0.1f,
		float defaultRelaxation = 1.0f)
		:BasicSPHSolver(particles->size()),
		maxIter(defaultMaxIter),
		xSPH_c(defaultXSPH_c),
		relaxation(defaultRelaxation),
		bufferInt(particles->size()),
		fluidPosLast(particles->size()),
		bufferFloat3(particles->size()),
		bufferFloat(particles->size()) {
		initializePosLast(particles->getPos());
	}

	virtual ~PBDSolver() noexcept {	}

	void initializePosLast(const DArray<float3>& posFluid) {
		CUDA_CALL(cudaMemcpy(fluidPosLast.addr(), posFluid.addr(), sizeof(float3) * fluidPosLast.length(), cudaMemcpyDeviceToDevice));
		posLastInitialized = true;
	}

protected:
	void predict(std::shared_ptr<SPHParticles>& fluids, float dt, float3 spaceSize);

	// overwrite and hide the project function in BasicSPHSolver
	virtual int project(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
	                    const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
	                    float rho0, int3 cellSize, float3 spaceSize, float cellLength,
	                    float radius, int maxIter);

	// overwrite and hide the diffuse function in BasicSPHSolver, apply XSPH viscosity
	virtual void diffuse(std::shared_ptr<SPHParticles>& fluids, const DArray<int>& cellStartFluid,
	                     int3 cellSize, float cellLength, float rho0,
	                     float radius, float visc);

private:
	bool posLastInitialized = false;
	const int maxIter;
	const float xSPH_c;
	const float relaxation;
	DArray<int> bufferInt;
	DArray<float3> fluidPosLast;
	DArray<float3> bufferFloat3;
	DArray<float> bufferFloat;
	void updateNeighborhood(const std::shared_ptr<SPHParticles>& particles);
};
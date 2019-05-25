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

class BasicSPHSolver: public BaseSolver {
public:
	virtual void step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, float3 spaceSize,
		int3 cellSize, float cellLength, float radius, float dt,
		float rho0, float rhoB, float stiff, float visc, float3 G,
		float surfaceTensionIntensity, float airPressure) override;
	explicit BasicSPHSolver(int num) :bufferFloat3(num) {}
	virtual ~BasicSPHSolver() noexcept { }
protected:
	virtual void force(std::shared_ptr<SPHParticles>& fluids, float dt, float3 G) override final;
	virtual void advect(std::shared_ptr<SPHParticles>& fluids, float dt, float3 spaceSize) override final;
	virtual void project(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, float rho0, float stiff,
		int3 cellSize, float cellLength, float radius, float dt);
	virtual void diffuse(std::shared_ptr<SPHParticles>& fluids, const DArray<int>& cellStartFluid,
		int3 cellSize, float cellLength, float rho0,
		float radius, float visc, float dt);
	virtual void handleSurface(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		float rho0, float rhoB, int3 cellSize, float cellLength, float radius,
		float dt, float surfaceTensionIntensity, float airPressure);
private:
	DArray<float3> bufferFloat3;
	void computeDensity(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, int3 cellSize, float cellLength, float radius) const;
	void surfaceDetection(DArray<float3>& colorGrad, const std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		float rho0, float rhoB, int3 cellSize, float cellLength, float radius);
	void applySurfaceEffects(std::shared_ptr<SPHParticles>& fluids, const DArray<float3>& colorGrad,
		const DArray<int>& cellStartFluid, float rho0, int3 cellSize, float cellLength,
		float radius, float dt, float surfaceTensionIntensity, float airPressure);
};
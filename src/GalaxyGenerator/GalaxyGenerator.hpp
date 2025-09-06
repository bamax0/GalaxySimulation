#pragma once
#include "ExportDLL.hpp"
#include "Physics/Star.hpp"
#include <vector>
#include <iostream>

struct GenParam
{
    MassMs minMass, maxMass;
    DistanceLy min, max;

    SpeedScalarLyY minSpeed, maxSpeed;
    PosLy center{};
};


// GALAXY_SIM_DLL_EXPORT std::vector<Star> getSquareUnivers(const size_t& nbStars, const GenParam& param);

GALAXY_SIM_DLL_EXPORT Star* getSquareUniversFromGPU(const size_t& nbStars, const GenParam& param);
GALAXY_SIM_DLL_EXPORT Star* getPlummerGalaxyFromGPU(const size_t& nbStars, GenParam param);
GALAXY_SIM_DLL_EXPORT std::vector<Star> getPlummerGalaxyFromGPUToCpu(const size_t& nbStars, const GenParam& param);
GALAXY_SIM_DLL_EXPORT std::vector<Star> getSquareUniversFromGPUToCPU(const size_t& nbStars, const GenParam& param);
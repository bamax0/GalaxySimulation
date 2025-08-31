#pragma once
// #include "../external/glm-master/glm-master/glm/glm.hpp"
#include "dVec3.hpp"
#include "Vec3.hpp"




// Ly = Light year
// y = year
// Ms = solar mass 

// using Vec3 = Vec3;
// using dVec3 = dVec3;

// Custom unit
using MassMs = float;
using TimeY = float;
using DistanceLy = float;
using SpeedScalarLyY = float; 
using AccelScalarLyY2 = float; 

using PosLy = Vec3;
using SpeedLs = Vec3;
using AccelLyY2 = Vec3;


// standart unit
using MassKg = double;
using AngleRadian = double;
using AngleDegree = double;
using TimeS = double;
using DistanceM = double;
using ForceScalarN = double;
using SpeedScalarMS = double;
using AccemScalarMS2 = double;

using PosM = dVec3;
using ForceN = dVec3;
using SpeedMS = dVec3;
using AccemMS2 = dVec3;

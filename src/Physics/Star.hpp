#pragma once
#include "Physics/TypeDef.hpp"

struct Star
{
	PosLy position;
	SpeedLs speed;
	AccelLyY2 acceleration;
	MassMs mass{};
};
#pragma once
#include "ExportDLL.hpp"
#include "Physics/Star.hpp"
#include "Bbox.hpp"

class GALAXY_SIM_DLL_EXPORT ITreeNode
{
public:
	virtual void reset(const Bbox& bbox) = 0;
	virtual void startInserting() = 0;
	virtual void appendStar(const Star& star, size_t depth = 0) = 0;
	virtual void endInserting() = 0;
	virtual Vec3 computeTotalForce(const Star& star) const = 0;
};

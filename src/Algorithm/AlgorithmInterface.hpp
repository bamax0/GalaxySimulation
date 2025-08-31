#pragma once
#include <vector>
#include "Physics/Star.hpp"
#include "ExportDLL.hpp"
namespace GalaxySim
{
	class GALAXY_SIM_DLL_EXPORT AlgorithmInterface
	{
	public:
		AlgorithmInterface() {}

		virtual ~AlgorithmInterface() {};

		virtual void init(const std::vector<Star>& stars) = 0;

		virtual void next(TimeY dt) = 0;

		virtual std::vector<Star> getStars() const = 0;
	};
}
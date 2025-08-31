#pragma once
#include "Algorithm/AlgorithmInterface.hpp"
#include "ExportDLL.hpp"

namespace GalaxySim
{
	class GALAXY_SIM_DLL_EXPORT CpuSimpleAlgorithm : public AlgorithmInterface
	{
	public:
		CpuSimpleAlgorithm() : AlgorithmInterface() {};

		~CpuSimpleAlgorithm() override = default;

		void init(const std::vector<Star>& stars) override;

		void next(TimeY dt) override;

		std::vector<Star> getStars() const override;

	private:
		std::vector<Star> m_stars;
		size_t m_nbStars{ 0u };
	};
}
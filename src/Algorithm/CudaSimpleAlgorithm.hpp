#pragma once
#include "Algorithm/AlgorithmInterface.hpp"
#include "Physics/function.hpp"
#include "ExportDLL.hpp"

namespace GalaxySim
{
	class GALAXY_SIM_DLL_EXPORT CudaSimpleAlgorithm : public AlgorithmInterface
	{
	public:
		CudaSimpleAlgorithm() : AlgorithmInterface() {};

		~CudaSimpleAlgorithm() override;

		void initWithGPU(size_t nbStars, Star* stars);

		void init(const std::vector<Star>& stars) override;

		void next(TimeY dt) override;

		std::vector<Star> getStars() const override;

		const Star* getCudaStars() const;

	private:
		Star* m_gpuStarsPtr{ nullptr };
		Vec3* m_gpuForcesPtr{ nullptr };
		size_t m_nbStars{ 0u };
	};
}
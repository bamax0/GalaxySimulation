#pragma once
#include "Algorithm/AlgorithmInterface.hpp"
#include "ExportDLL.hpp"

namespace GalaxySim
{
	/* WIP (pas assez prometteur comme tecnique, abandonné pour le moment */
	class GALAXY_SIM_DLL_EXPORT CudaGFieldAlgorithm : public AlgorithmInterface
	{
	public:
		CudaGFieldAlgorithm(size_t fieldSize) : AlgorithmInterface(), m_fieldSize(fieldSize){};

		virtual ~CudaGFieldAlgorithm() override;

		void initWithGPU(size_t nbStars, Star* stars);

		void init(const std::vector<Star>& stars) override;

		void next(TimeY dt) override;

		std::vector<Star> getStars() const override;

		const Star* getCudaStars() const;

	private:
		Star* m_gpuStarsPtr{ nullptr };
		Vec3* m_gpuFieldPtr{ nullptr };
		size_t m_nbStars{ 0u };
		size_t m_fieldSize{ 0u };
	};
}
#pragma once
#include "Nodes/TreeNode.hpp"
#include "AlgorithmInterface.hpp"
#include <Algorithm/Nodes/FirstTreeNode.hpp>
#include <Algorithm/Nodes/CudaTreeNode.hpp>

namespace GalaxySim
{
	class GALAXY_SIM_DLL_EXPORT CudaBarnesHutAlgorithm : public AlgorithmInterface
	{
	public:
		CudaBarnesHutAlgorithm(const Bbox& bbox)
		{
			m_rootNode  = new CudaTreeNode(bbox);
			m_firstNode = new FirstTreeNode<8>(static_cast<SimplifiedTreeNode<8>*>(m_rootNode));
		}

		~CudaBarnesHutAlgorithm();

		void init(const std::vector<Star>& stars) override;

		void next(TimeY dt) override;

		std::vector<Star> getStars() const override;

		void initWithGPU(size_t nbStars, Star* stars);
	private:
		Bbox computeBbox() const;

		CudaTreeNode* m_rootNode;
		FirstTreeNode<8>* m_firstNode;
		
		Star* m_gpuStarsPtr{ nullptr };
		Vec3* m_gpuForcesPtr{ nullptr };
		size_t m_nbStars{ 0u };
	};
}
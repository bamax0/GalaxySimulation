#pragma once
#include "Nodes/TreeNode.hpp"
#include "AlgorithmInterface.hpp"

namespace GalaxySim
{
	class GALAXY_SIM_DLL_EXPORT BarnesHutAlgorithm : public AlgorithmInterface
	{
	public:
		BarnesHutAlgorithm(ITreeNode* rootNode) : m_rootNode(rootNode) {}

		~BarnesHutAlgorithm() { delete m_rootNode; };

		void init(const std::vector<Star>& stars) override;

		void next(TimeY dt) override;

		std::vector<Star> getStars() const override
		{
			return m_stars;
		}
	private:
		Bbox computeBbox() const;

		std::vector<Star> m_stars;

		ITreeNode* m_rootNode;
	};
}
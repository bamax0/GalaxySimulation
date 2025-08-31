#pragma once
#include "ExportDLL.hpp"
#include "OctreeNode.hpp"
#include "CudaLastTreeNode.hpp"
#include <shared_mutex>
#include <mutex>
#include <set>

struct CudaTreeNodeFlat;

struct OctreePtr
{
	CudaTreeNodeFlat* nodes;
	CudaLastTreeNodeFlat* lastNodes;
};

/* Structure pour représenté un arbre (pour mettre les noeuds à plat pour cuda)*/
struct CudaTreeNodeFlat
{
	Vec3 centerOfMass;
	MassMs mass;

	float barnesHutCst{ 0.f }; // m_bbox.size * 2.f / s_theta
	// uint64_t childIndexList[8u]{0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
	int64_t c0{0};
	int64_t c1{0};
	int64_t c2{0};
	int64_t c3{0};
	int64_t c4{0};
	int64_t c5{0};
	int64_t c6{0};
	int64_t c7{0};

	__device__ __host__ __inline__ int64_t getChildIdx(unsigned char i) const
	{
		if (i == 0u)
		{
			return c0;
		}
		else if (i == 1u)
		{
			return c1;
		}
		else if (i == 2u)
		{
			return c2;
		}
		else if (i == 3u)
		{
			return c3;
		}
		else if (i == 4u)
		{
			return c4;
		}
		else if (i == 5u)
		{
			return c5;
		}
		else if (i == 6u)
		{
			return c6;
		}
		return c7;
	}

	__device__ __host__ __inline__ void setChildIdx(unsigned char i, int64_t childIdx)
	{
		if (i == 0u)
		{
			c0 = childIdx;
			return;
		}
		else if (i == 1u)
		{
			c1 = childIdx;
			return;
		}
		else if (i == 2u)
		{
			c2 = childIdx;
			return;
		}
		else if (i == 3u)
		{
			c3 = childIdx;
			return;
		}
		else if (i == 4u)
		{
			c4 = childIdx;
			return;
		}
		else if (i == 5u)
		{
			c5 = childIdx;
			return;
		}
		else if (i == 6u)
		{
			c6 = childIdx;
			return;
		}
		c7 = childIdx;
		return;
	}


	__device__ __host__ __inline__ Vec3 computeTotalForce(const OctreePtr octreeNodes, const Star star, int stack = 0) const
	{
		//printf("stack = %i\n", stack);

		if (centerOfMass == star.position)
			return  { 0, 0, 0 };

		float r = Vec3::distance(centerOfMass, star.position);
		
		bool a = barnesHutCst  < r;
		if (a) 
		{
			// printf("----------- Fin calcule radius: %.1f < %.1f = %f  || stack = %i\n",
			// 	barnesHutCst, r, a ? 1.f : 0.f, stack);
			return computeForce(star.position, star.mass, centerOfMass, mass, r);
		}

		bool hasChild{false};
		Vec3 force{ 0.f, 0.f, 0.f };
		for (unsigned char  i{0u}; i<8u; ++i)
		{
			int64_t  childIdx = getChildIdx(i);
			if (childIdx > 0)
			{
				force += octreeNodes.nodes[childIdx].computeTotalForce(octreeNodes, star, stack + 1);
				hasChild = true;
			}
			else if (childIdx < 0)
			{
				force += octreeNodes.lastNodes[-1 - childIdx].computeTotalForce(octreeNodes.lastNodes, star);
				hasChild = true;
			}
		}
		if(!hasChild) return computeForce(star.position, star.mass, centerOfMass, mass, r);
		return force;
	}

	__device__ __host__ __inline__ Vec3 computeTotalForceWithoutRec(const OctreePtr octreeNodes, const Star star) const
	{

		Vec3 force{ 0.f, 0.f, 0.f };

		constexpr int MAX_STACK_SIZE = 64;
		const CudaTreeNodeFlat* stack[MAX_STACK_SIZE]{ nullptr };
		int stackIndex{ 1u };
		stack[0] = this;

		uint32_t tps{ 0u };
		while (stackIndex > 0)
		{
			++tps;
			--stackIndex;
			const CudaTreeNodeFlat* currNode = stack[stackIndex];

			if (currNode->centerOfMass == star.position)
				continue;

			float r = Vec3::distance(currNode->centerOfMass, star.position);

			if (currNode->barnesHutCst < r)
			{
				// printf("----------- Fin calcule radius: %.1f < %.1f = %f  || stack Idx = %i\n",
				// 	barnesHutCst, r, a ? 1.f : 0.f, stackIndex);
				force += computeForce(star.position, star.mass, currNode->centerOfMass, currNode->mass, r);
				continue;
			}


			bool hasChild{ false };
			for (uint64_t i{ 0u }; i < 8u; ++i)
			{
				int64_t  childIdx = currNode->getChildIdx(i);
				if (childIdx > 0)
				{
					stack[stackIndex] = &octreeNodes.nodes[childIdx];
					++stackIndex;
					hasChild = true;
				}
				else if (childIdx < 0)
				{
					force += octreeNodes.lastNodes[-1-childIdx].computeTotalForceWithoutRec(octreeNodes.lastNodes, star);
					hasChild = true;
				}
			}
			if (!hasChild) 
			{
				force += computeForce(star.position, star.mass, currNode->centerOfMass, currNode->mass, r);
			}
		}
		return force;
	}

};

using  CudaTreeNodeFlatPtr = std::unique_ptr<CudaTreeNodeFlat>;

class GALAXY_SIM_DLL_EXPORT CudaTreeNode : public SimplifiedTreeNode<8>
{
public:
	CudaTreeNode(const Bbox& bbox)
		: SimplifiedTreeNode<8>(bbox),
		m_mutex( std::make_shared<std::mutex>() ),
		m_nodes(std::make_shared<std::vector<CudaTreeNodeFlat>>(
			std::vector<CudaTreeNodeFlat>{CudaTreeNodeFlat{}})),
		m_lastNodes(std::make_shared<std::vector<CudaLastTreeNodeFlat>>(
			std::vector<CudaLastTreeNodeFlat>{})),
		m_nodeIdx{0u}
	{
	}

	~CudaTreeNode()
	{
		reset({});
	}


	CudaTreeNode(
		std::shared_ptr<std::mutex> mutex, 
		std::shared_ptr<std::vector<CudaTreeNodeFlat>> nodes,
		std::shared_ptr<std::vector<CudaLastTreeNodeFlat>> lastNodes,
		size_t idx, const Bbox& bbox)
		: SimplifiedTreeNode<8>(bbox), m_nodes(nodes), m_mutex(mutex), m_nodeIdx(idx), m_lastNodes(lastNodes)
	{
	}

	virtual void reset(const Bbox& bbox) override;
	void update(const PosLy& centerMass, MassMs mass) override
	{
		m_nodes->at(m_nodeIdx).centerOfMass = centerMass;
		m_nodes->at(m_nodeIdx).mass = mass;
	}
	PosLy getCenterOfMass() const override 
	{ 
		return  m_nodes->at(m_nodeIdx).centerOfMass;
	}
	MassMs getMass() const override { 
		return m_nodes->at(m_nodeIdx).mass;
	}

	OctreePtr createCudaPtr() const;
	bool fillIndex(std::set<size_t>& idxVec) const
	{
		if (idxVec.find(m_nodeIdx) == idxVec.end())
		{
			idxVec.insert(m_nodeIdx);
			for (unsigned short i{ 0u }; i < 8u; ++i)
			{
				const CudaTreeNode* c = dynamic_cast<const CudaTreeNode*>(getChild(i));
				if (c != nullptr)
				{
					if (!c->fillIndex(idxVec)) return false;
				}
			}

			return true;
		}
		return false;
	}

	using SimplifiedTreeNode<8>::startInserting;
	using SimplifiedTreeNode<8>::endInserting;
	using SimplifiedTreeNode<8>::appendStar;
	using SimplifiedTreeNode<8>::computeTotalForce;
protected:

	unsigned char getIndex(const PosLy& pos) const override;
	Bbox getSubBbox(unsigned char id) const override;
	SimplifiedTreeNode<8>* createChild(const Bbox& bbox, unsigned char idx) const override;
	ITreeNode* createLastTreeNode(const PosLy& pos, MassMs mass, unsigned char idx) const override;

	std::shared_ptr<std::vector<CudaTreeNodeFlat>> m_nodes;
	std::shared_ptr<std::mutex> m_mutex;

	std::shared_ptr<std::vector<CudaLastTreeNodeFlat>> m_lastNodes;
	size_t m_nodeIdx{ 0u };

	using SimplifiedTreeNode<8>::createChild;
};
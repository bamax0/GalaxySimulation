#pragma once
#include "ExportDLL.hpp"
#include "OctreeNode.hpp"
#include <shared_mutex>
#include <mutex>
#include <set>

/* Structure pour représenté un arbre (pour mettre les noeuds à plat pour cuda)*/
struct CudaLastTreeNodeFlat
{
	Vec3 starPos;
	MassMs starMass;
	uint64_t childIdx{0u};

	__device__ __host__ __inline__ Vec3 computeTotalForce(const CudaLastTreeNodeFlat* lastNodes, 
		const Star star) const
	{

		DistanceLy r = Vec3::distance(starPos, star.position);
		Vec3 force = computeForce(star.position, star.mass, starPos, starMass, r);
		if (childIdx > 0u)
		{
			force += lastNodes[childIdx].computeTotalForce(lastNodes, star);
		}
		return force;
	}

	__device__ __host__ __inline__ Vec3 computeTotalForceWithoutRec(const CudaLastTreeNodeFlat* lastNodes,
		const Star star) const
	{
		DistanceLy r = Vec3::distance(starPos, star.position);
		Vec3 force = computeForce(star.position, star.mass, starPos, starMass, r);

		const CudaLastTreeNodeFlat* node = this;
		while (node->childIdx > 0u)
		{
			//if (node->childIdx > 620000) printf("%i\n", node->childIdx);
			node = &lastNodes[node->childIdx];

			r = Vec3::distance(node->starPos, star.position);
			force += computeForce(star.position, star.mass, node->starPos, node->starMass, r);
		}
		// printf("%.3f \n", force.length());
		return force;
	}
};


class GALAXY_SIM_DLL_EXPORT CudaLastTreeNode : public ITreeNode
{
public:


	CudaLastTreeNode(
		std::shared_ptr<std::mutex> mutex,
		std::shared_ptr<std::vector<CudaLastTreeNodeFlat>> nodes, 
		const Vec3& starPos, MassMs mass);

	virtual ~CudaLastTreeNode() override
	{
		reset({});
	}
	virtual void reset(const Bbox& bbox) override {/*A voir si utile*/ }

	void startInserting() override {};
	void endInserting() override {};
	void appendStar(const Star& star, size_t depth = 0) override;

	Vec3 computeTotalForce(const Star& star) const override { return Vec3(); } 

	size_t getIndex() const { return m_nodeFirstIdx; };

private:
	std::shared_ptr<std::vector<CudaLastTreeNodeFlat>> m_nodes;
	std::shared_ptr<std::mutex> m_mutex;
	size_t m_nodeFirstIdx{ 0u };
	size_t m_nodeLastIdx{ 0u };
};